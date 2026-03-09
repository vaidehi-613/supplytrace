import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, avg, count, sum as spark_sum,
    current_timestamp, regexp_replace, upper, trim,
    broadcast, round as spark_round
)
from delta import configure_spark_with_delta_pip

print("🚀 Starting SupplyTrace Silver Layer Transform...")

builder = (SparkSession.builder
    .appName("SupplyTrace-Silver-Master")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"✅ Spark {spark.version} ready\n")

# ── Paths ─────────────────────────────────────────────────────────
BRONZE_SHIPPING   = "data/bronze/shipping_events"
BRONZE_SUPPLIER   = "data/bronze/supplier_master"
BRONZE_LPI        = "data/bronze/supplier_risk"
BRONZE_WEATHER    = "data/bronze/weather_events"
BRONZE_COMMODITY  = "data/bronze/commodity_prices"
BRONZE_NEWS       = "data/bronze/news_disruptions"
SILVER_PATH       = "data/silver/supply_chain_master"

# ════════════════════════════════════════════════════════════════
# STEP 1: Load shipping_enriched (already has delay features)
# ════════════════════════════════════════════════════════════════
print("📂 Step 1: Loading Bronze tables...")

df_shipping = spark.read.format("delta").load(BRONZE_SHIPPING)

# Re-engineer delay features (in case shipping_enriched not used)
df_shipping = (df_shipping
    .withColumn("delay_days",
        col("Days_for_shipping_real") - col("Days_for_shipment_scheduled"))
    .withColumn("is_delayed",
        when(col("delay_days") > 0, 1).otherwise(0))
    .withColumn("route_key",
        col("Order_Country"))
    .withColumn("profit_margin_flag",
        when(col("Order_Profit_Per_Order") < 0, 1).otherwise(0))
)
print(f"   Shipping rows: {df_shipping.count():,}")

# ════════════════════════════════════════════════════════════════
# STEP 2: Join Supplier Master → assign each order a supplier
# ════════════════════════════════════════════════════════════════
print("\n⚙️  Step 2: Joining Supplier Master...")

df_supplier = spark.read.format("delta").load(BRONZE_SUPPLIER)
print(f"   Supplier master rows: {df_supplier.count():,}")

# Each order gets a supplier deterministically by Order_Id mod 2000
df_shipping = df_shipping.withColumn(
    "supplier_id_join",
    (col("Order_Id") % 2000).cast("int")
)
df_supplier_slim = df_supplier.select(
    col("supplier_id").cast("int").alias("supplier_id_key"),
    col("supplier_name"),
    col("tier").alias("supplier_tier"),
    col("region").alias("supplier_region"),
    col("country_code").alias("supplier_country_code"),
    col("lead_time_days"),
    col("quality_score"),
    col("esg_score"),
    col("risk_notes").alias("supplier_risk_notes"),
)

df_joined = df_shipping.join(
    broadcast(df_supplier_slim),
    df_shipping["supplier_id_join"] == df_supplier_slim["supplier_id_key"],
    "left"
).drop("supplier_id_join", "supplier_id_key")

print(f"   After supplier join: {df_joined.count():,} rows")

# ════════════════════════════════════════════════════════════════
# STEP 3: Join LPI Supplier Risk → country logistics score
# ════════════════════════════════════════════════════════════════
print("\n⚙️  Step 3: Joining World Bank LPI scores...")

df_lpi = spark.read.format("delta").load(BRONZE_LPI)
df_lpi_slim = df_lpi.select(
    col("country").alias("lpi_country"),
    col("lpi_score"),
    col("logistics_risk_tier"),
    col("infrastructure_score"),
    col("timeliness_score"),
)

# Normalize country names for join
df_joined = df_joined.withColumn(
    "order_country_clean", trim(col("Order_Country"))
)
df_lpi_slim = df_lpi_slim.withColumn(
    "lpi_country_clean", trim(col("lpi_country"))
)

df_joined = df_joined.join(
    broadcast(df_lpi_slim),
    df_joined["order_country_clean"] == df_lpi_slim["lpi_country_clean"],
    "left"
).drop("lpi_country", "lpi_country_clean", "order_country_clean")

# Feature: supplier_risk_flag
df_joined = df_joined.withColumn(
    "supplier_risk_flag",
    when(col("logistics_risk_tier").isin(
        "Very High Risk", "High Risk"), 1).otherwise(0)
)

lpi_matched = df_joined.filter(col("lpi_score").isNotNull()).count()
print(f"   LPI matched rows: {lpi_matched:,}")

# ════════════════════════════════════════════════════════════════
# STEP 4: Weather Risk Score per US State
# ════════════════════════════════════════════════════════════════
print("\n⚙️  Step 4: Computing weather risk scores...")

df_weather = spark.read.format("delta").load(BRONZE_WEATHER)

# Aggregate: avg severity per state
df_weather_agg = (df_weather
    .groupBy("state")
    .agg(
        avg("severity_score").alias("avg_weather_severity"),
        count("event_id").alias("storm_event_count"),
        spark_sum(when(col("severity_score") >= 4, 1).otherwise(0))
            .alias("high_critical_storm_count")
    )
    .withColumn("weather_risk_score",
        spark_round(col("avg_weather_severity"), 2))
    .withColumn("state_upper", upper(trim(col("state"))))
)
print(f"   Weather aggregated for {df_weather_agg.count()} states")

# Join on Customer_State (USCA orders)
df_joined = df_joined.withColumn(
    "customer_state_upper", upper(trim(col("Customer_State")))
)
df_joined = df_joined.join(
    broadcast(df_weather_agg.select(
        "state_upper", "weather_risk_score",
        "storm_event_count", "high_critical_storm_count"
    )),
    df_joined["customer_state_upper"] == df_weather_agg["state_upper"],
    "left"
).drop("state_upper", "customer_state_upper")

# Fill nulls for non-US orders
df_joined = df_joined.fillna({
    "weather_risk_score": 1.0,
    "storm_event_count": 0,
    "high_critical_storm_count": 0
})

# ════════════════════════════════════════════════════════════════
# STEP 5: Commodity Shock Flag
# ════════════════════════════════════════════════════════════════
print("\n⚙️  Step 5: Computing commodity shock flag...")

df_commodity = spark.read.format("delta").load(BRONZE_COMMODITY)

# Get latest oil price and flag if high (above $85)
try:
    df_oil = (df_commodity
        .filter(col("oil_price_usd").isNotNull())
        .orderBy(col("date").desc())
        .limit(1)
        .select(col("oil_price_usd").cast("double").alias("latest_oil_price"))
    )
    latest_oil = df_oil.collect()[0]["latest_oil_price"]
    print(f"   Latest oil price: ${latest_oil:.2f}")
    commodity_shock = 1 if latest_oil and latest_oil > 85 else 0
except Exception as e:
    print(f"   ⚠️  Could not get oil price: {e}, defaulting to 0")
    commodity_shock = 0

df_joined = df_joined.withColumn(
    "commodity_shock_flag", lit(commodity_shock)
)
print(f"   Commodity shock flag: {commodity_shock}")

# ════════════════════════════════════════════════════════════════
# STEP 6: News Risk Score
# ════════════════════════════════════════════════════════════════
print("\n⚙️  Step 6: Computing news risk level...")

df_news = spark.read.format("delta").load(BRONZE_NEWS)

# Score news by keywords in title
df_news_scored = df_news.withColumn(
    "news_risk_score",
    when(col("title").rlike(
        "(?i)(crisis|critical|severe|collapse|shutdown|blockage|strike)"), 3)
    .when(col("title").rlike(
        "(?i)(disruption|delay|shortage|congestion|surge|risk|warning)"), 2)
    .otherwise(1)
)

# Overall news risk level = avg score across all articles
news_avg = df_news_scored.agg(
    avg("news_risk_score").alias("avg_news_risk")
).collect()[0]["avg_news_risk"]

news_level = (
    "HIGH"   if news_avg >= 2.5 else
    "MEDIUM" if news_avg >= 1.5 else
    "LOW"
)
print(f"   Avg news risk score: {news_avg:.3f} → {news_level}")

df_joined = df_joined.withColumn(
    "news_risk_level", lit(news_level)
).withColumn(
    "news_risk_score_global", lit(float(news_avg))
)

# ════════════════════════════════════════════════════════════════
# STEP 7: Route Delay Rate
# ════════════════════════════════════════════════════════════════
print("\n⚙️  Step 7: Computing route delay rates...")

df_route_stats = (df_joined
    .groupBy("Order_Country", "Order_Region", "Shipping_Mode")
    .agg(
        avg("is_delayed").alias("route_delay_rate"),
        avg("delay_days").alias("route_avg_delay_days"),
        count("Order_Id").alias("route_order_count")
    )
)

df_joined = df_joined.join(
    df_route_stats,
    on=["Order_Country", "Order_Region", "Shipping_Mode"],
    how="left"
)
print(f"   Route delay rates computed for {df_route_stats.count()} routes")

# ════════════════════════════════════════════════════════════════
# STEP 8: Final Silver schema — select key columns
# ════════════════════════════════════════════════════════════════
print("\n📋 Step 8: Building final Silver schema...")

df_silver = (df_joined.select(
    # Order identifiers
    col("Order_Id").alias("order_id"),
    col("Order_Date").alias("order_date"),
    col("Shipping_Date").alias("shipping_date"),
    col("Order_Country").alias("order_country"),
    col("Order_Region").alias("order_region"),
    col("Market").alias("market"),
    col("Shipping_Mode").alias("shipping_mode"),
    col("Customer_Segment").alias("customer_segment"),
    col("Product_Name").alias("product_name"),
    # Target variable
    col("Late_delivery_risk").alias("late_delivery_risk"),
    col("Delivery_Status").alias("delivery_status"),
    # Engineered features
    col("delay_days"),
    col("is_delayed"),
    col("profit_margin_flag"),
    col("route_delay_rate"),
    col("route_avg_delay_days"),
    col("route_order_count"),
    # Supplier features
    col("supplier_name"),
    col("supplier_tier"),
    col("supplier_region"),
    col("lead_time_days"),
    col("quality_score").alias("supplier_quality_score"),
    col("esg_score").alias("supplier_esg_score"),
    col("supplier_risk_flag"),
    # LPI features
    col("lpi_score"),
    col("logistics_risk_tier"),
    col("infrastructure_score"),
    col("timeliness_score"),
    # Weather features
    col("weather_risk_score"),
    col("storm_event_count"),
    col("high_critical_storm_count"),
    # Commodity features
    col("commodity_shock_flag"),
    # News features
    col("news_risk_level"),
    col("news_risk_score_global"),
    # Financial
    col("Sales").alias("sales"),
    col("Order_Profit_Per_Order").alias("order_profit"),
    # Metadata
    current_timestamp().alias("_silver_processed_at"),
    lit("1.0").alias("_pipeline_version"),
)
.fillna({
    "lpi_score": 3.0,
    "logistics_risk_tier": "Medium Risk",
    "infrastructure_score": 3.0,
    "timeliness_score": 3.0,
    "supplier_risk_flag": 0,
    "route_delay_rate": 0.5,
    "route_avg_delay_days": 0.0,
})
)

total = df_silver.count()
print(f"   Silver rows: {total:,}")
print(f"   Silver columns: {len(df_silver.columns)}")

# ════════════════════════════════════════════════════════════════
# STEP 9: Quality summary
# ════════════════════════════════════════════════════════════════
print("\n🔍 Step 9: Silver quality summary...")
delayed       = df_silver.filter(col("is_delayed") == 1).count()
high_risk_sup = df_silver.filter(col("supplier_risk_flag") == 1).count()
shock         = df_silver.filter(col("commodity_shock_flag") == 1).count()

print(f"   Delayed orders:         {delayed:,}  ({delayed/total*100:.1f}%)")
print(f"   High-risk supplier:     {high_risk_sup:,}  ({high_risk_sup/total*100:.1f}%)")
print(f"   Commodity shock orders: {shock:,}  ({shock/total*100:.1f}%)")
print(f"   News risk level:        {news_level}")

print("\n�� Delay rate by Market:")
df_silver.groupBy("market").agg(
    avg("is_delayed").alias("delay_rate"),
    count("order_id").alias("orders")
).orderBy("delay_rate", ascending=False).show()

print("\n📊 Delay rate by Shipping Mode:")
df_silver.groupBy("shipping_mode").agg(
    avg("is_delayed").alias("delay_rate"),
    avg("delay_days").alias("avg_delay_days")
).orderBy("delay_rate", ascending=False).show()

print("\n👀 Sample silver rows (engineered features):")
df_silver.select(
    "order_id", "order_country", "shipping_mode",
    "delay_days", "is_delayed", "supplier_risk_flag",
    "weather_risk_score", "commodity_shock_flag",
    "news_risk_level", "lpi_score", "route_delay_rate"
).show(5, truncate=False)

# ════════════════════════════════════════════════════════════════
# STEP 10: Write Silver Delta table
# ════════════════════════════════════════════════════════════════
print(f"\n💾 Writing Silver Delta table → {SILVER_PATH}")
(df_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("market")
    .save(SILVER_PATH)
)

df_verify = spark.read.format("delta").load(SILVER_PATH)
print(f"✅ Verified: {df_verify.count():,} rows in Silver Delta table")
print(f"   Partitioned by: market")

print("\n" + "="*60)
print("🎉 SILVER LAYER COMPLETE!")
print(f"   Rows:    {df_verify.count():,}")
print(f"   Columns: {len(df_verify.columns)}")
print(f"   Path:    {SILVER_PATH}")
print(f"   Features engineered: 8")
print("="*60)

spark.stop()

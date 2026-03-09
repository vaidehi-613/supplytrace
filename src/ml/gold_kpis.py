import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, sum as spark_sum, round as spark_round,
    when, lit, current_timestamp, max as spark_max, min as spark_min,
    desc
)
from delta import configure_spark_with_delta_pip

print("🚀 Starting SupplyTrace Gold Layer — KPI Aggregations...")

builder = (SparkSession.builder
    .appName("SupplyTrace-Gold-KPIs")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"✅ Spark {spark.version} ready\n")

SILVER_PATH      = "data/silver/supply_chain_master"
GOLD_PREDICTIONS = "data/gold/risk_predictions"
GOLD_KPI_PATH    = "data/gold/kpi_summary"
GOLD_ROUTE_PATH  = "data/gold/route_risk"
GOLD_SUPPLIER_PATH = "data/gold/supplier_leaderboard"
GOLD_MARKET_PATH = "data/gold/market_dashboard"

df = spark.read.format("delta").load(SILVER_PATH)
df_pred = spark.read.format("delta").load(GOLD_PREDICTIONS)
print(f"Silver rows: {df.count():,}  |  Prediction rows: {df_pred.count():,}\n")

# ════════════════════════════════════════════════════════════
# KPI 1: Overall Pipeline Summary
# ════════════════════════════════════════════════════════════
print("📊 KPI 1: Overall Pipeline Summary...")
total         = df.count()
total_delayed = df.filter(col("is_delayed") == 1).count()
total_sales   = df.agg(spark_sum("sales")).collect()[0][0]
total_profit  = df.agg(spark_sum("order_profit")).collect()[0][0]
avg_delay     = df.agg(avg("delay_days")).collect()[0][0]

print(f"   Total Orders:        {total:,}")
print(f"   Total Delayed:       {total_delayed:,}  ({total_delayed/total*100:.1f}%)")
print(f"   Total Sales:         ${total_sales:,.0f}")
print(f"   Total Profit:        ${total_profit:,.0f}")
print(f"   Avg Delay Days:      {avg_delay:.2f}")

# ════════════════════════════════════════════════════════════
# KPI 2: Market Dashboard
# ════════════════════════════════════════════════════════════
print("\n📊 KPI 2: Market Dashboard...")
df_market = (df.groupBy("market")
    .agg(
        count("order_id").alias("total_orders"),
        spark_round(avg("is_delayed") * 100, 1).alias("delay_rate_pct"),
        spark_round(avg("delay_days"), 2).alias("avg_delay_days"),
        spark_round(spark_sum("sales"), 0).alias("total_sales_usd"),
        spark_round(avg("lpi_score"), 2).alias("avg_lpi_score"),
        spark_round(avg("weather_risk_score"), 2).alias("avg_weather_risk"),
    )
    .orderBy("delay_rate_pct", ascending=False)
    .withColumn("_processed_at", current_timestamp())
)
df_market.show(truncate=False)

(df_market.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").save(GOLD_MARKET_PATH))
print(f"   ✅ Saved to {GOLD_MARKET_PATH}")

# ════════════════════════════════════════════════════════════
# KPI 3: Route Risk Leaderboard (Top 20 riskiest routes)
# ════════════════════════════════════════════════════════════
print("\n📊 KPI 3: Route Risk Leaderboard...")
df_route = (df.groupBy("order_country", "order_region", "shipping_mode")
    .agg(
        count("order_id").alias("order_count"),
        spark_round(avg("is_delayed") * 100, 1).alias("delay_rate_pct"),
        spark_round(avg("delay_days"), 2).alias("avg_delay_days"),
        spark_round(avg("lpi_score"), 2).alias("avg_lpi_score"),
        spark_round(avg("weather_risk_score"), 2).alias("avg_weather_risk"),
    )
    .filter(col("order_count") >= 50)
    .orderBy("delay_rate_pct", ascending=False)
    .withColumn("risk_rank", lit(None).cast("int"))
    .withColumn("_processed_at", current_timestamp())
)

print(f"   Top 10 riskiest routes:")
df_route.show(10, truncate=False)

(df_route.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").save(GOLD_ROUTE_PATH))
print(f"   ✅ Saved to {GOLD_ROUTE_PATH}")

# ════════════════════════════════════════════════════════════
# KPI 4: Supplier Leaderboard
# ════════════════════════════════════════════════════════════
print("\n📊 KPI 4: Supplier Leaderboard...")
df_supplier = (df.groupBy("supplier_name", "supplier_tier", "supplier_region")
    .agg(
        count("order_id").alias("order_count"),
        spark_round(avg("is_delayed") * 100, 1).alias("delay_rate_pct"),
        spark_round(avg("delay_days"), 2).alias("avg_delay_days"),
        spark_round(avg("supplier_quality_score"), 2).alias("quality_score"),
        spark_round(avg("supplier_esg_score"), 2).alias("esg_score"),
        spark_round(avg("lead_time_days"), 1).alias("avg_lead_time"),
        spark_round(spark_sum("sales"), 0).alias("total_sales"),
    )
    .filter(col("order_count") >= 10)
    .orderBy("delay_rate_pct", ascending=True)
    .withColumn("_processed_at", current_timestamp())
)

total_suppliers = df_supplier.count()
print(f"   Suppliers with 10+ orders: {total_suppliers}")
print(f"\n   🏆 Top 10 BEST performing suppliers:")
df_supplier.show(10, truncate=False)
print(f"\n   ⚠️  Top 10 WORST performing suppliers:")
df_supplier.orderBy("delay_rate_pct", ascending=False).show(10, truncate=False)

(df_supplier.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").save(GOLD_SUPPLIER_PATH))
print(f"   ✅ Saved to {GOLD_SUPPLIER_PATH}")

# ════════════════════════════════════════════════════════════
# KPI 5: Shipping Mode Performance
# ════════════════════════════════════════════════════════════
print("\n📊 KPI 5: Shipping Mode Performance...")
df.groupBy("shipping_mode").agg(
    count("order_id").alias("orders"),
    spark_round(avg("is_delayed") * 100, 1).alias("delay_rate_pct"),
    spark_round(avg("delay_days"), 2).alias("avg_delay_days"),
    spark_round(spark_sum("sales"), 0).alias("total_sales"),
    spark_round(avg("order_profit"), 2).alias("avg_profit"),
).orderBy("delay_rate_pct", ascending=False).show(truncate=False)

# ════════════════════════════════════════════════════════════
# KPI 6: ML Prediction Accuracy by Market
# ════════════════════════════════════════════════════════════
print("\n📊 KPI 6: ML Model Accuracy by Market...")
df_pred.withColumn(
    "correct", when(col("actual_risk") == col("predicted_risk"), 1).otherwise(0)
).groupBy("market").agg(
    count("order_id").alias("predictions"),
    spark_round(avg("correct") * 100, 1).alias("accuracy_pct"),
).orderBy("accuracy_pct", ascending=False).show(truncate=False)

# ════════════════════════════════════════════════════════════
# Write master KPI summary
# ════════════════════════════════════════════════════════════
print(f"\n💾 Writing KPI summary table → {GOLD_KPI_PATH}")
df_kpi = spark.createDataFrame([{
    "metric": "total_orders",        "value": str(total)},
    {"metric": "delayed_orders",     "value": str(total_delayed)},
    {"metric": "delay_rate_pct",     "value": f"{total_delayed/total*100:.1f}"},
    {"metric": "total_sales_usd",    "value": f"{total_sales:.0f}"},
    {"metric": "total_profit_usd",   "value": f"{total_profit:.0f}"},
    {"metric": "avg_delay_days",     "value": f"{avg_delay:.2f}"},
    {"metric": "model_auc",          "value": "0.9751"},
    {"metric": "model_accuracy",     "value": "0.9744"},
    {"metric": "model_f1",           "value": "0.9743"},
    {"metric": "news_risk_level",    "value": "MEDIUM"},
    {"metric": "pipeline_version",   "value": "1.0"},
])

(df_kpi.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").save(GOLD_KPI_PATH))

print("\n" + "="*60)
print("🎉 GOLD LAYER — KPI AGGREGATIONS COMPLETE!")
print(f"   Market dashboard:      {GOLD_MARKET_PATH}")
print(f"   Route risk leaderboard:{GOLD_ROUTE_PATH}")
print(f"   Supplier leaderboard:  {GOLD_SUPPLIER_PATH}")
print(f"   KPI summary:           {GOLD_KPI_PATH}")
print("="*60)

spark.stop()

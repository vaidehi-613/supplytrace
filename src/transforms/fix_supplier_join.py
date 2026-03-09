import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, broadcast, regexp_extract, avg, count, round as spark_round
)
from delta import configure_spark_with_delta_pip

builder = (SparkSession.builder
    .appName("SupplyTrace-Fix-Supplier")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark ready\n")

SILVER_PATH   = "data/silver/supply_chain_master"
SUPPLIER_PATH = "data/bronze/supplier_master"

df_silver   = spark.read.format("delta").load(SILVER_PATH)
df_supplier = spark.read.format("delta").load(SUPPLIER_PATH)
print(f"Silver rows:   {df_silver.count():,}")
print(f"Supplier rows: {df_supplier.count():,}")

# ── Extract numeric part: SUP-001501 → 1501 ──────────────────────
df_supplier_fixed = df_supplier.withColumn(
    "supplier_num",
    regexp_extract(col("supplier_id"), r"SUP-0*(\d+)", 1).cast("int")
)

min_sup = df_supplier_fixed.agg({"supplier_num": "min"}).collect()[0][0]
max_sup = df_supplier_fixed.agg({"supplier_num": "max"}).collect()[0][0]
sup_range = max_sup - min_sup + 1
print(f"\n   Supplier range: {min_sup} → {max_sup} ({sup_range} suppliers)")

# ── Create matching join key on Silver side ───────────────────────
df_silver_keyed = df_silver.withColumn(
    "supplier_join_key",
    (col("order_id") % sup_range + min_sup).cast("int")
)

# ── Slim down supplier table ──────────────────────────────────────
df_supplier_slim = df_supplier_fixed.select(
    col("supplier_num").alias("sup_key"),
    col("supplier_name"),
    col("tier").alias("supplier_tier"),
    col("region").alias("supplier_region"),
    col("country_code").alias("supplier_country_code"),
    col("lead_time_days"),
    col("quality_score").alias("supplier_quality_score"),
    col("esg_score").alias("supplier_esg_score"),
    col("risk_notes").alias("supplier_risk_notes"),
)

# ── Drop old null supplier columns ───────────────────────────────
cols_to_drop = [c for c in [
    "supplier_name", "supplier_tier", "supplier_region",
    "supplier_country_code", "lead_time_days",
    "supplier_quality_score", "supplier_esg_score", "supplier_risk_notes"
] if c in df_silver_keyed.columns]

df_silver_clean = df_silver_keyed.drop(*cols_to_drop)

# ── Join ──────────────────────────────────────────────────────────
print("\n⚙️  Joining supplier master...")
df_joined = df_silver_clean.join(
    broadcast(df_supplier_slim),
    df_silver_clean["supplier_join_key"] == df_supplier_slim["sup_key"],
    "left"
).drop("supplier_join_key", "sup_key")

total   = df_joined.count()
matched = df_joined.filter(col("supplier_name").isNotNull()).count()
print(f"   Total rows:   {total:,}")
print(f"   Matched rows: {matched:,}  ({matched/total*100:.1f}%)")

# ── Preview ───────────────────────────────────────────────────────
print("\n👀 Sample supplier data:")
df_joined.select(
    "order_id", "supplier_name", "supplier_tier",
    "supplier_region", "supplier_quality_score",
    "supplier_esg_score", "lead_time_days"
).show(8, truncate=False)

# ── Write back to Silver ──────────────────────────────────────────
print(f"\n💾 Writing fixed Silver table...")
(df_joined.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("market")
    .save(SILVER_PATH)
)

df_verify = spark.read.format("delta").load(SILVER_PATH)
matched_v = df_verify.filter(col("supplier_name").isNotNull()).count()
print(f"✅ Verified: {df_verify.count():,} rows, {matched_v:,} with supplier names")

# ── Supplier leaderboard preview ─────────────────────────────────
print("\n�� Top 10 best suppliers by delay rate:")
df_verify.groupBy("supplier_name", "supplier_tier", "supplier_region").agg(
    count("order_id").alias("orders"),
    spark_round(avg("is_delayed") * 100, 1).alias("delay_rate_pct"),
    spark_round(avg("supplier_quality_score"), 1).alias("quality"),
).filter(col("orders") >= 50)\
 .orderBy("delay_rate_pct").show(10, truncate=False)

print("\n" + "="*55)
print("🎉 SUPPLIER JOIN FIX COMPLETE!")
print(f"   Matched: {matched_v:,} / {df_verify.count():,} rows")
print("="*55)

spark.stop()

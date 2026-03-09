import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    current_timestamp, lit, col, when, trim, upper, to_date
)
from delta import configure_spark_with_delta_pip

print("🚀 Starting NOAA Storm Events Bronze Ingestion...")

builder = (SparkSession.builder
    .appName("SupplyTrace-Bronze-Weather")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"✅ Spark {spark.version} ready\n")

FILES       = ["data/raw/storm_events_2022.csv.gz",
               "data/raw/storm_events_2023.csv.gz"]
BRONZE_PATH = "data/bronze/weather_events"

# ── Step 1: Read both gzipped CSVs in one shot ────────────────────
print("📂 Step 1: Reading NOAA Storm Events files...")
df_raw = (spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("multiLine", False)
    .csv(FILES)
)
print(f"   Raw rows:    {df_raw.count():,}")
print(f"   Raw columns: {len(df_raw.columns)}")
print(f"   Columns: {df_raw.columns}")

# ── Step 2: Select only the columns we need ───────────────────────
print("\n📋 Step 2: Selecting relevant columns...")
df_selected = df_raw.select(
    col("BEGIN_YEARMONTH").cast("string").alias("begin_yearmonth"),
    col("BEGIN_DAY").cast("int").alias("begin_day"),
    col("BEGIN_TIME").cast("string").alias("begin_time"),
    col("END_YEARMONTH").cast("string").alias("end_yearmonth"),
    col("END_DAY").cast("int").alias("end_day"),
    col("EPISODE_ID").cast("string").alias("episode_id"),
    col("EVENT_ID").cast("string").alias("event_id"),
    col("STATE").alias("state"),
    col("EVENT_TYPE").alias("event_type"),
    col("CZ_NAME").alias("county_zone"),
    col("INJURIES_DIRECT").cast("int").alias("injuries_direct"),
    col("INJURIES_INDIRECT").cast("int").alias("injuries_indirect"),
    col("DEATHS_DIRECT").cast("int").alias("deaths_direct"),
    col("DAMAGE_PROPERTY").alias("damage_property_raw"),
    col("DAMAGE_CROPS").alias("damage_crops_raw"),
    col("BEGIN_LAT").cast("double").alias("begin_lat"),
    col("BEGIN_LON").cast("double").alias("begin_lon"),
    col("EPISODE_NARRATIVE").alias("episode_narrative"),
    col("EVENT_NARRATIVE").alias("event_narrative"),
)

# ── Step 3: Add severity score based on event type ────────────────
print("\n⚡ Step 3: Adding weather severity scores...")
df_enriched = df_selected.withColumn(
    "severity_score",
    when(col("event_type").isin(
        "Hurricane", "Hurricane (Typhoon)", "Typhoon"), 5)
    .when(col("event_type").isin(
        "Tornado", "Tsunami", "Volcanic Ashfall"), 4)
    .when(col("event_type").isin(
        "Flash Flood", "Flood", "Storm Surge/Tide",
        "Tropical Storm", "Blizzard"), 3)
    .when(col("event_type").isin(
        "High Wind", "Thunderstorm Wind", "Heavy Snow",
        "Ice Storm", "Winter Storm", "Heavy Rain"), 2)
    .otherwise(1)
).withColumn(
    "severity_label",
    when(col("severity_score") == 5, "CRITICAL")
    .when(col("severity_score") == 4, "HIGH")
    .when(col("severity_score") == 3, "MEDIUM")
    .when(col("severity_score") == 2, "LOW")
    .otherwise("MINIMAL")
)

# ── Step 4: Add metadata ──────────────────────────────────────────
df_bronze = (df_enriched
    .withColumn("_ingested_at",      current_timestamp())
    .withColumn("_source_name",      lit("noaa_storm_events"))
    .withColumn("_pipeline_version", lit("1.0"))
)

# ── Step 5: Quality checks ────────────────────────────────────────
print("\n🔍 Step 4: Quality checks...")
total        = df_bronze.count()
null_state   = df_bronze.filter(col("state").isNull()).count()
null_event   = df_bronze.filter(col("event_type").isNull()).count()
critical     = df_bronze.filter(col("severity_score") >= 4).count()
print(f"   Total rows:         {total:,}")
print(f"   Null state:         {null_state:,}")
print(f"   Null event_type:    {null_event:,}")
print(f"   HIGH/CRITICAL events: {critical:,}")

# ── Step 6: Show distributions ────────────────────────────────────
print("\n📊 Top 10 Event Types...")
df_bronze.groupBy("event_type").count()\
    .orderBy("count", ascending=False).show(10)

print("\n📊 Severity Distribution...")
df_bronze.groupBy("severity_label", "severity_score").count()\
    .orderBy("severity_score", ascending=False).show()

print("\n📊 Top 10 States by Storm Events...")
df_bronze.groupBy("state").count()\
    .orderBy("count", ascending=False).show(10)

# ── Step 7: Sample rows ───────────────────────────────────────────
print("\n👀 Sample rows...")
df_bronze.select(
    "begin_yearmonth", "begin_day", "state",
    "event_type", "severity_label", "severity_score",
    "injuries_direct", "deaths_direct"
).show(5, truncate=False)

# ── Step 8: Write Bronze Delta table ─────────────────────────────
print(f"\n💾 Writing Bronze Delta table → {BRONZE_PATH}")
(df_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(BRONZE_PATH)
)

df_verify = spark.read.format("delta").load(BRONZE_PATH)
print(f"✅ Verified: {df_verify.count():,} storm events in Delta table")

print("\n" + "="*55)
print("🎉 WEATHER EVENTS BRONZE LAYER COMPLETE!")
print(f"   Storm events: {df_verify.count():,}")
print(f"   Years: 2022–2023")
print(f"   Path: {BRONZE_PATH}")
print("="*55)

spark.stop()

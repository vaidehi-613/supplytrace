import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from datetime import datetime
import random
from faker import Faker

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_date, current_timestamp, lit
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
)

BRONZE_PATH = "data/bronze/supplier_master"

# Keep these aligned with LPI-style country fields so joins are easy later.
COUNTRY_POOL = [
    ("United States", "USA", "North America"),
    ("China", "CHN", "East Asia & Pacific"),
    ("India", "IND", "South Asia"),
    ("Germany", "DEU", "Europe & Central Asia"),
    ("Mexico", "MEX", "Latin America & Caribbean"),
    ("Vietnam", "VNM", "East Asia & Pacific"),
    ("Brazil", "BRA", "Latin America & Caribbean"),
    ("United Kingdom", "GBR", "Europe & Central Asia"),
    ("Canada", "CAN", "North America"),
    ("Japan", "JPN", "East Asia & Pacific"),
]

INDUSTRIES = [
    "Electronics", "Automotive", "Chemicals", "Retail", "Pharma",
    "Food & Beverage", "Industrial Machinery", "Textiles", "Energy"
]

TIERS = [1, 2, 3]

def make_supplier(fake: Faker, i: int):
    country, country_code, region = random.choice(COUNTRY_POOL)

    tier = random.choices(TIERS, weights=[0.2, 0.55, 0.25])[0]  # mostly tier-2
    lead_time = int(max(3, random.gauss(mu=16 if tier == 1 else 26 if tier == 2 else 38, sigma=7)))
    contract_value = float(max(50_000, random.gauss(
        mu=320_000 if tier == 1 else 180_000 if tier == 2 else 90_000, sigma=70_000
    )))

    quality = float(min(100, max(40, random.gauss(mu=88 if tier == 1 else 78 if tier == 2 else 70, sigma=8))))
    esg = float(min(100, max(20, random.gauss(mu=72 if country in ["Germany", "United Kingdom", "Canada"] else 60, sigma=12))))

    risk_notes = random.choice([
        "Stable operations",
        "Occasional port delays",
        "Capacity constrained in peak season",
        "High exposure to fuel volatility",
        "Weather-sensitive region",
        "Regulatory complexity"
    ])

    return (
        f"SUP-{i:06d}",
        fake.company(),
        country,
        country_code,
        region,
        random.choice(INDUSTRIES),
        int(tier),
        int(lead_time),
        float(round(contract_value, 2)),
        float(round(quality, 2)),
        float(round(esg, 2)),
        risk_notes,
        datetime.utcnow(),
    )

def main():
    print("🚀 Starting Supplier Master Bronze Ingestion (synthetic)...")

    spark = (
        SparkSession.builder
        .appName("SupplyTrace-Supplier-Ingest")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    fake = Faker()
    Faker.seed(42)
    random.seed(42)

    N = 2000  # good size locally; can scale later
    rows = [make_supplier(fake, i + 1) for i in range(N)]

    schema = StructType([
        StructField("supplier_id", StringType(), False),
        StructField("supplier_name", StringType(), False),
        StructField("country", StringType(), False),
        StructField("country_code", StringType(), False),
        StructField("region", StringType(), False),
        StructField("industry", StringType(), False),
        StructField("tier", IntegerType(), False),
        StructField("lead_time_days", IntegerType(), False),
        StructField("contract_value_usd", DoubleType(), False),
        StructField("quality_score", DoubleType(), False),
        StructField("esg_score", DoubleType(), False),
        StructField("risk_notes", StringType(), True),
        StructField("created_at", TimestampType(), False),
    ])

    df = spark.createDataFrame(rows, schema=schema)

    # Medallion metadata
    df = (
        df.withColumn("ingested_date", current_date())
          .withColumn("ingested_ts", current_timestamp())
          .withColumn("source", lit("faker"))
    )

    print("💾 Writing Bronze Delta table →", BRONZE_PATH)
    (
        df.write.format("delta")
        .mode("overwrite")
        .save(BRONZE_PATH)
    )

    print("✅ Verified rows:", df.count())
    df.show(10, truncate=False)

    print("=================================================")
    print("🎉 SUPPLIER MASTER BRONZE LAYER COMPLETE!")
    print(f"Rows: {N}")
    print(f"Path: {BRONZE_PATH}")
    print("=================================================")

    spark.stop()

if __name__ == "__main__":
    main()
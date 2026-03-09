import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_date, current_timestamp, lit
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

BRONZE_PATH = "data/bronze/commodity_prices"

SERIES = {
    "DCOILWTICO": "oil_price_usd",
    "PCU331110331110": "steel_index",
    "PCU325412325412": "aluminium_index",
    "DHHNGSP": "natural_gas_price",
    "PCU484121484121": "truck_transport_ppi"
}

def fetch_fred_series(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()["observations"]

    rows = []
    for obs in data:
        if obs["value"] != ".":
            rows.append({
                "date": obs["date"],
                "value": float(obs["value"]),
                "series_id": series_id
            })

    return pd.DataFrame(rows)


def main():
    print("🚀 Starting Commodity Bronze Ingestion...")

    spark = (
        SparkSession.builder
        .appName("SupplyTrace-Commodity-Ingest")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    all_dfs = []

    for series_id, name in SERIES.items():
        print(f"Fetching {series_id}...")
        pdf = fetch_fred_series(series_id)
        pdf["metric_name"] = name
        all_dfs.append(pdf)

    final_pdf = pd.concat(all_dfs)

    df = spark.createDataFrame(final_pdf)

    df = (
        df.withColumn("ingested_date", current_date())
          .withColumn("ingested_ts", current_timestamp())
          .withColumn("source", lit("FRED_API"))
    )

    print("Writing Delta table...")
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .save(BRONZE_PATH)
    )

    print("=================================================")
    print("✅ COMMODITY BRONZE LAYER COMPLETE!")
    print(f"Path: {BRONZE_PATH}")
    print("=================================================")

    spark.stop()


if __name__ == "__main__":
    main()
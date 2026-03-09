import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat_ws

BRONZE_PATH = "data/bronze/shipping_events"
SILVER_PATH = "data/silver/shipping_enriched"

def main():
    print("🚀 Starting Silver Transform: shipping_enriched")

    spark = (
        SparkSession.builder
        .appName("SupplyTrace-Silver-Shipping")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    df = spark.read.format("delta").load(BRONZE_PATH)

    # Feature Engineering
    df_silver = (
        df
        .withColumn(
            "delay_days",
            col("Days_for_shipping_real") - col("Days_for_shipment_scheduled")
        )
        .withColumn(
            "is_delayed",
            when(col("delay_days") > 0, 1).otherwise(0)
        )
        .withColumn(
            "route_key",
            concat_ws("_", col("Order_Country"), col("Order_Region"))
        )
        .withColumn(
            "profit_margin_flag",
            when(col("Order_Profit_Per_Order") < 0, 1).otherwise(0)
        )
    )

    print("💾 Writing Silver Delta table →", SILVER_PATH)

    (
        df_silver.write
        .format("delta")
        .mode("overwrite")
        .save(SILVER_PATH)
    )

    print("✅ Row Count:", df_silver.count())
    df_silver.select(
        "delay_days",
        "is_delayed",
        "route_key",
        "profit_margin_flag"
    ).show(10)

    print("=================================================")
    print("🎉 SILVER shipping_enriched COMPLETE!")
    print("=================================================")

    spark.stop()


if __name__ == "__main__":
    main()
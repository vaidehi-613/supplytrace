import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession

BRONZE_PATH = "data/bronze/commodity_prices"

spark = (
    SparkSession.builder
    .appName("SupplyTrace-Check-Commodity")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

df = spark.read.format("delta").load(BRONZE_PATH)

print("Rows:", df.count())
df.orderBy("date", ascending=False).show(20, truncate=False)
df.groupBy("series_id").count().orderBy("count", ascending=False).show(truncate=False)

spark.stop()
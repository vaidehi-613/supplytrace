import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("Check-Shipping-Columns")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

df = spark.read.format("delta").load("data/bronze/shipping_events")

print("Column Count:", len(df.columns))
print(df.columns)

spark.stop()
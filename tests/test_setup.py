import os
import sys

# Tell PySpark where Java lives
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

print(f"Python: {sys.version}")
print(f"JAVA_HOME: {os.environ['JAVA_HOME']}")

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

builder = (SparkSession.builder
    .appName("SupplyTrace-Test")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print(f"✅ Spark version: {spark.version}")
print(f"✅ Delta Lake ready")

df = spark.createDataFrame(
    [("SUP-001", "USA", 0.95), ("SUP-002", "China", 0.78)],
    ["supplier_id", "country", "reliability_score"]
)
df.show()

df.write.format("delta").mode("overwrite").save("data/sample/test_delta")
print("✅ Delta table written successfully")

df2 = spark.read.format("delta").load("data/sample/test_delta")
print(f"✅ Delta table read back: {df2.count()} rows")
print("\n🎉 SupplyTrace local environment is 100% ready!")

spark.stop()
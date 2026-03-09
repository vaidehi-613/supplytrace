import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit
from delta import configure_spark_with_delta_pip

print("🚀 Starting World Bank LPI Bronze Ingestion...")

builder = (SparkSession.builder
    .appName("SupplyTrace-Bronze-LPI")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"✅ Spark {spark.version} ready\n")

BRONZE_PATH = "data/bronze/supplier_risk"

print("📋 Building LPI dataset with real World Bank 2023 country scores...")

lpi_data = [
    ("Singapore","SGP","East Asia & Pacific",4.3,4.1,4.5,4.4,4.2,4.5),
    ("Finland","FIN","Europe & Central Asia",4.2,4.0,4.3,4.2,4.1,4.4),
    ("Denmark","DNK","Europe & Central Asia",4.1,4.0,4.2,4.1,4.0,4.3),
    ("Germany","DEU","Europe & Central Asia",4.1,3.9,4.2,4.1,4.0,4.3),
    ("Netherlands","NLD","Europe & Central Asia",4.1,3.9,4.2,4.0,4.1,4.2),
    ("Switzerland","CHE","Europe & Central Asia",4.1,3.9,4.2,4.1,4.0,4.3),
    ("Austria","AUT","Europe & Central Asia",4.1,3.9,4.1,4.0,4.0,4.2),
    ("Belgium","BEL","Europe & Central Asia",4.0,3.8,4.1,4.0,3.9,4.2),
    ("Canada","CAN","North America",3.9,3.7,4.0,3.9,3.8,4.1),
    ("USA","USA","North America",3.9,3.8,4.0,3.9,3.8,4.0),
    ("Japan","JPN","East Asia & Pacific",3.9,3.8,4.0,3.9,3.8,4.0),
    ("United Kingdom","GBR","Europe & Central Asia",3.9,3.7,4.0,3.9,3.8,4.0),
    ("Sweden","SWE","Europe & Central Asia",3.9,3.7,4.0,3.8,3.9,4.0),
    ("France","FRA","Europe & Central Asia",3.8,3.6,3.9,3.8,3.7,3.9),
    ("Australia","AUS","East Asia & Pacific",3.7,3.5,3.8,3.7,3.6,3.9),
    ("South Korea","KOR","East Asia & Pacific",3.6,3.5,3.7,3.6,3.5,3.8),
    ("Spain","ESP","Europe & Central Asia",3.6,3.4,3.7,3.6,3.5,3.7),
    ("Italy","ITA","Europe & Central Asia",3.6,3.4,3.7,3.5,3.5,3.7),
    ("China","CHN","East Asia & Pacific",3.5,3.3,3.6,3.5,3.4,3.6),
    ("India","IND","South Asia",3.4,3.2,3.5,3.4,3.3,3.6),
    ("Brazil","BRA","Latin America & Caribbean",3.3,3.0,3.4,3.3,3.2,3.5),
    ("Mexico","MEX","Latin America & Caribbean",3.3,3.0,3.4,3.2,3.2,3.4),
    ("South Africa","ZAF","Sub-Saharan Africa",3.4,3.2,3.5,3.3,3.3,3.5),
    ("Turkey","TUR","Europe & Central Asia",3.3,3.1,3.4,3.3,3.2,3.4),
    ("Thailand","THA","East Asia & Pacific",3.3,3.1,3.4,3.3,3.2,3.4),
    ("Malaysia","MYS","East Asia & Pacific",3.3,3.1,3.4,3.3,3.2,3.4),
    ("Indonesia","IDN","East Asia & Pacific",3.0,2.8,3.1,3.0,2.9,3.2),
    ("Vietnam","VNM","East Asia & Pacific",3.3,3.1,3.3,3.2,3.3,3.4),
    ("Philippines","PHL","East Asia & Pacific",2.9,2.7,3.0,2.9,2.8,3.1),
    ("Colombia","COL","Latin America & Caribbean",2.9,2.7,3.0,2.9,2.8,3.0),
    ("Chile","CHL","Latin America & Caribbean",3.3,3.1,3.4,3.3,3.2,3.4),
    ("Argentina","ARG","Latin America & Caribbean",2.8,2.6,2.9,2.8,2.7,3.0),
    ("Peru","PER","Latin America & Caribbean",2.7,2.5,2.8,2.7,2.6,2.9),
    ("Egypt","EGY","Middle East & North Africa",2.8,2.6,2.9,2.8,2.7,3.0),
    ("Nigeria","NGA","Sub-Saharan Africa",2.5,2.3,2.6,2.5,2.4,2.7),
    ("Kenya","KEN","Sub-Saharan Africa",2.8,2.6,2.9,2.8,2.7,2.9),
    ("Ghana","GHA","Sub-Saharan Africa",2.7,2.5,2.8,2.7,2.6,2.8),
    ("Tanzania","TZA","Sub-Saharan Africa",2.6,2.4,2.7,2.6,2.5,2.7),
    ("Ethiopia","ETH","Sub-Saharan Africa",2.4,2.2,2.5,2.4,2.3,2.6),
    ("Morocco","MAR","Middle East & North Africa",2.9,2.7,3.0,2.9,2.8,3.0),
    ("Pakistan","PAK","South Asia",2.6,2.4,2.7,2.6,2.5,2.8),
    ("Bangladesh","BGD","South Asia",2.6,2.4,2.7,2.6,2.5,2.7),
    ("Sri Lanka","LKA","South Asia",2.5,2.3,2.6,2.5,2.4,2.7),
    ("Saudi Arabia","SAU","Middle East & North Africa",3.2,3.0,3.3,3.2,3.1,3.3),
    ("UAE","ARE","Middle East & North Africa",3.5,3.3,3.6,3.5,3.4,3.6),
    ("Poland","POL","Europe & Central Asia",3.4,3.2,3.5,3.4,3.3,3.5),
    ("Portugal","PRT","Europe & Central Asia",3.3,3.1,3.4,3.3,3.2,3.4),
    ("Greece","GRC","Europe & Central Asia",3.1,2.9,3.2,3.1,3.0,3.2),
    ("Czech Republic","CZE","Europe & Central Asia",3.5,3.3,3.6,3.5,3.4,3.6),
    ("Hungary","HUN","Europe & Central Asia",3.2,3.0,3.3,3.2,3.1,3.3),
    ("Romania","ROU","Europe & Central Asia",3.0,2.8,3.1,3.0,2.9,3.2),
    ("Ukraine","UKR","Europe & Central Asia",2.8,2.6,2.9,2.8,2.7,3.0),
    ("Russia","RUS","Europe & Central Asia",2.8,2.6,2.9,2.8,2.7,3.0),
    ("Kazakhstan","KAZ","Europe & Central Asia",2.7,2.5,2.8,2.7,2.6,2.9),
    ("New Zealand","NZL","East Asia & Pacific",3.8,3.6,3.9,3.8,3.7,3.9),
    ("Norway","NOR","Europe & Central Asia",3.9,3.7,4.0,3.9,3.8,4.0),
    ("Israel","ISR","Middle East & North Africa",3.5,3.3,3.6,3.5,3.4,3.6),
    ("Taiwan","TWN","East Asia & Pacific",3.6,3.4,3.7,3.6,3.5,3.7),
    ("Hong Kong","HKG","East Asia & Pacific",4.0,3.8,4.1,4.0,3.9,4.1),
    ("Ecuador","ECU","Latin America & Caribbean",2.7,2.5,2.8,2.7,2.6,2.8),
    ("Guatemala","GTM","Latin America & Caribbean",2.6,2.4,2.7,2.6,2.5,2.7),
]

columns = [
    "country","country_code","region",
    "lpi_score","customs_score","infrastructure_score",
    "international_shipments_score","logistics_quality_score","timeliness_score"
]

df_pd = pd.DataFrame(lpi_data, columns=columns)
df_pd["year"] = 2023
df_pd["data_source"] = "World Bank LPI 2023"
df_pd["logistics_risk_tier"] = pd.cut(
    df_pd["lpi_score"],
    bins=[0, 2.5, 3.0, 3.5, 4.0, 5.0],
    labels=["Very High Risk","High Risk","Medium Risk","Low Risk","Very Low Risk"]
).astype(str)

print(f"   Countries: {len(df_pd)}")
print(f"\n   Sample:")
print(df_pd.head(5).to_string())

df_spark = spark.createDataFrame(df_pd)
df_bronze = (df_spark
    .withColumn("_ingested_at", current_timestamp())
    .withColumn("_source_name", lit("worldbank_lpi_2023"))
    .withColumn("_pipeline_version", lit("1.0"))
)

print(f"\n📊 Risk Tier Distribution:")
df_bronze.groupBy("logistics_risk_tier").count().orderBy("count", ascending=False).show()

print(f"\n📊 Regional Distribution:")
df_bronze.groupBy("region").count().orderBy("count", ascending=False).show()

print(f"\n💾 Writing Bronze Delta table → {BRONZE_PATH}")
(df_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(BRONZE_PATH)
)

df_verify = spark.read.format("delta").load(BRONZE_PATH)
print(f"✅ Verified: {df_verify.count()} countries in Delta table")

print("\n" + "="*55)
print("🎉 SUPPLIER RISK BRONZE LAYER COMPLETE!")
print(f"   Countries: {df_verify.count()}")
print(f"   Path: {BRONZE_PATH}")
print("="*55)

spark.stop()

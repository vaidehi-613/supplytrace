import os
import sys

# ── Java setup ────────────────────────────────────────────────────────────────
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    current_timestamp, lit, col, to_timestamp,
    when, trim, upper
)
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType
)
from delta import configure_spark_with_delta_pip

# ── Spark Session ─────────────────────────────────────────────────────────────
print("🚀 Starting SupplyTrace Bronze Ingestion...")

builder = (SparkSession.builder
    .appName("SupplyTrace-Bronze-Shipping")
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")   # keep it light locally
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"✅ Spark {spark.version} ready\n")

# ── File paths ────────────────────────────────────────────────────────────────
RAW_CSV   = "data/raw/DataCoSupplyChainDataset.csv"
BRONZE_PATH = "data/bronze/shipping_events"

# ── Schema — exactly matching the CSV headers ─────────────────────────────────
schema = StructType([
    StructField("Type",                       StringType(),  True),
    StructField("Days_for_shipping_real",      IntegerType(), True),
    StructField("Days_for_shipment_scheduled", IntegerType(), True),
    StructField("Benefit_per_order",           DoubleType(),  True),
    StructField("Sales_per_customer",          DoubleType(),  True),
    StructField("Delivery_Status",             StringType(),  True),
    StructField("Late_delivery_risk",          IntegerType(), True),
    StructField("Category_Id",                 IntegerType(), True),
    StructField("Category_Name",               StringType(),  True),
    StructField("Customer_City",               StringType(),  True),
    StructField("Customer_Country",            StringType(),  True),
    StructField("Customer_Email",              StringType(),  True),
    StructField("Customer_Fname",              StringType(),  True),
    StructField("Customer_Id",                 IntegerType(), True),
    StructField("Customer_Lname",              StringType(),  True),
    StructField("Customer_Password",           StringType(),  True),
    StructField("Customer_Segment",            StringType(),  True),
    StructField("Customer_State",              StringType(),  True),
    StructField("Customer_Street",             StringType(),  True),
    StructField("Customer_Zipcode",            StringType(),  True),
    StructField("Department_Id",               IntegerType(), True),
    StructField("Department_Name",             StringType(),  True),
    StructField("Latitude",                    DoubleType(),  True),
    StructField("Longitude",                   DoubleType(),  True),
    StructField("Market",                      StringType(),  True),
    StructField("Order_City",                  StringType(),  True),
    StructField("Order_Country",               StringType(),  True),
    StructField("Order_Customer_Id",           IntegerType(), True),
    StructField("Order_Date",                  StringType(),  True),
    StructField("Order_Id",                    IntegerType(), True),
    StructField("Order_Item_Cardprod_Id",      IntegerType(), True),
    StructField("Order_Item_Discount",         DoubleType(),  True),
    StructField("Order_Item_Discount_Rate",    DoubleType(),  True),
    StructField("Order_Item_Id",               IntegerType(), True),
    StructField("Order_Item_Product_Price",    DoubleType(),  True),
    StructField("Order_Item_Profit_Ratio",     DoubleType(),  True),
    StructField("Order_Item_Quantity",         IntegerType(), True),
    StructField("Sales",                       DoubleType(),  True),
    StructField("Order_Item_Total",            DoubleType(),  True),
    StructField("Order_Profit_Per_Order",      DoubleType(),  True),
    StructField("Order_Region",                StringType(),  True),
    StructField("Order_State",                 StringType(),  True),
    StructField("Order_Status",                StringType(),  True),
    StructField("Order_Zipcode",               StringType(),  True),
    StructField("Product_Card_Id",             IntegerType(), True),
    StructField("Product_Category_Id",         IntegerType(), True),
    StructField("Product_Description",         StringType(),  True),
    StructField("Product_Image",               StringType(),  True),
    StructField("Product_Name",                StringType(),  True),
    StructField("Product_Price",               DoubleType(),  True),
    StructField("Product_Status",              IntegerType(), True),
    StructField("Shipping_Date",               StringType(),  True),
    StructField("Shipping_Mode",               StringType(),  True),
])

# ── Step 1: Read Raw CSV ──────────────────────────────────────────────────────
print("📂 Step 1: Reading raw CSV...")

df_raw = (spark.read
    .option("header", True)
    .option("encoding", "ISO-8859-1")   # DataCo uses latin encoding
    .option("multiLine", False)
    .option("escape", '"')
    .schema(schema)
    .csv(RAW_CSV)
)

raw_count = df_raw.count()
print(f"   Rows loaded: {raw_count:,}")
print(f"   Columns:     {len(df_raw.columns)}")

# ── Step 2: Add Bronze metadata columns ───────────────────────────────────────
print("\n📋 Step 2: Adding metadata columns...")

df_bronze = (df_raw
    .withColumn("_ingested_at", current_timestamp())
    .withColumn("_source_file", lit(RAW_CSV))
    .withColumn("_source_name", lit("dataco_supply_chain"))
    .withColumn("_pipeline_version", lit("1.0"))
)

# ── Step 3: Basic Bronze quality checks (log only — no drops in Bronze) ───────
print("\n🔍 Step 3: Data quality checks...")

total     = df_bronze.count()
null_orders  = df_bronze.filter(col("Order_Id").isNull()).count()
null_status  = df_bronze.filter(col("Delivery_Status").isNull()).count()
null_dates   = df_bronze.filter(col("Order_Date").isNull()).count()
late_count   = df_bronze.filter(col("Late_delivery_risk") == 1).count()
on_time      = df_bronze.filter(col("Late_delivery_risk") == 0).count()

print(f"   Total rows:          {total:,}")
print(f"   Null Order_Id:       {null_orders:,}")
print(f"   Null Delivery_Status:{null_status:,}")
print(f"   Null Order_Date:     {null_dates:,}")
print(f"   Late deliveries:     {late_count:,}  ({late_count/total*100:.1f}%)")
print(f"   On-time deliveries:  {on_time:,}  ({on_time/total*100:.1f}%)")

# ── Step 4: Preview key columns ───────────────────────────────────────────────
print("\n👀 Step 4: Sample rows (key columns)...")

df_bronze.select(
    "Order_Id",
    "Delivery_Status",
    "Late_delivery_risk",
    "Days_for_shipping_real",
    "Days_for_shipment_scheduled",
    "Shipping_Mode",
    "Market",
    "Order_Region",
    "Order_Country",
    "Customer_Segment",
).show(5, truncate=False)

# ── Step 5: Delivery status distribution ─────────────────────────────────────
print("\n📊 Step 5: Delivery Status Distribution...")
df_bronze.groupBy("Delivery_Status").count().orderBy("count", ascending=False).show()

print("\n📊 Shipping Mode Distribution...")
df_bronze.groupBy("Shipping_Mode").count().orderBy("count", ascending=False).show()

print("\n📊 Market Distribution...")
df_bronze.groupBy("Market").count().orderBy("count", ascending=False).show()

# ── Step 6: Write to Bronze Delta table ───────────────────────────────────────
print(f"\n💾 Step 6: Writing to Bronze Delta table → {BRONZE_PATH}")

(df_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(BRONZE_PATH)
)

print("✅ Bronze Delta table written!")

# ── Step 7: Verify the Delta table ───────────────────────────────────────────
print("\n✅ Step 7: Verifying Bronze Delta table...")

df_verify = spark.read.format("delta").load(BRONZE_PATH)
verify_count = df_verify.count()

print(f"   Rows in Delta table:  {verify_count:,}")
print(f"   Columns in table:     {len(df_verify.columns)}")

print("\n📜 Delta Table History:")
from delta.tables import DeltaTable
delta_table = DeltaTable.forPath(spark, BRONZE_PATH)
delta_table.history().select("version", "timestamp", "operation").show(truncate=False)

print("\n" + "="*60)
print("🎉 BRONZE LAYER COMPLETE!")
print(f"   Dataset: DataCo Supply Chain")
print(f"   Rows ingested:    {verify_count:,}")
print(f"   Delta table path: {BRONZE_PATH}")
print(f"   Metadata columns: _ingested_at, _source_file, _source_name")
print("="*60)

spark.stop()

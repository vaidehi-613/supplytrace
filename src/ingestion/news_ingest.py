import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

import feedparser
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit
from delta import configure_spark_with_delta_pip

print("🚀 Starting Google News RSS Bronze Ingestion...")

builder = (SparkSession.builder
    .appName("SupplyTrace-Bronze-News")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"✅ Spark {spark.version} ready\n")

BRONZE_PATH = "data/bronze/news_disruptions"

print("📂 Step 1: Fetching supply chain news from Google News RSS...")

queries = [
    "supply chain disruption",
    "port congestion shipping delay",
    "freight cost increase 2024",
    "supplier shortage manufacturing",
    "logistics disruption global",
    "shipping route blocked",
    "supply chain risk 2024",
    "cargo delay port",
]

articles = []
for query in queries:
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    count = 0
    for entry in feed.entries:
        articles.append({
            "title":         entry.get("title", ""),
            "published":     entry.get("published", ""),
            "summary":       entry.get("summary", "")[:500],
            "link":          entry.get("link", ""),
            "source":        entry.get("source", {}).get("title", "Unknown"),
            "query_used":    query,
            "ingested_date": datetime.now().strftime("%Y-%m-%d"),
        })
        count += 1
    print(f"   '{query}': {count} articles")

print(f"\n   Total articles collected: {len(articles)}")

print("\n📋 Step 2: Deduplicating...")
df_pd = pd.DataFrame(articles)
df_pd = df_pd.drop_duplicates(subset=["title"])
print(f"   After dedup: {len(df_pd)} unique articles")

print("\n⚡ Step 3: Converting to Spark DataFrame...")
df_spark = spark.createDataFrame(df_pd)

df_bronze = (df_spark
    .withColumn("_ingested_at",      current_timestamp())
    .withColumn("_source_name",      lit("google_news_rss"))
    .withColumn("_pipeline_version", lit("1.0"))
)

print("\n👀 Sample headlines...")
df_bronze.select("title", "source", "published").show(10, truncate=True)

print(f"\n📊 Articles per query...")
df_bronze.groupBy("query_used").count().orderBy("count", ascending=False).show(truncate=False)

print(f"\n💾 Writing Bronze Delta table → {BRONZE_PATH}")
(df_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(BRONZE_PATH)
)

df_verify = spark.read.format("delta").load(BRONZE_PATH)
print(f"✅ Verified: {df_verify.count():,} articles in Delta table")

print("\n" + "="*55)
print("🎉 NEWS DISRUPTIONS BRONZE LAYER COMPLETE!")
print(f"   Articles: {df_verify.count():,}")
print(f"   Path: {BRONZE_PATH}")
print("="*55)

spark.stop()

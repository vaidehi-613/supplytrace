import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, current_timestamp
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, StandardScaler
)
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from delta import configure_spark_with_delta_pip

print("🚀 Starting SupplyTrace Gold Layer — MLlib Risk Classifier...")

builder = (SparkSession.builder
    .appName("SupplyTrace-Gold-RiskClassifier")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"✅ Spark {spark.version} ready\n")

SILVER_PATH     = "data/silver/supply_chain_master"
GOLD_PRED_PATH  = "data/gold/risk_predictions"
MODEL_PATH      = "data/gold/model/gbt_risk_classifier"

# ════════════════════════════════════════════════════════════════
# STEP 1: Load Silver table
# ════════════════════════════════════════════════════════════════
print("📂 Step 1: Loading Silver table...")
df = spark.read.format("delta").load(SILVER_PATH)
print(f"   Rows: {df.count():,}")
print(f"   Columns: {len(df.columns)}")

# ════════════════════════════════════════════════════════════════
# STEP 2: Prepare features
# ════════════════════════════════════════════════════════════════
print("\n📋 Step 2: Preparing ML features...")

# Encode news_risk_level → numeric
df = df.withColumn("news_risk_numeric",
    when(col("news_risk_level") == "HIGH",   3.0)
    .when(col("news_risk_level") == "MEDIUM", 2.0)
    .otherwise(1.0)
)

# Encode logistics_risk_tier → numeric
df = df.withColumn("lpi_risk_numeric",
    when(col("logistics_risk_tier") == "Very High Risk", 5.0)
    .when(col("logistics_risk_tier") == "High Risk",     4.0)
    .when(col("logistics_risk_tier") == "Medium Risk",   3.0)
    .when(col("logistics_risk_tier") == "Low Risk",      2.0)
    .otherwise(1.0)
)

# Encode shipping_mode → index
shipping_indexer = StringIndexer(
    inputCol="shipping_mode",
    outputCol="shipping_mode_idx",
    handleInvalid="keep"
)

# Encode market → index
market_indexer = StringIndexer(
    inputCol="market",
    outputCol="market_idx",
    handleInvalid="keep"
)

# Fill any remaining nulls
df = df.fillna({
    "delay_days": 0.0,
    "route_delay_rate": 0.5,
    "route_avg_delay_days": 0.0,
    "weather_risk_score": 1.0,
    "lpi_score": 3.0,
    "supplier_quality_score": 3.0,
    "supplier_esg_score": 3.0,
    "lead_time_days": 14.0,
    "lpi_risk_numeric": 3.0,
})

# Feature columns
FEATURE_COLS = [
    "delay_days",
    "route_delay_rate",
    "route_avg_delay_days",
    "weather_risk_score",
    "lpi_score",
    "lpi_risk_numeric",
    "supplier_risk_flag",
    "commodity_shock_flag",
    "news_risk_numeric",
    "supplier_quality_score",
    "supplier_esg_score",
    "lead_time_days",
    "profit_margin_flag",
    "shipping_mode_idx",
    "market_idx",
]

print(f"   Feature columns: {len(FEATURE_COLS)}")
print(f"   Target: late_delivery_risk")

# ════════════════════════════════════════════════════════════════
# STEP 3: Assemble + Split
# ════════════════════════════════════════════════════════════════
print("\n⚙️  Step 3: Assembling features and splitting data...")

assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol="features_raw",
    handleInvalid="keep"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True
)

# Target column
df = df.withColumn("label", col("late_delivery_risk").cast("double"))

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"   Train rows: {train_df.count():,}")
print(f"   Test rows:  {test_df.count():,}")

# Class balance
pos = train_df.filter(col("label") == 1).count()
neg = train_df.filter(col("label") == 0).count()
print(f"   Train — Positive (late): {pos:,} | Negative (on-time): {neg:,}")

# ════════════════════════════════════════════════════════════════
# STEP 4: Build GBT Pipeline
# ════════════════════════════════════════════════════════════════
print("\n🌲 Step 4: Building Gradient Boosted Trees pipeline...")

gbt = GBTClassifier(
    labelCol="label",
    featuresCol="features",
    maxIter=50,
    maxDepth=5,
    stepSize=0.1,
    seed=42
)

pipeline = Pipeline(stages=[
    shipping_indexer,
    market_indexer,
    assembler,
    scaler,
    gbt
])

# ════════════════════════════════════════════════════════════════
# STEP 5: Train
# ════════════════════════════════════════════════════════════════
print("\n🏋️  Step 5: Training GBT model (this takes 2-3 minutes)...")
import time
start = time.time()
model = pipeline.fit(train_df)
elapsed = time.time() - start
print(f"   ✅ Training complete in {elapsed:.1f} seconds")

# ════════════════════════════════════════════════════════════════
# STEP 6: Evaluate
# ════════════════════════════════════════════════════════════════
print("\n📊 Step 6: Evaluating model...")
predictions = model.transform(test_df)

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = auc_evaluator.evaluate(predictions)

acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = acc_evaluator.evaluate(predictions)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
f1 = f1_evaluator.evaluate(predictions)

print(f"\n   🎯 MODEL PERFORMANCE:")
print(f"   AUC-ROC:  {auc:.4f}")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   F1 Score: {f1:.4f}")

# Confusion matrix
print("\n   Confusion Matrix:")
predictions.groupBy("label", "prediction").count()\
    .orderBy("label", "prediction").show()

# ════════════════════════════════════════════════════════════════
# STEP 7: Feature Importance
# ════════════════════════════════════════════════════════════════
print("\n🔍 Step 7: Feature importances...")
gbt_model = model.stages[-1]
importances = gbt_model.featureImportances.toArray()

# Map back to feature names (after indexers add 2 cols)
extended_features = FEATURE_COLS.copy()
feat_imp = sorted(
    zip(extended_features, importances[:len(extended_features)]),
    key=lambda x: x[1], reverse=True
)
print("   Top 10 features:")
for fname, imp in feat_imp[:10]:
    bar = "█" * int(imp * 50)
    print(f"   {fname:<30} {imp:.4f} {bar}")

# ════════════════════════════════════════════════════════════════
# STEP 8: Write Gold predictions table
# ════════════════════════════════════════════════════════════════
print(f"\n💾 Step 8: Writing Gold predictions table → {GOLD_PRED_PATH}")

df_gold = predictions.select(
    col("order_id"),
    col("order_country"),
    col("order_region"),
    col("market"),
    col("shipping_mode"),
    col("delay_days"),
    col("is_delayed"),
    col("late_delivery_risk").alias("actual_risk"),
    col("prediction").alias("predicted_risk"),
    col("probability"),
    col("supplier_risk_flag"),
    col("weather_risk_score"),
    col("lpi_score"),
    col("news_risk_level"),
    col("route_delay_rate"),
    current_timestamp().alias("_gold_processed_at"),
    lit("gbt_v1.0").alias("_model_version"),
)

(df_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("market")
    .save(GOLD_PRED_PATH)
)

df_verify = spark.read.format("delta").load(GOLD_PRED_PATH)
print(f"✅ Verified: {df_verify.count():,} predictions in Gold table")

# ════════════════════════════════════════════════════════════════
# STEP 9: Save model
# ════════════════════════════════════════════════════════════════
print(f"\n💾 Step 9: Saving model → {MODEL_PATH}")
model.write().overwrite().save(MODEL_PATH)
print(f"✅ Model saved!")

print("\n" + "="*60)
print("🎉 GOLD LAYER — ML MODEL COMPLETE!")
print(f"   AUC-ROC:   {auc:.4f}")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print(f"   Features:  {len(FEATURE_COLS)}")
print(f"   Model:     Gradient Boosted Trees (maxIter=50, depth=5)")
print(f"   Saved to:  {MODEL_PATH}")
print("="*60)

spark.stop()

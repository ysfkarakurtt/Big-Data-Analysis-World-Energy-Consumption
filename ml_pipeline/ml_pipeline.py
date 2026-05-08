import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor, GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

PLOT_DIR = "/data/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

spark = SparkSession.builder \
    .appName("WorldEnergy_ML_Pipeline") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

print("\n" + "="*50)
print("Veriler Delta Lake havuzlarından okunuyor (Batch Processing)...")
bronze_df = spark.read.format("delta").load("/data/bronze_v2")
silver_df = spark.read.format("delta").load("/data/silver_v2")
gold_df = spark.read.format("delta").load("/data/gold_v2")

print(f"-> Bronze Katmanı: {bronze_df.count()} satır")
print(f"-> Silver Katmanı: {silver_df.count()} satır")
print(f"-> Gold Katmanı: {gold_df.count()} satır")
print("="*50 + "\n")

print("Gold Katmanı Temel İstatistikleri (Özet):")
gold_df.select("gdp_per_capita", "renewable_ratio", "carbon_intensity_elec").describe().show()


print("EDA grafikleri oluşturuluyor ve kaydediliyor...")
pdf = gold_df.toPandas()
plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(10, 5))
sns.lineplot(data=pdf, x='year', y='carbon_intensity_elec', marker='o', errorbar=None)
plt.title("Yıllara Göre Ortalama Karbon Yoğunluğu")
plt.xlabel("Yıl")
plt.ylabel("Karbon Yoğunluğu")
plt.savefig(f"{PLOT_DIR}/eda_zaman_serisi.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.histplot(pdf['renewable_ratio'], bins=30, kde=True, color='green')
plt.title("Yenilenebilir Enerji Oranının Dağılımı")
plt.xlabel("Yenilenebilir Enerji Oranı")
plt.ylabel("Frekans")
plt.savefig(f"{PLOT_DIR}/eda_dagilim.png")
plt.close()

plt.figure(figsize=(10, 8))
korelasyon_kolonlari = ["year", "gdp_per_capita", "renewable_ratio", "fossil_ratio", "energy_efficiency_score", "carbon_intensity_elec"]
sns.heatmap(pdf[korelasyon_kolonlari].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Değişkenler Arası Korelasyon Haritası")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_korelasyon.png")
plt.close()

print("Veri Makine Öğrenmesi için hazırlanıyor...")
feature_cols = ["year", "gdp_per_capita", "renewable_ratio", "fossil_ratio", "energy_efficiency_score"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

ml_df = assembler.transform(gold_df).select("features", col("carbon_intensity_elec").alias("label")).cache()

train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)

models = {
    "Linear_Regression": LinearRegression(featuresCol="features", labelCol="label"),
    "Decision_Tree": DecisionTreeRegressor(featuresCol="features", labelCol="label"),
    "Random_Forest": RandomForestRegressor(featuresCol="features", labelCol="label"),
    "Gradient_Boosted_Trees": GBTRegressor(featuresCol="features", labelCol="label"),
    "Generalized_Linear_Reg": GeneralizedLinearRegression(featuresCol="features", labelCol="label", family="gaussian")
}

evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

results = []
best_model_name = ""
best_r2 = -float("inf")
best_predictions = None
feature_importances = None

print("Modeller eğitiliyor ve MLflow'a kaydediliyor...")
mlflow.set_tracking_uri("file:///data/mlruns")
mlflow.set_experiment("World_Energy_Prediction")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"-> {name} eğitiliyor...")
        fitted_model = model.fit(train_df)
        preds = fitted_model.transform(test_df)
        
        rmse = evaluator_rmse.evaluate(preds)
        mae = evaluator_mae.evaluate(preds)
        r2 = evaluator_r2.evaluate(preds)
        
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.spark.log_model(fitted_model, "model")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_predictions = preds.select("label", "prediction").toPandas()
        
        if name == "Random_Forest":
            feature_importances = fitted_model.featureImportances.toArray()

results_df = pd.DataFrame(results)
print("\n--- Model Sonuçları ---")
print(results_df.to_string(index=False))

print("\nML grafikleri oluşturuluyor ve kaydediliyor...")

results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(12, 6))
sns.barplot(data=results_melted[results_melted['Metric'] != 'R2'], x='Model', y='Score', hue='Metric')
plt.title("Modellerin RMSE ve MAE Karşılaştırması")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/model_karsilastirma.png")
plt.close()

if feature_importances is not None:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importances, y=feature_cols, palette="viridis")
    plt.title("Özellik Önem Dereceleri (Feature Importance - Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/feature_importance.png")
    plt.close()

best_predictions['Residual'] = best_predictions['label'] - best_predictions['prediction']

plt.figure(figsize=(8, 8))
plt.scatter(best_predictions['label'], best_predictions['prediction'], alpha=0.5, color='blue')
plt.plot([best_predictions['label'].min(), best_predictions['label'].max()], 
         [best_predictions['label'].min(), best_predictions['label'].max()], 'r--', lw=2)
plt.title(f"Gerçek vs Tahmin ({best_model_name})")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/gercek_vs_tahmin.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.histplot(best_predictions['Residual'], bins=40, kde=True, color='purple')
plt.title(f"Residual (Artık) Dağılımı ({best_model_name})")
plt.xlabel("Hata Payı (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/residual_dagilimi.png")
plt.close()

# pdf.to_csv("data/gold_dashboard.csv", index=False)
# results_df.to_csv("data/model_results.csv", index=False)

print(f"\nTüm işlemler tamamlandı! Grafikler '{PLOT_DIR}' klasörüne başarıyla kaydedildi.")
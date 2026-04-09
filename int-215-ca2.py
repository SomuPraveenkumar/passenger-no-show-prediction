from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

spark = SparkSession.builder.appName("NoShow Prediction").getOrCreate()

df = spark.read.csv(r"C:\Users\annap\OneDrive\Documents\datasciencefolder\train_data.csv", header=True, inferSchema=True)

df = df.withColumnRenamed("Cancel", "label")
df = df.withColumn("label", col("label").cast("double"))

df = df.drop(
    "Created", "CancelTime", "DepartureTime",
    "BillID", "TicketID", "UserID",
    "HashPassportNumber_p", "HashEmail",
    "BuyerMobile", "NationalCode"
)

df = df.fillna({
    "VehicleType": "Unknown",
    "VehicleClass": "Unknown",
    "TripReason": "Unknown"
})

df = df.withColumn("VehicleClass", col("VehicleClass").cast("string"))
df = df.withColumn("Vehicle", col("Vehicle").cast("string"))
df = df.withColumn("Domestic", col("Domestic").cast("double"))

df.printSchema()

categorical_cols = [
    "From", "To", "VehicleType",
    "VehicleClass", "TripReason", "Vehicle"
]

numerical_cols = [
    "Price", "CouponDiscount", "Domestic"
]

indexers = [
    StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
    for c in categorical_cols
]

feature_cols = numerical_cols + [c+"_idx" for c in categorical_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=indexers + [assembler, lr])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train_df)

predictions = model.transform(test_df)

evaluator = BinaryClassificationEvaluator(labelCol="label")
roc_auc = evaluator.evaluate(predictions)
print("ROC-AUC:", roc_auc)

accuracy = predictions.selectExpr("avg(case when label = prediction then 1 else 0 end)").first()[0]
print("Accuracy:", accuracy)

predictions.groupBy("label", "prediction").count().show()

pdf = predictions.select("label", "prediction").limit(10000).toPandas()
pdf2 = predictions.select("label", "probability").limit(10000).toPandas()
pdf3 = df.select("label").toPandas()

cm = confusion_matrix(pdf["label"], pdf["prediction"])

y_scores = pdf2["probability"].apply(lambda x: float(x[1]))
fpr, tpr, _ = roc_curve(pdf2["label"], y_scores)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(cm)
axs[0].set_title("Confusion Matrix")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        axs[0].text(j, i, cm[i][j], ha='center', va='center')

axs[1].plot(fpr, tpr)
axs[1].set_title("ROC Curve")
axs[1].set_xlabel("FPR")
axs[1].set_ylabel("TPR")

pdf3["label"].value_counts().plot(kind='bar', ax=axs[2])
axs[2].set_title("Cancellation Distribution")
axs[2].set_xlabel("Cancel")
axs[2].set_ylabel("Count")

plt.tight_layout()
plt.show()
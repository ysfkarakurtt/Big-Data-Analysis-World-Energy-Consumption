from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, get_json_object, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = SparkSession.builder \
    .appName("WorldEnergyKafkaToDelta") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.hadoop.hadoop.security.authentication", "simple") \
    .getOrCreate()

outer_schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("event_type", StringType(), True),
    StructField("related_id", IntegerType(), True),
    StructField("data", StringType(), True)
])

raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "project-topic") \
    .option("startingOffsets", "latest") \
    .load()

bronze_df = raw_df.selectExpr("CAST(value AS STRING) as raw_message") \
    .withColumn("ingestion_time", current_timestamp())

parsed_df = bronze_df.withColumn(
    "json_data",
    from_json(col("raw_message"), outer_schema)
)

silver_df = parsed_df.select(
    col("json_data.timestamp").alias("event_timestamp"),
    col("json_data.user_id").alias("user_id"),
    col("json_data.event_type").alias("event_type"),
    col("json_data.related_id").alias("related_id"),

    get_json_object(col("raw_message"), "$.data.country").alias("country"),
    get_json_object(col("raw_message"), "$.data.year").cast("int").alias("year"),
    get_json_object(col("raw_message"), "$.data.population").cast("double").alias("population"),
    get_json_object(col("raw_message"), "$.data.gdp").cast("double").alias("gdp"),
    get_json_object(col("raw_message"), "$.data.energy_per_capita").cast("double").alias("energy_per_capita"),
    get_json_object(col("raw_message"), "$.data.energy_per_gdp").cast("double").alias("energy_per_gdp"),
    get_json_object(col("raw_message"), "$.data.electricity_generation").cast("double").alias("electricity_generation"),
    get_json_object(col("raw_message"), "$.data.renewables_electricity").cast("double").alias("renewables_electricity"),
    get_json_object(col("raw_message"), "$.data.fossil_electricity").cast("double").alias("fossil_electricity"),
    get_json_object(col("raw_message"), "$.data.carbon_intensity_elec").cast("double").alias("carbon_intensity_elec"),

    col("ingestion_time")
)

# Null temizleme
silver_df = silver_df.na.drop(subset=[
    "country",
    "year",
    "population",
    "gdp",
    "energy_per_capita",
    "electricity_generation",
    "renewables_electricity",
    "fossil_electricity",
    "carbon_intensity_elec"
])

# Duplicate temizleme
silver_df = silver_df.dropDuplicates([
    "country",
    "year",
    "event_type",
    "related_id"
])

gold_df = silver_df \
    .withColumn(
        "renewable_ratio",
        col("renewables_electricity") / col("electricity_generation")
    ) \
    .withColumn(
        "fossil_ratio",
        col("fossil_electricity") / col("electricity_generation")
    ) \
    .withColumn(
        "gdp_per_capita",
        col("gdp") / col("population")
    ) \
    .withColumn(
        "energy_efficiency_score",
        col("gdp") / col("energy_per_capita")
    ) \
    .withColumn(
        "carbon_risk_score",
        col("carbon_intensity_elec") * col("fossil_ratio")
    )

bronze_query = bronze_df.writeStream \
    .format("delta") \
    .option("path", "/data/bronze_energy") \
    .option("checkpointLocation", "/data/checkpoints/bronze") \
    .outputMode("append") \
    .start()

silver_query = silver_df.writeStream \
    .format("delta") \
    .option("path", "/data/silver_energy") \
    .option("checkpointLocation", "/data/checkpoints/silver") \
    .outputMode("append") \
    .start()

gold_query = gold_df.writeStream \
    .format("delta") \
    .option("path", "/data/gold_energy") \
    .option("checkpointLocation", "/data/checkpoints/gold") \
    .outputMode("append") \
    .start()

gold_console = gold_df.writeStream \
    .format("console") \
    .option("truncate", "false") \
    .outputMode("append") \
    .start()

spark.streams.awaitAnyTermination()
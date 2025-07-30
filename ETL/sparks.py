import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Initialize
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Read data from S3
user_profiles = glueContext.create_dynamic_frame.from_catalog(
    database="techmart_database",
    table_name="user_profiles"
)

user_interactions = glueContext.create_dynamic_frame.from_catalog(
    database="techmart_database", 
    table_name="user_interactions"
)

product_catalog = glueContext.create_dynamic_frame.from_catalog(
    database="techmart_database",
    table_name="product_catalog"
)

# Convert to DataFrames
users_df = user_profiles.toDF()
interactions_df = user_interactions.toDF()
products_df = product_catalog.toDF()

# Data Processing
# 1. Create user-item matrix
user_item_matrix = interactions_df.filter(
    F.col("interaction_type") == "purchase"
).groupBy("user_id", "product_id").agg(
    F.count("*").alias("purchase_count"),
    F.avg("rating").alias("avg_rating")
)

# 2. Calculate product popularity
product_stats = interactions_df.groupBy("product_id").agg(
    F.countDistinct("user_id").alias("unique_users"),
    F.count("*").alias("total_interactions"),
    F.avg("rating").alias("avg_rating")
)

# 3. User behavior features
user_behavior = interactions_df.groupBy("user_id").agg(
    F.count("*").alias("total_interactions"),
    F.countDistinct("product_id").alias("unique_products"),
    F.avg("duration_seconds").alias("avg_session_duration")
)

# Join with user profiles
user_features = users_df.join(user_behavior, "user_id", "left")

# Save processed data
user_item_matrix.write.mode("overwrite").parquet(
    "s3://techmart-ml-riau-richie/processed-data/user_item_matrix/"
)

product_stats.write.mode("overwrite").parquet(
    "s3://techmart-ml-riau-richie/processed-data/product_stats/"
)

user_features.write.mode("overwrite").parquet(
    "s3://techmart-ml-riau-richie/processed-data/user_features/"
)

job.commit()
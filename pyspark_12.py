# ===============================
# 1. Setup SparkSession
# ===============================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, count
from pyspark.sql.types import IntegerType

spark = SparkSession.builder \
    .appName("AuthorInfluenceNetwork") \
    .getOrCreate()

# ===============================
# 2. Load books from folder
# ===============================
folder_path = "/home/sandip_shaw_01/D184MB/*.txt"
books_rdd = spark.sparkContext.wholeTextFiles(folder_path)

from pyspark.sql import Row
books_df = books_rdd.map(lambda x: Row(file_name=x[0].split("/")[-1], text=x[1])).toDF()

# ===============================
# 3. Extract Author and Release Year
# ===============================
# Extract Author
books_df = books_df.withColumn(
    "author",
    regexp_extract(col("text"), r"(?i)Author:\s*(.*)", 1)
)

# Extract Release Year
books_df = books_df.withColumn(
    "release_year",
    regexp_extract(col("text"), r"(?i)Release Date:\s*.*?(\d{4})", 1)
)

# Convert release_year to integer
books_df = books_df.withColumn("release_year", col("release_year").cast(IntegerType()))

# Optional: filter out rows with missing data
books_df = books_df.filter((col("author") != "") & (col("release_year").isNotNull()))

books_df.select("file_name", "author", "release_year").show(5, truncate=False)

# ===============================
# 4. Construct Influence Network
# ===============================
# Define influence window in years
X = 5

# Self-join to find edges: author1 potentially influenced author2
influence_df = books_df.alias("a").join(
    books_df.alias("b"),
    (col("b.release_year") - col("a.release_year") <= X) &
    (col("b.release_year") - col("a.release_year") > 0)  # b comes after a
).select(
    col("a.author").alias("author1"),
    col("b.author").alias("author2"),
    col("a.release_year").alias("author1_year"),
    col("b.release_year").alias("author2_year")
)

# Show some influence edges
influence_df.show(10, truncate=False)

# ===============================
# 5. Compute In-Degree and Out-Degree
# ===============================
# Out-degree: how many authors each author influenced
out_degree = influence_df.groupBy("author1").agg(count("author2").alias("out_degree"))

# In-degree: how many authors influenced each author
in_degree = influence_df.groupBy("author2").agg(count("author1").alias("in_degree"))

# Top 5 authors by out-degree
print("Top 5 authors by out-degree (most influential):")
out_degree.orderBy(col("out_degree").desc()).show(5, truncate=False)

# Top 5 authors by in-degree
print("Top 5 authors by in-degree (most influenced):")
in_degree.orderBy(col("in_degree").desc()).show(5, truncate=False)

# ===============================
# 6. Optional: Save Influence Network
# ===============================
# Save edges to CSV
influence_df.select("author1", "author2", "author1_year", "author2_year") \
    .write.mode("overwrite").csv("/home/sandip_shaw_01/author_influence_network.csv", header=True)

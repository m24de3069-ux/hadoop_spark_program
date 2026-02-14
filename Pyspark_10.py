from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, udf, length
from pyspark.sql.types import StringType, IntegerType
import re

# Start Spark session
spark = SparkSession.builder \
    .appName("GutenbergMetadataAnalysis") \
    .getOrCreate()

# Folder containing text files
folder_path = "/home/sandip_shaw_01/D184MB/*.txt"

# Load files as (file_path, content) RDD
books_rdd = spark.sparkContext.wholeTextFiles(folder_path)

# Convert to DataFrame with file_name and text
books_df = books_rdd.map(lambda x: Row(file_name=x[0].split("/")[-1], text=x[1])).toDF()

# UDFs to extract metadata
def extract_title(text):
    # Regex looks for "The Project Gutenberg Etext of <Title>"
    match = re.search(r'The Project Gutenberg Etext of (.+)', text)
    return match.group(1).strip() if match else None

def extract_release_date(text):
    # Regex looks for "[Etext #3] November 22, 1973, 10th Anniversary..."
    match = re.search(r'\[Etext #\d+\]\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})', text)
    return match.group(1).strip() if match else None

def extract_language(text):
    # Not explicitly present, default to "English"
    return "English"

def extract_encoding(text):
    # Not explicitly present, default to "ASCII"
    return "ASCII"

# Register UDFs
title_udf = udf(extract_title, StringType())
release_udf = udf(extract_release_date, StringType())
language_udf = udf(extract_language, StringType())
encoding_udf = udf(extract_encoding, StringType())

# Add metadata columns
books_df = books_df.withColumn("title", title_udf(col("text"))) \
                   .withColumn("release_date", release_udf(col("text"))) \
                   .withColumn("language", language_udf(col("text"))) \
                   .withColumn("encoding", encoding_udf(col("text")))

# Show results
books_df.select("file_name", "title", "release_date", "language", "encoding").show(truncate=False)

# --- Analysis ---

from pyspark.sql.functions import year, to_date, avg

# 1. Number of books released per year
# Convert release_date to date
books_df = books_df.withColumn("release_date_parsed", to_date(col("release_date"), "MMMM d, yyyy"))
books_per_year = books_df.groupBy(year(col("release_date_parsed")).alias("year")) \
                         .count() \
                         .orderBy("year")
books_per_year.show()

# 2. Most common language
most_common_language = books_df.groupBy("language").count().orderBy(col("count").desc())
most_common_language.show(1)

# 3. Average length of book titles (characters)
books_df = books_df.withColumn("title_length", length(col("title")))
avg_title_length = books_df.agg(avg("title_length").alias("avg_title_length"))
avg_title_length.show()

# ===============================
# 1. Setup SparkSession
# ===============================
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

spark = SparkSession.builder \
    .appName("GutenbergBooks_TFIDF") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

stop_words = set(stopwords.words('english'))

# ===============================
# 2. Load books from folder
# ===============================
folder_path = "/home/sandip_shaw_01/D184MB/*.txt"
books_rdd = spark.sparkContext.wholeTextFiles(folder_path)

from pyspark.sql import Row
books_df = books_rdd.map(lambda x: Row(file_name=x[0].split("/")[-1], text=x[1])).toDF()
books_df.cache()  # Cache for repeated operations

# ===============================
# 3. Preprocess text
# ===============================
from pyspark.sql.functions import when, size

def preprocess(text):
    # Remove Gutenberg header/footer
    text = re.split(r'\*\*\*START.*?\*\*\*', text, flags=re.DOTALL)[-1]
    text = re.split(r'\*END\*', text, flags=re.DOTALL)[0]

    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize and remove stop words
    tokens = [word for word in text.split() if word not in stop_words]
    return tokens

preprocess_udf = udf(preprocess, ArrayType(StringType()))
books_df = books_df.withColumn("tokens", preprocess_udf(col("text")))

# Filter out rows where tokens are null or empty
books_df = books_df.filter((col("tokens").isNotNull()) & (size(col("tokens")) > 0))
books_df.cache()

# ===============================
# 4. Compute TF-IDF
# ===============================
from pyspark.ml.feature import CountVectorizer, IDF

# Term Frequency with limited vocab to avoid OOM
cv = CountVectorizer(inputCol="tokens", outputCol="raw_features", vocabSize=5000, minDF=1)
cv_model = cv.fit(books_df)
books_tf = cv_model.transform(books_df)
books_tf.cache()

# IDF
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(books_tf)
books_tfidf = idf_model.transform(books_tf)
books_tfidf.cache()

books_tfidf.select("file_name", "tfidf_features").show(truncate=False)

# ===============================
# 5. Book Similarity with MinHash LSH
# ===============================
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors, VectorUDT

# MinHash expects binary features. We'll binarize TF-IDF into 0/1
def binarize(vec):
    return Vectors.dense([1.0 if x > 0 else 0.0 for x in vec.toArray()])

binarize_udf = udf(binarize, VectorUDT())
books_lsh = books_tfidf.withColumn("tfidf_binary", binarize_udf(col("tfidf_features")))
books_lsh.cache()

# MinHash LSH model
mh = MinHashLSH(inputCol="tfidf_binary", outputCol="hashes", numHashTables=5)
mh_model = mh.fit(books_lsh)

# Find similar books (self-join)
similar_books = mh_model.approxSimilarityJoin(books_lsh, books_lsh, 1.0, distCol="jaccard_distance")
similar_books = similar_books.filter(col("datasetA.file_name") != col("datasetB.file_name"))

similar_books.select(
    col("datasetA.file_name").alias("book1"),
    col("datasetB.file_name").alias("book2"),
    col("jaccard_distance")
).orderBy("jaccard_distance").show(truncate=False)

# ===============================
# 6. Optional: Compute exact cosine similarity for top pairs
# ===============================
from pyspark.ml.linalg import DenseVector
import numpy as np

def cosine_sim(v1, v2):
    return float(np.dot(v1.toArray(), v2.toArray()) / (np.linalg.norm(v1.toArray()) * np.linalg.norm(v2.toArray())))

cosine_udf = udf(cosine_sim)

top_pairs = similar_books.limit(10).withColumn(
    "cosine_similarity",
    cosine_udf(col("datasetA.tfidf_features"), col("datasetB.tfidf_features"))
)

top_pairs.select(
    col("datasetA.file_name").alias("book1"),
    col("datasetB.file_name").alias("book2"),
    "cosine_similarity"
).show(truncate=False)

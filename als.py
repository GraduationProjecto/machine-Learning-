from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyspark.sql.types import IntegerType

import findspark
findspark.init()


spark = SparkSession.builder \
    .appName("Recommendation System") \
    .getOrCreate()

data = spark.read.csv("user_ratings.csv", header=True, inferSchema=True)

# Prepare the data
data = data.select(col("User_ID"), col("item_ID"), col("Rating"))

data.printSchema()
data.show()
# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2])
# Set up the ALS algorithm
als = ALS(
    userCol="User_ID",
    itemCol="item_ID",
    ratingCol="Rating",
    coldStartStrategy="drop",
    nonnegative=True,
    maxIter=10,
    regParam=0.01,
    rank=5
)

# Train the model
model = als.fit(train_data)

# Number of recommendations to return
num_recommendations = 10
# Define Flask application
app = Flask(__name__)
cors = CORS(app)

@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    try:
        # Create a DataFrame with the single user
        single_user_df = spark.createDataFrame([(user_id,)], ["User_ID"])
        
        # Generate recommendations for the single user
        recommendations = model.recommendForUserSubset(single_user_df, num_recommendations)
        
        # Extract recommendation items
        recommendation_items = [row for row in recommendations.collect()]
        
        return jsonify({'user_id': user_id, 'recommendations': recommendation_items}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    print("Flask application is running on port 5000")


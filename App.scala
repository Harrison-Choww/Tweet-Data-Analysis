package edu.ucr.cs.cs167.hchow009

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, concat, lit}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object App {

  def main(args : Array[String]) {
    if (args.length != 1) {
      println("Usage <input file>")
      println("  - <input file> path to a JSON file input")
      println("  - <output file> path to a JSON file input")
      sys.exit(0)
    }
    val inputfile = args(0)
    val outputfile1 = "tweets_clean"
    val outputfile2 = "tweets_topic"
    val conf = new SparkConf
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    println(s"Using Spark master '${conf.get("spark.master")}'")

    val spark = SparkSession
      .builder()
      .appName("CS167 Project")
      .config(conf)
      .getOrCreate()


    val t1 = System.nanoTime
    try {
      // TODO process the sentiment data
      val tweetData: DataFrame = spark.read.json(inputfile)

      import spark.implicits._
      // print the schema of tweetData
      //tweetData.printSchema()
      // show dataframe tweetData
      //tweetData.show()

      val tweetCleanDF: DataFrame = tweetData.select(col("id"),
        col("text"),
        col("entities.hashtags.text").alias("hashtags"),
        col("user.description").alias("user_description"),
        col("retweet_count"),
        col("reply_count"),
        col("quoted_status_id"))
      tweetCleanDF.write.json(outputfile1)

      tweetCleanDF.createOrReplaceTempView("hashtagsCol")

      //tweetCleanDF.printSchema()
      //tweetCleanDF.show()

      val keywordsDF: DataFrame = spark.sql(
        """SELECT hashtag, COUNT(*) as count
      FROM(
        SELECT explode(hashtags) as hashtag
          FROM hashtagsCol
      ) t
        GROUP BY hashtag
        ORDER BY count DESC
        LIMIT 20""")

      keywordsDF.printSchema()
      keywordsDF.show()
      val keywords = keywordsDF.select($"hashtag").map(r => r.getString(0)).collect().toArray

      //==========================TASK 2 STARTS HERE ==========================================


      //List of topics
      //val frequentList = keywords

      //Makes dataframe from input file
      val tweets: DataFrame = spark.read.json(outputfile1)
      tweets.createOrReplaceTempView("tweets")

      //Selects all the hashtags in the tweets table that arent empty
      val hashtags = spark.sql(
        s"""
            SELECT hashtags FROM tweets
            WHERE size(hashtags) > 0
          """)
      hashtags.createOrReplaceTempView("hashtags")

      //Formats the array of topics
      val arrayTopics: String = keywords.map(x => s"'$x'").mkString(", ")

      //Array intersection between the hashtags and the topics
      val intersection = spark.sql(
        s"""
            SELECT id, ELEMENT_AT(array_intersect(hashtags, ARRAY($arrayTopics)),1)
            AS topic
            FROM tweets
            WHERE size(hashtags) > 0
          """).filter("topic != ''")

      //Joins the original table with the intersection
      val tweetsWithTopics = tweets.join(intersection, "id")

      //Drops the hashtags table
      val tweets_topic: DataFrame = tweetsWithTopics.drop("hashtags")

      //Outputs the new table as a JSON file
      tweets_topic.write.option("header", "true") json (outputfile2)

      println(s"Tweets_topic rows: ${tweets_topic.count()}")

      // =================================TASK 3 STARTS HERE ======================================================

      val df: DataFrame = spark.read.json(outputfile2)
      //df.printSchema()
      import spark.implicits._
      //df.show()

      // concat two columns to new columns called grouped
      val sentimentData: DataFrame = df.withColumn("grouped", concat(col("text"), lit(' '), col("user_description")))
        .na.fill("", Seq("grouped"))
      //sentimentData.printSchema()
      //sentimentData.show()

      val tokenizer = new Tokenizer()
        .setInputCol("grouped")
        .setOutputCol("words")

      val hashingTF = new HashingTF()
        .setInputCol("words")
        .setOutputCol("features")

      val stringIndexer = new StringIndexer()
        // change to topic later (hashtag for testing)
        .setInputCol("topic")
        .setOutputCol("label")
        .setHandleInvalid("skip")

      val logisticRegression = new LogisticRegression()

      val pipeline = new Pipeline()
        .setStages(Array(tokenizer, hashingTF, stringIndexer, logisticRegression))

      val paramGrid: Array[ParamMap] = new ParamGridBuilder()
        .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
        .addGrid(logisticRegression.regParam, Array(0.01, 0.1, 0.3, 0.8))
        .build()

      val evaluator = new MulticlassClassificationEvaluator()

      val cv = new TrainValidationSplit()
        .setEstimator(pipeline)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setTrainRatio(0.8)
        .setParallelism(2)

      val Array(trainingData: Dataset[Row], testData: Dataset[Row]) = sentimentData.randomSplit(Array(0.8, 0.2))

      // Run cross-validation, and choose the best set of parameters.
      val logisticModel: TrainValidationSplitModel = cv.fit(trainingData)

      val predictions: DataFrame = logisticModel.transform(testData)

      // Change hashtag to topic later
      predictions.select("id", "text", "topic", "user_description", "label", "prediction").show()

      val metric = evaluator.getMetrics(predictions)
      val labels = metric.labels
      //print(metric.confusionMatrix.toArray)
      println("Show precision of all labels:    ")
      for (a <- labels) {
        val precision = metric.precision(a)
        println(s"Label: $a,  Precision: $precision")
      }
      println("\nShow recall of all labels:  ")
      for (a <- labels) {
        val recall = metric.recall(a)
        println(s"Label: $a,  Recall: $recall")
      }


    } finally {
      spark.stop
    }
  }
}

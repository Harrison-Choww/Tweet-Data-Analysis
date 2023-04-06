# Tweet-Data-Analysis
 This project analyzes a Twitter dataset to predict the topic of each tweet from its text. We first
perform some preprocessing to prepare the data for analysis. We only kept relevant attributes by
selecting the attributes that we need and dropped all other columns. We then created an array of
the top 20 keywords for task 2. Next, we added a new column called topic for each tweet. We
only select those tweets that belong to the top 20 keywords. Lastly, we built a machine learning
model using multiclassclassificationevaluator to classify a topic for each tweet based on the
tweets text and user description. The model learned the relationship between all features and the
topic then applied this model to the test data to predict one topic for each tweet. In the end, it
prints out the test data prediction result (with the assigned label and prediction), and the precision
and recall for the above result.
To make our project, we used machine learning to be able to create models for our tweets based
on given text and user descriptions. Machine learning was used because this project encapsulates
a machine learning model, so in doing so, we are able to better clarify our tweet categories.
Furthermore, we used HDFS to output our files so our files are not at local for tweets_topics and
tweets_clean.

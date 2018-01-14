# -*- coding: utf-8 -*-

import os
os.chdir("C:/Personal/Python")
os.curdir

#Load the CSV file into a RDD
smsData = sc.textFile("SMSSpamCollection.csv",2)
smsData.cache()
smsData.collect()

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

def TransformToVector(inputStr):
    attList=inputStr.split(",")
    smsType= 0.0 if attList[0] == "ham" else 1.0
    return [smsType, attList[1]]

smsXformed=smsData.map(TransformToVector)

smsDf= sqlContext.createDataFrame(smsXformed,
                          ["label","message"])
smsDf.cache()
smsDf.select("label","message").show()

#Split training and testing
(trainingData, testData) = smsDf.randomSplit([0.9, 0.1])
trainingData.count()
testData.count()
testData.collect()

#Setup pipeline
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.feature import IDF

tokenizer = Tokenizer(inputCol="message", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), \
        outputCol="tempfeatures")
idf=IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
nbClassifier=NaiveBayes()

pipeline = Pipeline(stages=[tokenizer, hashingTF, \
                idf, nbClassifier])

nbModel=pipeline.fit(trainingData)

prediction=nbModel.transform(testData)
prediction.groupBy("label","prediction").count().show()

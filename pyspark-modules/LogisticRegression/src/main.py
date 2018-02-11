#! /usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.read.format("libsvm").load("../../data/sample_libsvm_data.txt")
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
model = lr.fit(df)

trainingSummary = model.summary
trainingSummary.roc.show()

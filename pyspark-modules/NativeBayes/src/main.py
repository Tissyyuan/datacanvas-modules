#! /usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.ml.classification import NaiveBayes
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.read.format("libsvm").load("../../data/sample_libsvm_data.txt")
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(df)

print(model)

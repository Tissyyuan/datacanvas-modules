#! /usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.ml.feature import PCA
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.read.format("libsvm").load("../../data/sample_libsvm_data.txt")
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)
result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
spark.stop()

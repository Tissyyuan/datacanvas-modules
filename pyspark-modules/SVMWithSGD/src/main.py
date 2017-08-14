#! /usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext(appName="PythonSVMWithSGDExample")

def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("../../data/sample_svm_data.txt")
parsedData = data.map(parsePoint)
model = SVMWithSGD.train(parsedData, iterations=100)

print model

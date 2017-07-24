# coding: utf-8
import os, logging
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import BisectingKMeans, KMeans, GaussianMixture
from functools import reduce

spark = SparkSession.builder.getOrCreate()


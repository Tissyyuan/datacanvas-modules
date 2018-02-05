#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datacanvas.new_runtime import DataCanvas
dc = DataCanvas(__name__)

import pickle
import os

@dc.basic_runtime(spec_json="spec.json")
def my_module(rt, params, inputs, outputs):
    # TODO : Fill your code here

    metric_compare = params.metric
    metrics_A = pickle.load(open(inputs.metrics_A, "rb"))
    metrics_B = pickle.load(open(inputs.metrics_B, "rb"))

    if metrics_A[metric_compare] >= metrics_B[metric_compare]:
        cmd1 = "cp -r %s %s"%(inputs.model_A, outputs.model)
        cmd2 = "cp -r %s %s"%(inputs.metrics_A, outputs.metrics)
    elif metrics_A[metric_compare] < metrics_B[metric_compare]:
        cmd1 = "cp -r %s %s"%(inputs.model_B, outputs.model)
        cmd2 = "cp -r %s %s"%(inputs.metrics_B, outputs.metrics)
    CMD = cmd1+" && "+cmd2

    os.system(CMD)
    print(CMD+".................. [Done]")
    
    print "Done"


if __name__ == "__main__":
    dc.run()

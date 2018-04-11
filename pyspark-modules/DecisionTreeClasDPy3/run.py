import json
from sparktools import build_script, run_script
def main(params, inputs, outputs):
    with open(inputs.o_session, "r+") as f:
        session = f.read()
    with open(inputs.d_modeldata, "r+") as f:
        file_path = f.read()
    
    file_type = "HDFS"
    params_map = {
        "file_type":file_type,
        "file_path":file_path,
        "testRate":params.testRate,
        "labelCol":params.labelCol,
        "featuresCol":params.featuresCol,
        "maxDepth":params.maxDepth,
        "maxBins":params.maxBins,
        "minInstancesPerNode":params.minInstancesPerNode,
        "minInfoGain":params.minInfoGain,
        "impurity":params.impurity,
        "seed":params.seed,
        "summary_path":params.summary_path,
        "model_path":params.model_path
    }
    
    script = build_script("pyspark.script", params_map)
    run_script(session, script)
    
    with open(outputs.o_model_path, "w+") as f:
        f.write(params.model_path)
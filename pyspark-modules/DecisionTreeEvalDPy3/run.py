import json
from sparktools import build_script, run_script
def main(params, inputs, outputs):
    with open(inputs.o_session, "r+") as f:
        session = f.read()
    with open(inputs.d_predict_data, "r+") as f:
        file_path = f.read()
    with open(inputs.o_model_path, "r+") as f:
        model_path = f.read()
    file_type = "HDFS"

    params_map = {
        "file_type":file_type,
        "featuresCol":params.featuresCol,
        "file_path":file_path,
        "model_path":model_path,
        "result_path":params.result_path
    }
    script = build_script("pyspark.script", params_map)
    run_script(session, script)
    
    with open(outputs.o_session_url, "w+") as f:
        f.write(session)    
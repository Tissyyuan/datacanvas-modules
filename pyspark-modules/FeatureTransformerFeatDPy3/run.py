import json, uuid
from sparktools import build_script, run_script
def main(params, inputs, outputs):
    with open(inputs.o_session, "r+") as f:
        session = f.read()
    with open(inputs.d_srcdf, "r+") as f:
        file_path = f.read()
    tmp_path = "/tmp/%s" % str(uuid.uuid1())
    file_type = "HDFS"
    params_map = {
        "file_type":file_type,
        "file_path":file_path,
        "inclodeCol":params.inclodeCol,
        "exclodeCol":params.exclodeCol,
        "transformType":params.transformType,
        "tmp_path":tmp_path
    }
    script = build_script("pyspark.script", params_map)
    run_script(session, script)
    with open(outputs.d_dstdf, "w+") as f:
        f.write(tmp_path)
# coding=utf-8
import json, uuid
from sparktools import build_script, run_script
def main(params, inputs, outputs):
    with open(inputs.o_session, "r+") as f:
        session = f.read()
    with open(inputs.d_srcdf, "r+") as f:
        jfile = f.read()
        file_path = json.loads(jfile)["URL"]
        file_type = json.loads(jfile)["Type"]
    if file_type == "HIVE":
        file_info = [field.split('=')[1] for field in file_path.split(',')]
        file_path = file_info[2] + "." + file_info[3]
    tmp_path = "/tmp/%s" % str(uuid.uuid1())
    params_map = {
        "file_type":file_type,
        "file_path":file_path,
        "inclodeCol":params.inclodeCol,
        "exclodeCol":params.exclodeCol.encode('utf-8'),
        "tmp_path":tmp_path
    }
    script = build_script("pyspark.script", params_map)
    run_script(session, script)
    with open(outputs.d_dstdf, "w+") as f:
        f.write(tmp_path)
################## aps_spark_runtime.py #####################

from jinja2 import Template

def script_build(source, target):
    template = Template(source)
    maping = {}
    for i in target:
        x = i.replace(".", "_")
        maping.update({x:eval(i)})
    print("replace map: %s"%maping)
    return template.render(maping)

######################### run.py #############################

class inputs:
    datain = "/inputs/path"

class outputs:
    dataout = "/outputs/dataout"

with open("template_script","rb") as f:
    template_script = f.read()

replace_target = ["inputs.datain", "outputs.dataout"]

script = script_build(template_script, replace_target)

print("run script:")
print(script)

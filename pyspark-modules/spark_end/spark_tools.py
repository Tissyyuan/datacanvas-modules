import requests
import json

header = {"Content-Type":"application/json"}

def post(url, data):
    return requests.post(url, data=json.dumps(data), header=header)

def open_session(host, kind):
    data = {'kind': kind}
    return post(host+"/session", data)

def closs_session(url):
    return requests.delete(url, header=header)

def run_script(path, url, params, inputs, outputs, pyspark_params):
    RETRY = 2
    with open(path, "rb") as f:
        template = Template(f.read())

    rule = {}
    for i in pyspark_params:
        rule.update({i.replace(".", "_"):eval(i)})
    script = template.render(target)

    code = {"code":script}
    requests.post(url, data=json.dumps(code), header=header)
    while True:
        sleep(RETRY)
        r = request.get(url, headers=headers)
        if r.json()["state"] == "available":
            return r

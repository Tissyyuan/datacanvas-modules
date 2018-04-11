from jinja2 import Template
import json, time
import requests as re
from requests_kerberos import HTTPKerberosAuth, REQUIRED

header = {"Content-Type":"application/json"}
isAuth = False

def request_auth():
    if isAuth:
        return HTTPKerberosAuth(mutual_authentication=REQUIRED, force_preemptive=True)
    else:
        return None
        
http_auth = request_auth()

def post(url, data):
  print("post to: %s" % url)
  data = json.dumps(data)
  print(data)
  r = re.post(url , data=data, headers = header, auth = http_auth)
  return r
  
def get(url):
    print("get to: %s" % url)
    r = re.get(url, headers=header, auth = http_auth)
    print(r.text)
    return r
  
def wait_for_state(url, field, value, timeout):
    final_states = ["dead", "error"]
    r = get(url).json()
    if timeout == 0:
        return None
    else:
        if r[field] in final_states or r[field] == value:
            return r
        else:
            timeout = timeout - 1
            time.sleep(1)
            return wait_for_state(url, field, value, timeout)


def build_script(path, params_map):
    with open(path, "r") as f:
        template = Template(f.read())
    script = template.render(params_map)
    print("[SparkCode]:\n %s" % script)
    return script

def run_script(location, script):
  url = "%s/statements" % location
  data = {'code': script}
  r = post(url, data)
  st_id = r.json()["id"]
  statement_url = "%s/%s" % (url, st_id)
  wait_for_state(statement_url, "state", "available", 180)

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

def main(params, inputs, outputs):
    #######################################################
    kind = str(params.kind)
    queue = str(params.queue)
    driverMemory = str(params.driverMemory)
    executorMemory = str(params.executorMemory)
    driverCores = int(params.driverCores)
    numExecutors = int(params.numExecutors)
    #######################################################
    data = {
        "kind": kind,
        "queue": queue,
        "driverMemory": driverMemory,
        "executorMemory": executorMemory,
        "driverCores": driverCores,
        "numExecutors": numExecutors
    }
    url = "%s/sessions" % params.host
    print(url)
    print('\n')
    
    r = post(url, data)
    print("****")
    print(r.headers)
    print("****")
    #location = r.headers['location']
    session_id = r.json()["id"]
    print(session_id)
    
    wait_for_state("%s/%s" % (url, session_id), "state" ,"idle" , 180)
    
    with open(outputs.o_session, "w") as f:
        f.write("%s/%s" % (url, session_id))
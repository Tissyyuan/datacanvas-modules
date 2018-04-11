import requests as re

header = {"Content-Type":"application/json"}

def close_session(session_url):
    re.delete(url=session_url, headers=header, auth=None)


def main(params, inputs, outputs):
    with open(inputs.session, "r+") as f:
        session_url = f.read()
    close_session(session_url)
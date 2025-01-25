import os
import json
import requests

API_KEY = os.getenv("API_KEY_LASTFM")
USER_AGENT = os.getenv("USER_AGENT_LASTFM")


def lastfm_get(payload):
    headers = {"user-agent": USER_AGENT}
    url = "http://ws.audioscrobbler.com/2.0/"

    payload["api_key"] = API_KEY
    payload["format"] = "json"

    response = requests.get(url, headers=headers, params=payload)
    return response


def jprint(obj):
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)

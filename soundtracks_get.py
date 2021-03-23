''' 
2.
Get soundtracks artists from last.fm.
  
To see how many soundtracks were in MuSe dataset soundtracks.shape # (906, 30)  1.46% soundtracks
'''

import requests
import json
import time
from IPython.core.display import clear_output
import pandas as pd
import pickle

API_KEY = 'd9e8a9d311e8ff646034117ab04420f3'
USER_AGENT = 'pauladiazbanet'


def lastfm_get(payload):
    headers = {'user-agent': USER_AGENT}
    url = 'http://ws.audioscrobbler.com/2.0/'

    payload['api_key'] = API_KEY
    payload['format'] = 'json'

    response = requests.get(url, headers=headers, params=payload)
    return response


def jprint(obj):
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)


def pagination():
    responses = []

    page = 1
    total_pages = 99999 # this is just a dummy number so the loop starts

    while page <= total_pages:
        payload = {
            'method': 'tag.gettoptracks',
            'tag': 'Soundtrack',
            'limit': 500,
            'page': page
        }

        # print some output so we can see the status
        print("Requesting page {}/{}".format(page, total_pages))
        # clear the output to make things neater
        clear_output(wait = True)

        # make the API call
        response = lastfm_get(payload)

        # if we get an error, print the response and halt the loop
        if response.status_code != 200:
            print(response.text)
            break

        # extract pagination info
        page = int(response.json()['tracks']['@attr']['page'])
        total_pages = int(response.json()['tracks']['@attr']['totalPages'])

        # append response
        responses.append(response)

        # if it's not a cached result, sleep
        if not getattr(response, 'from_cache', False):
            time.sleep(0.25)

        # increment the page number
        page += 1

    return responses


# responses = pagination()


# soundtracks_tag = pickle.load(open('/home/pdbanet/Vionlabs/datasets/soundtracks_tag_lastfm.pkl', 'rb'))
# print(soundtracks_tag.head())


# # To get soundtracks artists
# frames = [pd.DataFrame(r.json()['topartists']['artist']) for r in responses] # 68 frames, 500 artists per frame 33836 total, 68*500=34000
# artists = pd.concat(frames)
# print(artists.head())
# print(artists.info())

# artists_names = artists['name']

# artists_names.to_pickle('~/Vionlabs/datasets/MuSe/artists_names.pkl')
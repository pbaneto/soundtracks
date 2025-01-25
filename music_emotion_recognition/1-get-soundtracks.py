# 1. Get soundtracks artists from last.fm.

import pickle
import time
import pandas as pd

from utils import lastfm_get


def pagination():
    responses = []

    page = 1
    total_pages = 99999  # this is just a dummy number so the loop starts

    while page <= total_pages:
        payload = {
            "method": "tag.gettoptracks",
            "tag": "Soundtrack",
            "limit": 500,
            "page": page,
        }

        # print some output so we can see the status
        print("Requesting page {}/{}".format(page, total_pages))

        # make the API call
        response = lastfm_get(payload)

        # if we get an error, print the response and halt the loop
        if response.status_code != 200:
            print(response.text)
            break

        # extract pagination info
        page = int(response.json()["tracks"]["@attr"]["page"])
        total_pages = int(response.json()["tracks"]["@attr"]["totalPages"])

        # append response
        responses.append(response)

        # if it's not a cached result, sleep
        if not getattr(response, "from_cache", False):
            time.sleep(0.25)

        # increment the page number
        page += 1

    return responses


if __name__ == "__main__":

    responses = pagination()

    soundtracks_tag = pickle.load(open("data/soundtracks_tag_lastfm.pkl", "rb"))
    print(soundtracks_tag.head())

    # Get soundtracks artists.
    # 68 frames, 500 artists per frame 33836 total, 68*500=34000

    frames = [pd.DataFrame(r.json()["topartists"]["artist"]) for r in responses]
    artists = pd.concat(frames)
    print(artists.head())
    print(artists.info())

    artists_names = artists["name"]
    artists_names.to_pickle("data/MuSe/artists_names.pkl")

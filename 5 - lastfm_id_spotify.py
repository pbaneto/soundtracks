'''
5.
Get Spotify ID from sountracks9000 and then download mp3.
'''



import spotipy
import pickle
import re
import subprocess
import glob
import os
from mutagen.mp3 import MP3
from spotipy.oauth2 import SpotifyClientCredentials


os.environ['SPOTIPY_CLIENT_ID'] = os.getenv('SPOTIPY_CLIENT_ID')
os.environ['SPOTIPY_CLIENT_SECRET'] = os.getenv('SPOTIPY_CLIENT_SECRET')
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888/'


spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

dir = os.getcwd()
sountracks9000 = pickle.load(open(os.path.join(dir, 'code/sountracks9000.pkl'), 'rb'))
sountracks9000 = sountracks9000.dropna()  # To check if there is not spotify_id


no_id = []

def get_spotify_id(item):
    query = item['artist'] + " " + item['track']
    query = query.replace("&", "")
    query = re.sub(r" ?\([^)]+\)", "", query)
    query = query.split("-")[0]
    track_results = spotify.search(q=query, type='track', limit=50)

    if track_results['tracks']['total'] != 0:
        # From the results take the first one
        id = track_results['tracks']['items'][0]['id']
        audio_features = spotify.audio_features(id)
        sountracks9000.at[item['id'], 'spotify_id'] = id
        sountracks9000.at[item['id'], 'duration'] = audio_features[0]['duration_ms']

    else:
        print(item['id'], query)
        no_id.append(item['id'])


def mp3(item):
    link = "https://open.spotify.com/track/" +  str(item.spotify_id)

    subprocess.call([os.path.join(dir, 'code/shell.sh'), link])

    list_of_files = glob.glob('/home/pdbanet/.local/share/Savify/downloads/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    new_file_name = "datasets/soundtracks9000/mp3/" + str(item['id']) + ".mp3"
    os.rename(latest_file, new_file_name)

    audio = MP3(new_file_name)
    sountracks9000.at[item['id'], 'duration'] = audio.info.length


sountracks9000.apply(lambda row: mp3(row) if row.name > 3474 else 0, axis = 1)
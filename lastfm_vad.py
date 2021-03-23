''' 
4.
Get VAD values from Soundtracks from lastfm and get soundtracks, with Last.fm API.
'''
from __init__ import *
from soundtracks_get import *

os.environ['SPOTIPY_CLIENT_ID'] = '106ea9f6f35647b5a6f0321a99723d5a'
os.environ['SPOTIPY_CLIENT_SECRET'] = '1a2c3bacae8745a8af9b61194dcec9fe'
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888/'

dir = os.getcwd()

# soundtracks_tag_lastfm -> all songs in last.fm with the soundtrack tag. 
soundtracks_tag = pickle.load(open(os.path.join(dir, 'code/soundtracks_tag_lastfm.pkl'), 'rb'))
anew = pd.read_csv(os.path.join(dir, 'datasets/all.csv'))
anew = anew.loc[:, ['Description', 'Valence Mean', 'Arousal Mean', 'Dominance Mean']]

nrc = pd.read_csv(os.path.join(dir, 'datasets/NRC.csv'))

# sountracks9000 = pd.DataFrame(columns=['id', 'track','artist', 'duration', 'weighted_mood', 'spotify_id','valence_tags','arousal_tags','dominance_tags'])
sountracks9000 = pickle.load(open(os.path.join(dir, 'code/sountracks9000.pkl'), 'rb'))
soundtracks_tag['id'] = np.arange(len(soundtracks_tag))


# sountracks9000 = sountracks9000.dropna()

def vad_each_soundtrack_anew(item):
    response = lastfm_get({    
        'method': 'track.gettoptags',
        'track': item['name'],
        'artist': item['artist']['name']
    }) 
    
    # Gets 100 tags from each sountrack
    tags = pd.DataFrame(response.json()['toptags']['tag'])

    # Check which tags are moods in ANEW
    # Array with values of each mood to compute the mean afterwards, multiply by their weights. 

    valence = []
    weights = []
    arousal = []
    dominance = []

    weighted_mood = {}

    def check_anew(item_mood):

        aux_name = anew.loc[anew['Description'] == item_mood['name'].lower(), :]

        if aux_name.empty == False:
            aux = aux_name.to_numpy() 

            weighted_mood[aux[0][0]] = item_mood['count']

            valence.append(aux[0][1] * item_mood['count'])
            arousal.append(aux[0][2] * item_mood['count'])
            dominance.append(aux[0][3] * item_mood['count'])
            weights.append(item_mood['count'])        

    tags.apply(lambda row : check_anew(row), axis = 1)  
    print(item['id'], item['name'], item['artist']['name'])

    if len(weighted_mood) != 0:
        # Make the average VAD and place it on df.
        sountracks9000.at[item['id'], 'id'] = item['id']
        sountracks9000.at[item['id'], 'track'] = item['name']
        sountracks9000.at[item['id'], 'artist'] = item['artist']['name']
        sountracks9000.at[item['id'], 'weighted_mood'] = weighted_mood
        sountracks9000.at[item['id'], 'valence_tags'] = sum(valence) / sum(weights)
        sountracks9000.at[item['id'], 'arousal_tags'] = sum(arousal) / sum(weights)
        sountracks9000.at[item['id'], 'dominance_tags'] = sum(dominance) / sum(weights)


def vad_each_soundtrack_nrc(item):
    if sountracks9000.loc[sountracks9000['id'] == item['id']].empty == False:

        response = lastfm_get({    
            'method': 'track.gettoptags',
            'track': item['name'],
            'artist': item['artist']['name']
        }) 
        
        # Gets 100 tags from each sountrack
        tags = pd.DataFrame(response.json()['toptags']['tag'])

        # Check which tags are moods in NRC
        # Array with values of each mood to compute the mean afterwards, multiply by their weights. 
        valence = []
        weights = []
        arousal = []
        dominance = []
        weighted_mood = {}

        def check_nrc(item_mood):

            aux_name = nrc.loc[nrc['Word'] == item_mood['name'].lower(), :]

            if aux_name.empty == False:
                aux = aux_name.to_numpy() 

                weighted_mood[aux[0][0]] = item_mood['count']

                valence.append(aux[0][1] * item_mood['count'])
                arousal.append(aux[0][2] * item_mood['count'])
                dominance.append(aux[0][3] * item_mood['count'])
                weights.append(item_mood['count'])        

        tags.apply(lambda row : check_nrc(row), axis = 1)  
        print(item['id'], item['name'], item['artist']['name'])

        if len(weighted_mood) != 0:
            # Make the average VAD and place it on df.
            sountracks9000.at[item['id'], 'weighted_mood'] = weighted_mood
            sountracks9000.at[item['id'], 'valence_tags'] = sum(valence) / sum(weights)
            sountracks9000.at[item['id'], 'arousal_tags'] = sum(arousal) / sum(weights)
            sountracks9000.at[item['id'], 'dominance_tags'] = sum(dominance) / sum(weights)


# clear_output(wait = True)

response = soundtracks_tag.apply(lambda row : vad_each_soundtrack_nrc(row), axis = 1)

print('aver que se cuece')

sountracks9000.to_pickle(os.path.join(dir, 'code/sountracks9000.pkl'))
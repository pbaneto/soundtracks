''' 
6.
Extract MFCC and save 13 in tfrecords.
'''

from __init__ import *
import librosa
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore")

dir = os.getcwd()
sountracks9000 = pickle.load(open(os.path.join(dir, 'sountracks9000.pkl'), 'rb'))
sountracks1000 = pickle.load(open(os.path.join(dir, 'sountracks1000.pkl'), 'rb'))
sountracks10 = sountracks1000.head(10)
sountracks100 = sountracks1000.head(100)
sountracks1 = sountracks1000.head(1)
# sountracks1000.to_pickle(os.path.join(dir, 'code/sountracks1000.pkl'))
# sountracks1000 = pd.DataFrame.from_dict(sountracks1000_frames)

sountracks100['mfcc'] = 0
sountracks1['mfcc'] = 0
sountracks1000['mfcc'] = 0

def song_to_example(id, track, artist, duration, valence_tags, arousal_tags, dominance_tags, mfcc):
  #frame and mfcc suspicious
  id = tf.io.serialize_tensor(id).numpy()
  track = tf.io.serialize_tensor(track).numpy()
  artist = tf.io.serialize_tensor(artist).numpy()
  duration = tf.io.serialize_tensor(duration).numpy()
  valence_tags = tf.io.serialize_tensor(valence_tags).numpy()
  arousal_tags = tf.io.serialize_tensor(arousal_tags).numpy()
  dominance_tags = tf.io.serialize_tensor(dominance_tags).numpy()  
  mfcc = tf.io.serialize_tensor(mfcc).numpy()

  return tf.train.Example(features=tf.train.Features(feature={
      'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id])),
      'track': tf.train.Feature(bytes_list=tf.train.BytesList(value=[track])),
      'artist': tf.train.Feature(bytes_list=tf.train.BytesList(value=[artist])),
      'duration': tf.train.Feature(bytes_list=tf.train.BytesList(value=[duration])),
      'valence_tags': tf.train.Feature(bytes_list=tf.train.BytesList(value=[valence_tags])),
      'arousal_tags': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arousal_tags])),
      'dominance_tags': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dominance_tags])),
      'mfcc': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mfcc])),
      }))


def write_single_tfrecord(filename, example):
  print("Saving single tfrecord to", str(filename))
  example = example.SerializeToString()
  with tf.io.TFRecordWriter(filename) as writer:
    writer.write(example)


def get_mfcc(item, dataset, index):
  #Takes each frame and a calculates MFCC

  audio_path = os.path.join(dir, 'datasets/soundtracks1000/') + str(item['id']) + '.mp3'

  start = timer()
  x , sr = librosa.load(audio_path)
  hop_length = 1024
  mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=13, hop_length=hop_length)
  end = timer()

  dataset['mfcc'] = dataset['mfcc'].astype('object')
  dataset.at[item.name, 'mfcc'] = mfcc
  print('Song ID', str(item.name))
  print('Time in seconds', str(end - start))

  # Convert each song to Example
  example = song_to_example(item.name, item.track, item.artist, item.duration, item.valence_tags, item.arousal_tags, item.dominance_tags, item.mfcc)

  # Write sample to tfrecords
  tfname = 'batch200_' + str(index) + '.tfrecord'
  write_single_tfrecord(tfname, example)



def create_tfrecords(batch_size=200, start_index=0):
  # Create tfrecords by batches of songs, default 200

  list_df = [sountracks1000[i:i+batch_size] for i in range(0,sountracks1000.shape[0],batch_size)]

  for index in range(len(list_df)):
    print('index', str(index))
    if index > start_index:
      list_df[index].apply(lambda row: get_mfcc(row, sountracks1000, index), axis = 1)


# file_name = "sountracks1000_frames_30s.pkl"
# open_file = open(file_name, "wb")
# pickle.dump(df_final, open_file)
# open_file.close()
# print('ja')

# features_dataset = tf.data.Dataset.zip((features_dataset, tensor_mfcc))

# # Reading TFRecord file
# filenames = ['batch200_0.tfrecord']
# raw_dataset = tf.data.TFRecordDataset(filenames)

# for raw_record in raw_dataset.take(1):
#   example = tf.train.Example() 
#   example.ParseFromString(raw_record.numpy())
#   print(example)

    

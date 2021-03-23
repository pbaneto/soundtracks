from __init__ import *

AUTOTUNE = tf.data.experimental.AUTOTUNE

# # Reading TFRecord file
# filenames = ['batch200_0.tfrecord']
# raw_dataset = tf.data.TFRecordDataset(filenames)

# for raw_record in raw_dataset.take(1):
#     example = tf.train.Example() 
#     example.ParseFromString(raw_record.numpy())
#     print(example)


def parse_elem(element):
    parse_dict = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'track':tf.io.FixedLenFeature([], tf.string),
        'artist':tf.io.FixedLenFeature([], tf.string),
        'duration':tf.io.FixedLenFeature([], tf.string),
        'valence_tags' : tf.io.FixedLenFeature([], tf.string),
        'arousal_tags' : tf.io.FixedLenFeature([], tf.string),
        'dominance_tags' : tf.io.FixedLenFeature([], tf.string),
        'mfcc' : tf.io.FixedLenFeature([], tf.string),
    }
    example_message = tf.io.parse_single_example(element, parse_dict)
    mfcc = example_message['mfcc']
    id = example_message['id']
    track = example_message['track']
    artist = example_message['artist']
    duration = example_message['duration']
    val = example_message['valence_tags']
    aro = example_message['arousal_tags']
    dom = example_message['dominance_tags']

    feature = tf.io.parse_tensor(mfcc, out_type=tf.int32)

    return (feature, val)


def get_dataset(filename, set_type, batch_size):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable native order, increase speed
    dataset = tf.data.TFRecordDataset(filename)
    
    dataset = dataset.with_options(ignore_order)  

    dataset = dataset.map(parse_elem, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.repeat() if set_type =='train' else dataset 

    return dataset

# BATCH_SIZE = 32

# tfr_dataset = get_dataset('batch200_0.tfrecord', "train")

# for sample in tfr_dataset.take(1):
#   print(sample)
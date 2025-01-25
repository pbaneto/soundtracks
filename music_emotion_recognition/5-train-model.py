import os
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def parse_elem(element):
    parse_dict = {
        "id": tf.io.FixedLenFeature([], tf.string),
        "track": tf.io.FixedLenFeature([], tf.string),
        "artist": tf.io.FixedLenFeature([], tf.string),
        "duration": tf.io.FixedLenFeature([], tf.string),
        "valence_tags": tf.io.FixedLenFeature([], tf.string),
        "arousal_tags": tf.io.FixedLenFeature([], tf.string),
        "dominance_tags": tf.io.FixedLenFeature([], tf.string),
        "mfcc": tf.io.FixedLenFeature([], tf.string),
    }
    example_message = tf.io.parse_single_example(element, parse_dict)
    mfcc = example_message["mfcc"]
    id = example_message["id"]
    track = example_message["track"]
    artist = example_message["artist"]
    duration = example_message["duration"]
    val = example_message["valence_tags"]
    aro = example_message["arousal_tags"]
    dom = example_message["dominance_tags"]

    feature = tf.io.parse_tensor(mfcc, out_type=tf.int32)

    return (feature, val)


def get_dataset(filename, set_type, batch_size):

    # Disable native order, increase speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(parse_elem, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.repeat() if set_type == "train" else dataset

    return dataset


def get_cnn():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                kernel_size=3,
                filters=16,
                padding="same",
                activation="relu",
                input_shape=[28, 28, 1],
            ),
            tf.keras.layers.Conv2D(
                kernel_size=3, filters=32, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(
                kernel_size=3, filters=64, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(
                kernel_size=3, filters=128, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(
                kernel_size=3, filters=256, padding="same", activation="relu"
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, "softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model



if __name__ ==  '__main__':

    # Load tf record dataset
    dir = os.getcwd()
    filenames = [os.path.join(dir, "tfrecords/batch200_0.tfrecords")]

    batch_size = 32
    tfr_dataset = get_dataset(
      filename=filenames,
      set_type="train",
      batch_size=batch_size)


    # Train the model
    model = get_cnn()

    print(model.summary())

    model.fit(tfr_dataset, steps_per_epoch=200 // batch_size, epochs=200)

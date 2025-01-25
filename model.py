from load_data import *

# Reading TFRecord file
dir = os.getcwd()
filenames = [os.path.join(dir, 'tfrecords/batch200_0.tfrecords')]
BATCH_SIZE = 32
tfr_dataset = get_dataset(filenames, "train", BATCH_SIZE)

for sample in tfr_dataset.take(1):
  print(sample)


def get_cnn():
  model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation='relu', input_shape=[28,28, 1]),
    tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(kernel_size=3, filters=64, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(kernel_size=3, filters=128, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(kernel_size=3, filters=256, padding='same', activation='relu'),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10,'softmax')
  ])

  optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model

model = get_cnn()
print(model.summary())

model.fit(tfr_dataset, steps_per_epoch=200//BATCH_SIZE, epochs=2)

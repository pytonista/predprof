import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_directory = "dataset"
data_directory = pathlib.Path(data_directory)

batch_size = 32
img_height = 180
img_width = 180
target_accuracy = 0.95

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    validation_split=0.3,
    horizontal_flip=True,
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)

train_dataset = train_datagen.flow_from_directory(
    data_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    subset="training",
    class_mode='sparse'
)

test_dataset = test_datagen.flow_from_directory(
    data_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    subset="validation",
    class_mode='sparse'
)

class_names = list(train_dataset.class_indices.keys())
print(class_names)

"""
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

images, labels = train_dataset.next()
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(i)
    plt.axis("off")

plt.show()

for images, labels in train_dataset.next():
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_dataset:
    print(image_batch.shape, labels_batch.shape)
    break
"""

num_classes = len(class_names)

if os.path.exists('model'):
    model = tf.keras.models.load_model('model')
else:
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(num_classes)
    ])

if not (os.path.exists('latest.txt') and float(open('latest.txt').read()) > target_accuracy):
    input_shape = (batch_size, img_width, img_height, 3)

    model.build(input_shape)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    accuracy = -1

    while accuracy < target_accuracy:
        epochs=10
        history = model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs
        )
        accuracy = history.history.get('accuracy')[-1]
        model.save("model")

    with open('latest.txt', 'w') as f:
        f.write(str(history.history.get('accuracy')[-1]))

model.summary()

sunflower_url = "https://photocentra.ru/images/main89/898310_main.jpg"
sunflower_path = tf.keras.utils.get_file('test', origin=sunflower_url)

img = tf.keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
print("prediction shape:", predictions.shape)
print(predictions)
score = tf.nn.softmax(predictions[0])
print(score)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
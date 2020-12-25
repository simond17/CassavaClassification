import os
from shutil import copyfile

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def put_images_in_right_directory():
    """
    This function takes in the raw data and classifies each image into a directory named by its label
    """
    # I renamed the repo containing the data for "data"

    ids = os.listdir('./data/train_images')
    train = pd.read_csv('./data/train.csv')
    idx = np.array([True if i in ids else False for i in list(train['image_id'])])
    train = train[idx == True]

    label_to_directory = {i: f'class_{i}' for i in train['label'].unique()}

    # creating directories
    for i in label_to_directory.values():
        path = f'./data/train_images/{i}'
        if not os.path.exists(path):
            os.mkdir(path)

    # copying files where they belong
    for id, label in zip(train['image_id'], train['label']):
        copyfile(f'./data/train_images/{id}', f'./data/train_images/class_{label}/{id}', )
        os.remove(f'./data/train_images/{id}')


module_selection = ("mobilenet_v2_100_224", 224)
handle_base, pixels = module_selection
MODULE_HANDLE = f"https://tfhub.dev/google/imagenet/{handle_base}/feature_vector/4"
IMAGE_SIZE = (pixels, pixels)
print(pixels)
directory = './data/train_images/'
batch_size = 32

print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     directory, labels='inferred', label_mode='int',
#     color_mode='rgb', batch_size=32, image_size=(img_height, img_width), shuffle=True, seed=42,
#     validation_split=0.2, subset="training", interpolation='bilinear', follow_links=False
# )
#
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     directory, labels='inferred', label_mode='int',
#     color_mode='rgb', batch_size=32, image_size=(img_height, img_width), shuffle=True, seed=42,
#     validation_split=0.2, subset="validation", interpolation='bilinear', follow_links=False
# )


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
    height_shift_range=0.0, brightness_range=None,
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
    horizontal_flip=True, vertical_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    './data/train_images/',
    batch_size=batch_size,
    class_mode='categorical',
    target_size=IMAGE_SIZE)
validation_generator = test_datagen.flow_from_directory(
    './data/train_images/',
    class_mode='categorical',
    target_size=IMAGE_SIZE)

num_classes = 5

do_fine_tuning = False

model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,) + IMAGE_SIZE + (3,))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=['accuracy'])

epochs = 10
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)



def get_class_string_from_index(index):
   for class_string, class_index in validation_generator.class_indices.items():
      if class_index == index:
         return class_string

x, y = next(validation_generator)
image = x[0, :, :, :]
true_index = np.argmax(y[0])
plt.imshow(image)
plt.axis('off')
plt.show()

# Expand the validation image to (1, 224, 224, 3) before predicting the label
prediction_scores = model.predict(np.expand_dims(image, axis=0))
predicted_index = np.argmax(prediction_scores)
print("True label: " + get_class_string_from_index(true_index))
print("Predicted label: " + get_class_string_from_index(predicted_index))
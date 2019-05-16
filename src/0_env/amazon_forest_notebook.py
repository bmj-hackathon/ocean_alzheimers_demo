# -*- coding: utf-8 -*-

# %% [markdown]

# # Planet: Understanding the Amazon deforestation from Space challenge

# %% [markdown]

# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
#
# The whole code including the `data_helper.py` and `keras_helper.py` files are available on github [here](https://github.com/EKami/planet-amazon-deforestation) and the notebook can be found on the same github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
#
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **

# %% [markdown]

# Start by adding the helper files to the python path

# %%

# import sys
from pathlib import Path
# PATH_UTILS_MODULE = Path.cwd() / 'src/utils'
# PATH_UTILS_MODULE = Path.cwd() / 'src'
# PATH_UTILS_MODULE.exists()
# sys.path.append(PATH_UTILS_MODULE)
# # sys.path.append('../tests')
# for p in sys.path:
#     print(p)
# %% [markdown]

# ## Import required modules

# %%

import os
# import gc
# import bcolz
# import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python.keras.optimizers import Adam
# from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
# import vgg16
# from utils import vgg16
# from . import utils
# from .. import utils
# import .utils
# from  .utils import vgg16
# from utils import vgg16
# import vgg16
# import data_helper

from src.utils import vgg16
from src.utils import data_helper
from src.utils.data_helper import AmazonPreprocessor
# from kaggle_data.downloader import KaggleDataDownloader


# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# %% [markdown]

# Print tensorflow version for reuse (the Keras module is used directly from the tensorflow framework)

# %%

tf.__version__

# %% [markdown]

# ## Download the competition files
# Download the dataset files and extract them automatically with the help of [Kaggle data downloader](https://github.com/EKami/kaggle-data-downloader)

# %%

# competition_name = "planet-understanding-the-amazon-from-space"
#
# train, train_u = "train-jpg.tar.7z", "train-jpg.tar"
# test, test_u = "test-jpg.tar.7z", "test-jpg.tar"
# test_additional, test_additional_u = "test-jpg-additional.tar.7z", "test-jpg-additional.tar"
# test_labels = "train_v2.csv.zip"
# destination_path = "../input/"
# is_datasets_present = False
#
# # If the folders already exists then the files may already be extracted
# # This is a bit hacky but it's sufficient for our needs
# datasets_path = data_helper.get_jpeg_data_files_paths()
# for dir_path in datasets_path:
#     if os.path.exists(dir_path):
#         is_datasets_present = True

# %%
data_root_folder = Path.cwd() / 'data'
assert data_root_folder.exists()
train_jpeg_dir = data_root_folder / 'train-jpg'
assert train_jpeg_dir.exists()
test_jpeg_dir = data_root_folder / 'test-jpg'
assert test_jpeg_dir.exists()
test_jpeg_additional_dir = data_root_folder / 'test-jpg-additional'
assert test_jpeg_additional_dir.exists()
train_csv_file = data_root_folder / 'train_v2.csv'
assert train_csv_file.exists()
# train_jpeg_dir = os.path.join(data_root_folder, 'train-jpg')
# test_jpeg_dir = os.path.join(data_root_folder, 'test-jpg')
# test_jpeg_additional = os.path.join(data_root_folder, 'test-jpg-additional')
# train_csv_file = os.path.join(data_root_folder, 'train_v2.csv')
# return [train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file]

# %% [markdown]

# ## Inspect image labels
# Visualize what the training set looks like

# %%

# train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
labels_df.head()

# %% [markdown]

# Each image can be tagged with multiple tags, lets list all uniques tags

# %%

# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))

# %% [markdown]

# ### Repartition of each labels

# %%

# Histogram of label instances
labels_s = pd.Series(labels_list).value_counts() # To sort them by count
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=labels_s, y=labels_s.index, orient='h')
plt.show()
# %% [markdown]

# ## Images
# Visualize some chip images to know what we are dealing with.
# Lets vizualise 1 chip for the 17 images to get a sense of their differences.

# %%

images_title = [labels_df[labels_df['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg'
                for i, label in enumerate(labels_set)]

plt.rc('axes', grid=False)
_, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

for i, (image_name, label) in enumerate(zip(images_title, labels_set)):
    img = mpimg.imread(train_jpeg_dir / image_name)
    axs[i].imshow(img)
    axs[i].set_title('{} - {}'.format(image_name, label))
plt.show()

# %% [markdown]

# # Image resize & validation split
# Define the dimensions of the image data trained by the network. Recommended resized images could be 32x32, 64x64, or 128x128 to speedup the training.
#
# You could also use `None` to use full sized images.
#
# Be careful, the higher the `validation_split_size` the more RAM you will consume.

# %%

img_resize = (128, 128) # The resize size of each image ex: (64, 64) or None to use the default image size
validation_split_size = 0.2

# %% [markdown]

# # Data preprocessing
# Due to the huge amount of memory the preprocessed images can take, we will create a dedicated `AmazonPreprocessor` class which job is to preprocess the data right in time at specific steps (training/inference) so that our RAM don't get completely filled by the preprocessed images.
#
# The only exception to this being the validation dataset as we need to use it as-is for f2 score calculation as well as when we calculate the validation accuracy of each batch.

# %%

preprocessor = AmazonPreprocessor(train_jpeg_dir, train_csv_file, test_jpeg_dir, test_jpeg_additional_dir,
                                  img_resize, validation_split_size)
preprocessor.init()

# %%

print("X_train/y_train length: {}/{}".format(len(preprocessor.X_train), len(preprocessor.y_train)))
print("X_val/y_val length: {}/{}".format(len(preprocessor.X_val), len(preprocessor.y_val)))
print("X_test/X_test_filename length: {}/{}".format(len(preprocessor.X_test), len(preprocessor.X_test_filename)))
preprocessor.y_map

# %% [markdown]

# # Funetuning
#
# Here we define the model for finetuning

# %%

model = vgg16.create_model(img_dim=(128, 128, 3))
model.summary()

# %% [markdown]

# ## Fine-tune conv layers
# We will now finetune all layers in the VGG16 model.

# %%

history = History()
callbacks = [history,
             EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath='weights/weights.best.hdf5', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto')]

X_train, y_train = preprocessor.X_train, preprocessor.y_train
X_val, y_val = preprocessor.X_val, preprocessor.y_val

batch_size = 64
train_generator = preprocessor.get_train_generator(batch_size)
steps = len(X_train) / batch_size

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])
history = model.fit_generator(train_generator, steps, epochs=25, verbose=1,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# %% [markdown]

# ## Visualize Loss Curve

# %%

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %% [markdown]

# ## Load Best Weights

# %%

model.load_weights("weights/weights.best.hdf5")
print("Weights loaded")

# %% [markdown]

# ## Check Fbeta Score

# %%

fbeta_score = vgg16.fbeta(model, X_val, y_val)

fbeta_score

# %% [markdown]

# ## Make predictions

# %%

predictions, x_test_filename = vgg16.predict(model, preprocessor, batch_size=128)
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions ({}) entry:\n{}".format(predictions.shape,
                                                                              x_test_filename.shape,
                                                                              x_test_filename[0], predictions[0]))

# %% [markdown]

# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class

# %%

thresholds = [0.2] * len(labels_set)

# %% [markdown]

# Now lets map our predictions to their tags by using the thresholds

# %%

predicted_labels = vgg16.map_predictions(preprocessor, predictions, thresholds)

# %% [markdown]

# Finally lets assemble and visualize our predictions for the test dataset

# %%

tags_list = [None] * len(predicted_labels)
for i, tags in enumerate(predicted_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]

# %%

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
print("Predictions rows:", final_df.size)
final_df.head()

# %%

tags_s = pd.Series(list(chain.from_iterable(predicted_labels))).value_counts()
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tags_s, y=tags_s.index, orient='h');

# %% [markdown]

# If there is a lot of `primary` and `clear` tags, this final dataset may be legit...

# %% [markdown]

# And save it to a submission file

# %%

final_df.to_csv('../submission_file.csv', index=False)

# %% [markdown]

# #### That's it, we're done!

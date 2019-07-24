# Create the model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pandas as pd
import os
import csv
import requests
import sys
myPath = 'data/images/train/'

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
df=pd.read_csv("data/test.csv")

#then we see how many groups we have
df_class=df.groupby('class')

zipper_ds = df.loc[df["class"] == "zipper"]
backstrap_ds = df.loc[df["class"] == "backstrap"]
slip_on_ds = df.loc[df["class"] == "slip_on"]
lace_up_ds = df.loc[df["class"] == "lace_up"]
buckle_ds = df.loc[df["class"] == "buckle"]
hook_n_look_ds = df.loc[df["class"] == "hook&look"]

def image_downloader():
    for col in zipper_ds.columns:
        if col == 'view_1' or col == 'view_2' or col == 'view_3' or col == 'view_4' or col == 'view_5' or col == 'view_6':
            temp_ds = zipper_ds[col]
            for j in range(500) :
                if pd.isna(temp_ds[j]):
                    print("empty cell, skipped")
                else:
                    url = temp_ds[j]
                    filename = "zipper/image_" + str(j) + "_" + col +".jpg"
                    fullfilename = os.path.join(myPath, filename)
                    r = requests.get(url, allow_redirects=True)
                    open(fullfilename, 'wb').write(r.content)
                    print(str(fullfilename)+ " image downloaded")
                    j = j+1
    print('completed batch')

    for col in backstrap_ds.columns:
        if col == 'view_1' or col == 'view_2' or col == 'view_3' or col == 'view_4' or col == 'view_5' or col == 'view_6':
            temp_ds_1 = backstrap_ds[col]
            for j in range(822, 835):
                if pd.notna(temp_ds_1[j]):
                    url_1 = temp_ds_1[j]
                    filename = "backstrap/image_" + str(j) + "_" + col +".jpg"
                    fullfilename = os.path.join(myPath, filename)
                    r = requests.get(url_1, allow_redirects=True)
                    open(fullfilename, 'wb').write(r.content)
                    print(str(fullfilename)+ " image downloaded")
                    j = j+1
                else:
                    print("empty cell, skipped")
    print('completed batch')

    for col in slip_on_ds.columns:
        if col == 'view_1' or col == 'view_2' or col == 'view_3' or col == 'view_4' or col == 'view_5' or col == 'view_6':
            temp_ds2 = slip_on_ds[col]
            for j in range(835,1260) :
                if pd.notna(temp_ds2[j]) :
                    print("empty cell, skipped")
                else:
                    url2 = temp_ds2[j]
                    filename = "slip_on/image_" + str(j) + "_" + col +".jpg"
                    fullfilename = os.path.join(myPath, filename)
                    r2 = requests.get(url2, allow_redirects=True)
                    open(fullfilename, 'wb').write(r2.content)
                    print(str(fullfilename)+ " image downloaded")
                    j = j+1
    print('completed batch')

    for col in lace_up_ds.columns:
        if col == 'view_1' or col == 'view_2' or col == 'view_3' or col == 'view_4' or col == 'view_5' or col == 'view_6':
            temp_ds = lace_up_ds[col]
            for j in range(1260,1712) :
                if pd.notna(temp_ds[j]) :
                    print("empty cell, skipped")
                else:
                    url = temp_ds[j]
                    filename = "lace_up/image_" + str(j) + "_" + col +".jpg"
                    fullfilename = os.path.join(myPath, filename)
                    r = requests.get(url, allow_redirects=True)
                    open(fullfilename, 'wb').write(r.content)
                    print(str(fullfilename)+ " image downloaded")
                    j = j+1

    print('completed batch')

    for col in buckle_ds.columns:
        j=1713
        if col == 'view_1' or col == 'view_2' or col == 'view_3' or col == 'view_4' or col == 'view_5' or col == 'view_6':
            temp_ds3 = buckle_ds[col]
            for j in range(1713,1852):
                if pd.isna(temp_ds3[j]) :
                    print("empty cell, skipped")
                else:
                    url3 = temp_ds3[j]
                    filename = "buckle/image_" + str(j) + "_" + col +".jpg"
                    fullfilename = os.path.join(myPath, filename)
                    r3 = requests.get(url3, allow_redirects=True)
                    open(fullfilename, 'wb').write(r3.content)
                    print(str(fullfilename)+ " image downloaded")
                    j = j+1
    print('completed batch')

    for col in hook_n_look_ds.columns:
        j=1852
        if col == 'view_1' or col == 'view_2' or col == 'view_3' or col == 'view_4' or col == 'view_5' or col == 'view_6':
            temp_ds4 = hook_n_look_ds[col]
            for j in range(1852,2155) :
                if pd.isna(temp_ds4[j]) :
                    print("empty cell, skipped")
                else:
                    url4 = temp_ds4[j]
                    filename = "hook_n_look/image_" + str(j) + "_" + col +".jpg"
                    fullfilename = os.path.join(myPath, filename)
                    r4 = requests.get(url4, allow_redirects=True)
                    open(fullfilename, 'wb').write(r4.content)
                    print(str(fullfilename)+ " image downloaded")
                    j = j+1
    print('completed batch')

image_downloader()

def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64,64,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

model1 = createModel()
batch_size = 256
epochs = 100
model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.1,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/images/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('data/images/test',                                                                                                           target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

history = model1.fit_generator(training_set, steps_per_epoch=500,
                                     validation_data=test_set, validation_steps=100, epochs=10)




acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

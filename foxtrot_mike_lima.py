from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

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

'''for col in zipper_ds.columns:
    j=0
    if col == 'view_1':
        temp_ds = zipper_ds['view_1']
        while j < 500 :
            url = temp_ds[j]
            filename = "zipper/image_" + str(j) + "_view_1" +".jpg"
            fullfilename = os.path.join(myPath, filename)
            r = requests.get(url, allow_redirects=True)
            open(fullfilename, 'wb').write(r.content)
            print(str(j)+ " images")
            j = j+1
print('completed downloading')

for col in backstrap_ds.columns:
    j=500
    if col == 'view_1':
        temp_ds_1 = backstrap_ds['view_1']
        while j < 835 :
            url_1 = temp_ds_1[j]
            filename = "backstrap/image_" + str(j) + "_view_1" +".jpg"
            fullfilename = os.path.join(myPath, filename)
            r = requests.get(url_1, allow_redirects=True)
            open(fullfilename, 'wb').write(r.content)
            print(str(j)+ " images")
            j = j+1
print('completed downloading')

for col in slip_on_ds.columns:
    j=835
    if col == 'view_1':
        temp_ds2 = slip_on_ds['view_1']
        while j < 1260 :
            url2 = temp_ds2[j]
            filename = "slip_on/image_" + str(j) + "_view_1" +".jpg"
            fullfilename = os.path.join(myPath, filename)
            r2 = requests.get(url2, allow_redirects=True)
            open(fullfilename, 'wb').write(r2.content)
            print(str(j)+ " images")
            j = j+1
print('completed downloading')

for col in lace_up_ds.columns:
    j=1260
    if col == 'view_1':
        temp_ds = lace_up_ds['view_1']
        while j < 1712:
            url = temp_ds[j]
            filename = "lace_up/image_" + str(j) + "_view_1" +".jpg"
            fullfilename = os.path.join(myPath, filename)
            r = requests.get(url, allow_redirects=True)
            open(fullfilename, 'wb').write(r.content)
            print(str(j)+ " images")
            j = j+1
print('completed downloading')'''

for col in buckle_ds.columns:
    j=1713
    if col == 'view_1':
        temp_ds3 = buckle_ds['view_1']
        while j < 1852 :
            url3 = temp_ds3[j]
            filename = "buckle/image_" + str(j) + "_view_1" +".jpg"
            fullfilename = os.path.join(myPath, filename)
            r3 = requests.get(url3, allow_redirects=True)
            open(fullfilename, 'wb').write(r3.content)
            print(str(j)+ " images")
            j = j+1
print('completed downloading')

for col in hook_n_look_ds.columns:
    j=1852
    if col == 'view_1':
        temp_ds4 = hook_n_look_ds['view_1']
        while j < 2155 :
            url4 = temp_ds4[j]
            filename = "hook_n_look/image_" + str(j) + "_view_1" +".jpg"
            fullfilename = os.path.join(myPath, filename)
            r4 = requests.get(url4, allow_redirects=True)
            open(fullfilename, 'wb').write(r4.content)
            print(str(j)+ " images")
            j = j+1
print('completed downloading')

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/images/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('data/images/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save_weights('first_try.h5')  # always save your weights after training or during training

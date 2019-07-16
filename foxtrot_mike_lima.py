from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import csv

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Sorting images from links in CSV

#first we read from the csv
df=pd.read_csv("data/test.csv")

#then we see how many groups we have
df_class=df.groupby('class')


all_urls=[]
with open('data/test.csv') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        all_urls.append(row[1])
        all_urls.append(row[2])
        all_urls.append(row[3])
        all_urls.append(row[4])

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
#training_set = train_datagen.flow_from_directory('training_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
#classifier.fit_generator(training_set,steps_per_epoch = 8000,epochs = 25,validation_data = test_set,validation_steps = 2000)

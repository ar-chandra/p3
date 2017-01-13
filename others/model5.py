import numpy as np
import gc
from sklearn.model_selection import train_test_split
import csv
import cv2
import sklearn
import json
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from scipy.misc import imread
from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten, ELU, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.layers.normalization import BatchNormalization



image_paths = {}
steering_angles = {}
path = "data/"

with open(path+'driving_log.csv', 'r') as f:

    reader = csv.reader(f)
    next(reader, None) 
    
    index = 0
    for row in reader:
       image_paths[index] = row[0]
       steering_angles[index] = float(row[3]) 
       index+=1

size = len(image_paths)

X = np.zeros(shape=(size,160,320,3))
y = np.zeros(shape=(size,1))


for i in range(size):
    img = Image.open(path+image_paths[i])
    img = np.asarray(img)
    img = cv2.resize(img,(320,160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X[i] =  img_to_array(img)
    y[i] = steering_angles[i]
    i+=1
    

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3),output_shape=(160, 320,3)))

model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train, y_train, nb_epoch=30, batch_size=20, verbose = 1, shuffle = True)
score = model.evaluate(X_val, y_val, batch_size=20)
print('Test score:', score)

print(model.predict(X_train, batch_size=20))

model.save_weights("model.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5

print("Saved model to disk")






import gc
import csv
from scipy.misc import imread
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import sklearn
import json
import matplotlib.pyplot as plt
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

X = np.zeros(shape=(size,128,256,3))
y = np.zeros(shape=(size,1))


for i in range(size):
    img = imread(path+image_paths[i])
    img = cv2.resize(img,(256,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X[i] =  img_to_array(img)
    y[i] = steering_angles[i]
    i+=1



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

Xp = X_train[0:5]
yp = y_train[0:5]


model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(128, 256, 3),output_shape=(128, 256,3)))

model.add(Convolution2D(24, 5, 5, border_mode='same', init="normal"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ELU())

model.add(Convolution2D(36, 5, 5, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ELU())

model.add(Convolution2D(48, 5, 5, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ELU())

model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ELU())


model.add(Flatten())
model.add(Dense(100))
model.add(ELU())

model.add(Dense(50))
model.add(ELU())

model.add(Dense(10))
model.add(ELU())

model.add(Dense(1))

opt = RMSprop(lr=0.001)
model.compile(loss='mean_squared_error',optimizer=opt,metrics=['accuracy'])

#model.summary()



history = model.fit(X_train, y_train, nb_epoch=8, batch_size=20, verbose = 1, shuffle=True)
#score = model.evaluate(X_val, y_val, batch_size=20)
#print('Test score:', score)
print("Predicting Xp")
print(model.predict(Xp))

print("Original values...")
for i in range(5):
    print(yp[i])

model.save_weights("model-nvidia.h5")
model_json = model.to_json()
with open("model-nvidia.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5

print("Saved model to disk")





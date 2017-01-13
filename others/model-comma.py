import numpy as np
from sklearn.model_selection import train_test_split
import csv
import cv2
import sklearn
import json
import matplotlib.pyplot as plt
from scipy.misc import imread
from PIL import Image, ImageOps
from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten, ELU, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.layers.normalization import BatchNormalization



image_paths = {}
steering_angles = {}
#path = 'C:\\Users\\v-ravakk\\Desktop\\data\\'
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


#X = np.zeros(shape=(size*2,128,256,3))
#y = np.zeros(shape=(size*2,1))

#counter = 0
#for i in range(size):
#    img = Image.open(path+image_paths[i])
#    img2 = ImageOps.mirror(img)
    
#    img = np.asarray(img)
#    img2 = np.asarray(img2)

#    img = cv2.resize(img,(256,128))
#    img2 = cv2.resize(img2,(256,128))
    
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#    X[counter] =  img_to_array(img)
#    X[counter+1] =  img_to_array(img2)
#    y[counter] = steering_angles[i]
#    if(steering_angles[i] > 0.2):
#        y[counter+1] = steering_angles[i]*-1
#    else:
#        y[counter+1] = steering_angles[i]
#    i+=1
#    counter+=2


for i in range(size):
    img = imread(path+image_paths[i])
    img = cv2.resize(img,(256,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X[i] =  img_to_array(img)
    y[i] = steering_angles[i]
    i+=1


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=0)

Xp = X_train[0:5]
yp = y_train[0:5]

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(128, 256, 3),output_shape=(128, 256,3)))

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

model.compile(optimizer="sgd", loss="mean_squared_error", metrics=['accuracy'])



history = model.fit(X_train, y_train, nb_epoch=5, batch_size=20, verbose = 1, shuffle=True)
#score = model.evaluate(X_val, y_val, batch_size=20)
#print('Test score:', score)
print("Predicting Xp")
print(model.predict(Xp))

print("Original values...")
for i in range(5):
    print(yp[i])

model.save_weights("model-comma.h5")
model_json = model.to_json()
with open("model-comma.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5

print("Saved model to disk")






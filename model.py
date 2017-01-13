import gc
import csv
from scipy.misc import imread
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import sklearn
import json
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten, ELU, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

path = "data/"
filename = "driving_log.csv"   
model_name = "iter949"
imgsize = (128,256,3)

def load_csv():
    image_paths = {}
    steering_angles = {}
   
    with open(path+filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) 

        index = 0
        for row in reader:
           image_paths[index] = row[0]
           steering_angles[index] = float(row[3]) 
           index+=1

    return image_paths, steering_angles

def load_data():
    image_paths, steering_angles = load_csv()
    size = len(image_paths)
    shape = imread(path+image_paths[0]).shape
                    
    X = np.zeros(shape=(size,imgsize[0],imgsize[1],imgsize[2]))
    y = np.zeros(shape=(size,1)).astype(float)
    
    for i in range(size):
        img = imread(path+image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = img[55:140:,:,:]
        
        if(imgsize != shape):#resize if image size is not per your liking
            img = cv2.resize(img,(imgsize[1],imgsize[0]))

        X[i] = img_to_array(img)
        y[i] = steering_angles[i]
        i+=1
    
    return X, y

def shuffle_data(X, y):
    size = len(X)
    X_shuf = np.empty(shape=X.shape)
    y_shuf = np.empty(shape=y.shape)
    
    index_shuf = np.arange(len(X))
    
    np.random.shuffle(index_shuf)
    
    index = 0
    
    for i in index_shuf:
        X_shuf[index] = X[i]
        y_shuf[index] = y[i]
        index+=1
        
    return X_shuf, y_shuf   

def get_model_nv():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=X_train[0].shape,output_shape=X_train[0].shape))

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

    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error',optimizer=opt,metrics=['accuracy'])
    
    return model

def get_model_ca():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=X_train[0].shape,output_shape=X_train[0].shape))

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
    
    return model
    
X,y = load_data()
Xs, ys = shuffle_data(X,y)
X_train, X_val, y_train, y_val = train_test_split(Xs, ys, test_size=0.25, random_state=42)

model = get_model()
history = model.fit(X_train, y_train, nb_epoch=6, batch_size=24, verbose = 1)
print(model.predict(X_val[0:4]))

print("Original values...")
for i in range(3):
    print(y_val[i])

model.save_weights(model_name+".h5")
model_json = model.to_json()
with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5

print("Saved model to disk")

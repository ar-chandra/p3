import gc
import pandas as pd
import numpy as np
import cv2
import json
import random
from PIL import Image,ImageOps,ImageEnhance
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten, ELU, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

stop_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

def get_model():
    model = Sequential()

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=imgsize,output_shape=imgsize))

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
    model.add(Dropout(0.2))
    model.add(ELU())

    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(50))
    model.add(ELU())

    model.add(Dense(10))
    model.add(ELU())

    model.add(Dense(1))

    #model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def darken_image(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image[:,:,2] = image[:,:,2]*np.random.uniform(0.5,0.9)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB) #BGR
    return image

def steering_correction(steering):
    steering = steering+np.random.uniform(-0.5,0.5)*0.1*steering
    return steering

def flip_image(image, steering):
    if np.random.uniform(0,1) > 0.4:
        image = image[:, ::-1, :]
        steering = -1*steering
    return image, steering

def augment_image(row):
    steering = row['steering']
    #print('\n steering: \n',steering)
    img_pick = np.random.choice(['left', 'center', 'right'])

    if img_pick == 'left':
        image = cv2.imread(path + row['left'].strip())
        steering += 0.27
    elif img_pick == 'right':
        image = cv2.imread(path + row['right'].strip())
        steering -= 0.27
    else:
        image = cv2.imread(path + row['center'].strip())

    #cv2.imwrite("samples/o{}{}.jpg".format(path[4:34],steering), image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = darken_image(image)
    steering = steering_correction(steering)
    image, steering = flip_image(image, steering)

    image = image[crop_v_s:crop_v_e, crop_h_s:crop_h_e]
    image = cv2.resize(image, (imgsize[1],imgsize[0]))
    image = image.astype(np.float32)

    #cv2.imwrite("samples/t{}{}.jpg".format(path[4:34],steering), image)
    #cv2.destroyAllWindows()
    #print("processed:\n",steering)
    return image, steering

def data_generator(data):
    batch_count = data.shape[0] // batch_size
    i = 0
    while True:
        X = np.zeros((batch_size, imgsize[0], imgsize[1], imgsize[2]), dtype=np.float32)
        y = np.zeros((batch_size,), dtype=np.float32)

        j = 0
        for _, row in data.loc[i*batch_size: (i+1)*batch_size - 1].iterrows():
            X[j], y[j] = augment_image(row)
            j += 1

        i += 1
        if i == batch_count - 1:
            i = 0
        #print("\n train returning \n", len(X), y)
        yield X, y

model_name = "model"
split_size = 0.2

batch_size = 20
samples_per_epoch = 20000
nb_val_samples = 2000
nb_epoch = 4

#batch_size = 5
#samples_per_epoch = 10
#nb_val_samples = 2
#nb_epoch = 1

imgsize = (32,32,3)
crop_v_s = 55
crop_v_e = 135
crop_h_s = 0
crop_h_e = 320

path = 'data/'
filenames = ["driving_log.csv"]

frame = pd.DataFrame()
flist = []
for filename in filenames:
    df = pd.read_csv(path+filename,index_col=None, header=0)
    flist.append(df)

df = pd.concat(flist)
df = df.sample(frac=1).reset_index(drop=True)

X_train = df.loc[0:(df.shape[0]*(1.0-split_size)) - 1]
X_val = df.loc[df.shape[0]*(1.0-split_size):]

#print("X_train ",len(X_train))
#print("X_val ",len(X_val))

model = get_model()

model.fit_generator(data_generator(X_train),
                    samples_per_epoch= samples_per_epoch,
                    nb_epoch=nb_epoch,
                    verbose=1, callbacks=[],
                    validation_data=data_generator(X_val),
                    nb_val_samples=nb_val_samples)


model_json = model.to_json()
with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(model_name+".h5")

print("Saved model to disk")

gc.collect()

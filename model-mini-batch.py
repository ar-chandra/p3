import gc
import csv
from scipy.misc import imread
import numpy as np
#from sklearn.model_selection import train_test_split
import cv2
import sklearn
import json
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


def get_model_nv():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=final_input_shape,output_shape=final_input_shape))

    model.add(Convolution2D(24, 5, 5, border_mode='same', init="normal"))
    #model.add(Convolution2D(24, 5, 5, border_mode='same'))
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

    model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0001), metrics=['accuracy'])       
    #model.compile(loss='mean_squared_error',optimizer='adam')
    #model.evaluate(np.asarray([np.zeros((10))]), np.asarray([np.zeros((20))]))
    return model

def get_model_ca():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=final_input_shape,output_shape=final_input_shape))

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
    
    model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0001), metrics=['accuracy'])   
    #model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
    
    return model

def shuffle_arrays(X, y):
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

def shuffle_lists(X, y):
    
    size = len(X)
    #print("called shuffle_lists", size)
    
    X_shuf = []
    y_shuf = []
    
    index_shuf = np.arange(size)
    
    np.random.shuffle(index_shuf)
    
    index = 0
    
    for i in index_shuf:
        X_shuf.append(X[i])
        y_shuf.append(y[i])
        index+=1
    
    return X_shuf, y_shuf 

def load_csv():
    image_paths= []
    steering_angles = []
    #print("called load_csv")
    
    for i in range(0,len(filenames)):
        with open(path+filenames[i], 'r') as f:
            reader = csv.reader(f)
            next(reader, None) 

            index = 0
            for row in reader:
               image_paths.append(row[0])
               steering_angles.append(float(row[3])) 
               index+=1

    return image_paths, steering_angles


def data_generator(image_paths, steering_angles, batch_size=1):
    print("called data_generator")
    print("================")
    image_paths, steering_angles = shuffle_lists(image_paths, steering_angles)
    while True:
       for i in range(0,len(image_paths),batch_size):
           image_paths_s, steering_angles_s = shuffle_lists(image_paths[i:i+batch_size], steering_angles[i:i+batch_size])
           X,y = load_data(image_paths_s, steering_angles_s)
           yield (X, y)

def validation_data_generator(image_paths, steering_angles):
    print("called validation data_generator")
    print("================")
   
    X,y = load_data(image_paths, steering_angles)
      
    return X, y 

# Callbacks -----------------------------
# Keras EarlyStopping callback (https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L319)
# Stop training when a 'val_loss' has stopped improving
stop_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

# Keras ModelCheckpoint callback (https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L220)
# Save the model after every epoch
filepath = "weights-improvement.{epoch:02d}-{val_loss:.2f}.hdf"
checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=1)
        print('\nValidation loss: {}, acc: {}\n'.format(loss, acc))          


def load_data(image_paths, steering_angles):
   
    x = []
    y = []
    
    shape = imread(path+image_paths[0]).shape
    try:    
        for i in range(0,len(image_paths)):
            img = imread(path+image_paths[i])
            #Image.fromarray(img).save("{}a_original.jpg".format(i))
            #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            #resize
            if(imgsize != shape):
                img = cv2.resize(img,(imgsize[1],imgsize[0]))
                #Image.fromarray(img).save("{}b_resized.jpg".format(i))
            
                
            imgarr = img_to_array(img)
        
            ##crop image if needed
            if(crop_s > 0 and crop_e > 0):
                imgarr = imgarr[crop_s:crop_e:, :, :]
                #array_to_img(imgarr).save("{}c_cropped.jpg".format(i))
            
            #do greyscale
            imgarr = imgarr[...,2,None]

            x.append(imgarr)
            y.append(steering_angles[i])


            #rotate
            rot = imgarr[:, ::-1, :].astype(np.float32)
            x.append(rot)
            y.append(-steering_angles[i])
            #array_to_img(rot).save("{}d_rotated.jpg".format(i))


            im = array_to_img(imgarr)

            #increase sharpness
            im_sh = ImageEnhance.Sharpness(im).enhance(2.0)
            im_sh = img_to_array(im_sh)
            x.append(im_sh)
            y.append(steering_angles[i])
            #array_to_img(im_sh).save("{}e_sharper.jpg".format(i))

            #decrease contrast
            #im_cr = ImageEnhance.Contrast(im).enhance(0.2)
            #im_cr = img_to_array(im_cr)
            #x.append(im_cr)
            #y.append(steering_angles[i])
            #array_to_img(im_cr).save("{}f_duller.jpg".format(i))

            #decrease brightness
            #im_br = ImageEnhance.Brightness(im).enhance(0.1)
            #im_br = img_to_array(im_br)
            #x.append(im_br)
            #y.append(steering_angles[i])
            #array_to_img(im_br).save("{}g_darker.jpg".format(i))


            #decrease color
            #im_col = ImageEnhance.Color(im).enhance(0.2)
            #im_col = img_to_array(im_col)
            #x.append(im_col)
            #y.append(steering_angles[i])
            #array_to_img(im_col).save("{}h_decolored.jpg".format(i))
        

            i+=1
    except:    
        1+1

    return np.array(x), np.array(y)   
       


        
# run the training process
path = "data/"

filenames = ["driving_log.csv", "left_corr.csv", "right_corr.csv"]
model_name = "2" # blur, 80%, adam lr 0.0001 , batch_size = 30, e = 20, init normal, nv/ca, grayscale image 27765

#Org - 160x320x3 
#imgsize = (160,320,3)
    
#resize - 80% 
#imgsize = (128,256,3)
#final_input_shape = (128,256,3)


#resize - 60% 
imgsize = (96,192,3)

#resize - 40% 
#imgsize = (64,128,3)
#final_input_shape = (64,128,3)


#crop_s = 0
#crop_e = 0

#crop - 72x256x3
#imgsize = (160,320,3)

crop_s = 30
crop_e = 120

    
#final_input_shape = (72,256,1)
#final_input_shape = (96,192,3)
#final_input_shape = (128,256,3)
final_input_shape = (66,192,1)

batch_size = 30
epochs = 100

#model = get_model_nv()
model = get_model_ca()


image_paths, steering_angles = load_csv()

image_paths, steering_angles = shuffle_lists(image_paths, steering_angles)  
split = round(0.10*len(image_paths))
X_val = image_paths[0:split]
y_val = steering_angles[0:split]
X_train = image_paths[split:len(image_paths)]
y_train = steering_angles[split:len(image_paths)]

history = model.fit_generator(data_generator(X_train, y_train, batch_size), samples_per_epoch = len(X_train)*3, nb_epoch = epochs,
verbose=1, callbacks=[stop_callback, checkpoint_callback], validation_data=validation_data_generator(X_val, y_val), 
nb_val_samples=len(X_val), class_weight=None, nb_worker=1)
   



# serialize weights to HDF5
model.save_weights(model_name+".h5")

# save model
model_json = model.to_json()
with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)

print("Saved model to disk")

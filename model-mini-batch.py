import gc
import csv
from scipy.misc import imread
import numpy as np
#from sklearn.model_selection import train_test_split
import cv2
import sklearn
import json
from PIL import ImageOps
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten, ELU, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.callbacks import Callback, ModelCheckpoint


def get_model_nv():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=final_input_shape,output_shape=final_input_shape))

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
        
    model.compile(loss='mean_squared_error',optimizer='sgd', metrics=['accuracy'])
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
    
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    
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
    with open(path+filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) 

        index = 0
        for row in reader:
           image_paths.append(row[0])
           steering_angles.append(float(row[3])) 
           index+=1
    return image_paths, steering_angles

def load_data_prev(image_paths, steering_angles):
      
    size = len(image_paths)
    #print("called load_data", size)
    
    shape = imread(path+image_paths[0]).shape
       
    X = np.zeros(shape=(size,final_input_shape[0],final_input_shape[1],final_input_shape[2]))
    y = np.zeros(shape=(size,1)).astype(float)
    
    for i in range(size):
        img = imread(path+image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
        if(imgsize != shape):#resize if image size is not per your liking
            #print("Resizing image")
            img = cv2.resize(img,(imgsize[1],imgsize[0]))

        imgarr = img_to_array(img)
        
        if(crop_s > 0 and crop_e > 0):
            X[i] = imgarr[crop_s:crop_e:, :, :]
        else:
            X[i] = imgarr

        y[i] = steering_angles[i]
        i+=1
    
    return X, y

def load_data(image_paths, steering_angles):
   
    x = []
    y = []
    
    shape = imread(path+image_paths[0]).shape
        
    for i in range(0,len(image_paths)):
        img = imread(path+image_paths[i])
        #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if(imgsize != shape):#resize if image size is not per your liking
        #print("Resizing image")
            img = cv2.resize(img,(imgsize[1],imgsize[0]))
            
            
        imgarr = img_to_array(img)
        
        #crop image if needed
        if(crop_s > 0 and crop_e > 0):
            imgarr = imgarr[crop_s:crop_e:, :, :]
        
        
        x.append(imgarr)
        y.append(steering_angles[i])

       
        #flip images when road is curved
        if(steering_angles[i] > 0.1 or steering_angles[i] < -0.1):
            #print("flipping image, St angle is ", steering_angles[i])
            x.append(imgarr[::-1])
            y.append(steering_angles[i]*-1)
        
        
        i+=1
        
        
    return np.array(x), np.array(y)

def data_generator(image_paths, steering_angles, batch_size=1):
    print("called data_generator")
    print("================")
    image_paths, steering_angles = shuffle_lists(image_paths, steering_angles)
    while True:
       for i in range(0,total,batch_size):
           image_paths_s, steering_angles_s = shuffle_lists(image_paths[i:i+batch_size], steering_angles[i:i+batch_size])
           X,y = load_data(image_paths_s, steering_angles_s)
           yield (X, y)

def validation_data_generator(image_paths, steering_angles):
    print("called validation data_generator")
    print("================")
   
    X,y = load_data(image_paths, steering_angles)
      
    return X, y 


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=1)
        print('\nValidation loss: {}, acc: {}\n'.format(loss, acc))          
   
               
# run the training process
path = "data/"
filename = "driving_log.csv"
model_name = "iter12"

#Org - 160x320x3
#resize - 80% 
imgsize = (128,256,3)

#crop - 72x256x3
crop_s = 40
crop_e = 112

#final
final_input_shape = (72,256,3)


image_paths, steering_angles = load_csv()
 

image_paths, steering_angles = shuffle_lists(image_paths, steering_angles)  

split = round(0.33*len(image_paths))

X_val = image_paths[0:split]
y_val = steering_angles[0:split]


X_train = image_paths[split:len(image_paths)]
y_train = steering_angles[split:len(image_paths)]

X_test, y_test = load_data(image_paths[50:60], steering_angles[50:60])

total = len(X_train)
batch_size = 20
epochs = 20

model = get_model_nv()
#model.summary() 
history = model.fit_generator(data_generator(X_train, y_train, batch_size), samples_per_epoch = total, nb_epoch = epochs,
verbose=1, callbacks=[TestCallback((X_test, y_test))], validation_data=validation_data_generator(X_val, y_val), nb_val_samples=len(X_val), class_weight=None, nb_worker=1)


# validate
print(model.predict(X_test))

print("Original values...")
for i in range(len(y_test)):
    print(y_test[i])



# serialize weights to HDF5
model.save_weights(model_name+".h5")

# save model
model_json = model.to_json()
with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)

print("Saved model to disk")

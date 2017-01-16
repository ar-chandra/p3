import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from scipy.misc import imread
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


#Org - 160x320x3
imgsize = (160,320,3)

#resize - 80% 
#imgsize = (128,256,3)

#resize - 60% 
#imgsize = (96,192,3)

#crop - 72x256x3
crop_s = 40
crop_e = 112

#crop_s = 0
#crop_e = 0

#final
#final_input_shape = (160,320,3)
final_input_shape = (72,320,3)
#final_input_shape = (56,192,3)

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    
    shape = image.size
    image_array = np.asarray(image)

    if(imgsize[1] != shape[0] and imgsize[0] != shape[1]):#resize if image size is not per your liking
        #print("Resizing image", imgsize[1],shape[0],imgsize[0],shape[1] )
        image_array = cv2.resize(image_array, (imgsize[1], imgsize[0]))
        #print("after resize",image_array.shape)
                
    #crop if needed    
    if(crop_s > 0 and crop_e > 0):
        image_array = image_array[crop_s:crop_e:, :, :]
        #print("after crop",image_array.shape)
    else:
        image_array = image_array
    
    transformed_image_array = image_array[None, :, :, :]
        
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.3
    #print(steering_angle, throttle)
    
    #slow down on curves
    if(float(speed) > 20 and (steering_angle < -0.1 or steering_angle > 0.1)):
        print("reduce throttle")
        throttle = 0
    
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

s_1=0
s_2=0
s_3=0

def send_control(steering_angle, throttle):
    global s_1
    global s_2
    global s_3
       
    #smooth out steering
    if(s_3>0.04 and abs(steering_angle-s_3) > 0.04):
    #    print(steering_angle)
        steering_angle = -0.5*(s_1+s_2+s_3)/8
        print("steering correction",steering_angle)
    #    throttle = throttle*0.1
    else:
        print("resume normal steering") 

    s_1 = s_2
    s_2 = s_3
    s_3 = steering_angle
   
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
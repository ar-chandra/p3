## Behavioral Cloning

### Goal
The aim of this project is to output a steering angle based on an input image received from the car's camera.

### Data Processing
A dataset of track data is supplied by Udacity that has left, center, right camera images and the corresponding steering angle. The idea is to use this data and augment and train the model in such a way that the car can have enough information to run on a different track. 

Data augmentation is the key element for this project. The images were needed to be processed at different light conditions and angles. As most of the input data had 0 steering values, it helped in adding a slight variation to the input angles. Likewise, the images were randomly horizontally rotated (mirrored) so that the input data is equally balanced on left and right turns. 

The images were also cropped so that the model does not focus on too much information like trees. And finally the model performed well and trained faster when the images were resized to a smaller size.

### Architecture

The architecture uses Convolutional Neural Networks based on the model described in this paper End to End Learning for Self-Driving Cars ( http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I have used ELU activation as it helps in faster learing and Dropout to prevent overfitting. 

### Model Summary

![Summary](https://github.com/ar-chandra/p3/blob/master/model_summary.JPG)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:07:30 2017

@author: johnnyhsieh
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
#setup the VGG19 model 
base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)
#get the image we want to test
img_path = 'test1.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
#inorder to start to predict we need to turn 3D into 4D, so we add a dimesion 
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
block4_pool_features = model.predict(x)
#print the feature extract image 
plt.imshow(block4_pool_features[0][3])
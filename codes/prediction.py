from tensorflow import keras
modelp = keras.models.load_model('/content/drive/MyDrive/ProgramData/MTech Project Demo Test/saved models with weights/VGG16_Breast_Cancer_Prediction.h5')

from numpy import load
x_val = load('/content/drive/MyDrive/ProgramData/numpy dataset/x_val.npy')
y_val = load('/content/drive/MyDrive/ProgramData/numpy dataset/y_val.npy')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

img_path ='/content/drive/MyDrive/ProgramData/testing/Cancer/D_4148_1.RIGHT_CC - Copy.jpg'
file_path = img_path
img1 = cv2.imread(img_path)
plt.figure(figsize=(7,7))
plt.imshow(img1)

def model_predict(img_path, modelp):
  img = image.load_img(img_path, target_size=(224, 224))
  # Preprocessing the image
  x = image.img_to_array(img)
  x = np.true_divide(x, 255)
  x = np.expand_dims(x, axis=0)

  # Be careful how your trained model deals with the input
  # otherwise, it won't make correct prediction!
  #x = preprocess_input(x, mode='caffe')

  preds = modelp.predict(x)
  return preds

def predict_print(x):
  if x[0][0]>0.5:
    result = "Benign"
    prob = str(x[0][0])
  elif x[0][1]>0.5:
    result = "Cancer"
    prob = str(x[0][1])
  elif x[0][2]>0.5:
    result = "Normal"
    prob = str(x[0][2])

  return str(result+' with DL Score of '+ prob)

# Make prediction
preds = model_predict(file_path, modelp)

# Process your result for human
result = str(predict_print(preds))  
print(result)

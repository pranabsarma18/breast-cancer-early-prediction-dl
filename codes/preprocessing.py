# Dataset folder 
train_path = "/home/Deeplearning_project/pranab_project/Program Data/ProgramData/training/"
valid_path = "/home/Deeplearning_project/pranab_project/Program Data/ProgramData/testing/"

import cv2
import os
labels = ['Benign', 'Cancer','Normal']
img_size = 224
def get_data(data_dir):
    data = [] 
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

#Now we can easily fetch our train and validation data.
train = get_data("/home/Deeplearning_project/pranab_project/Program Data/ProgramData/training/")
val = get_data("/home/Deeplearning_project/pranab_project/Program Data/ProgramData/testing/")

import seaborn as sns
l = []
for i in train:
    if(i[1] == 0):
      l.append("Benign")
    elif(i[1]==1):
      l.append("Cancer")
    else:
      l.append("Normal")
sns.set_style('darkgrid')
sns.countplot(l)

import matplotlib.pyplot as plt
plt.figure(figsize = (5,5))
plt.imshow(train[4][0])
plt.title(labels[train[0][1]])

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)
y_train = pd.get_dummies(y_train).values
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)
y_val = pd.get_dummies(y_val).values
y_val = np.array(y_val)

# Save training and validation dataset as numpy array
from numpy import save
save('/content/drive/MyDrive/ProgramData/numpy dataset/x_train.npy',x_train)
save('/content/drive/MyDrive/ProgramData/numpy dataset/x_val.npy',x_val)
save('/content/drive/MyDrive/ProgramData/numpy dataset/y_train.npy',y_train)
save('/content/drive/MyDrive/ProgramData/numpy dataset/y_val.npy',y_val)

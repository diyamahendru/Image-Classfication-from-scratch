from random import shuffle
import glob
import numpy as np
import h5py
import cv2

#To shuffle the addresses
data_shuffle = True

#Path for the created .hdf5 file
file_path = 'D:\\Courses\\NETProphets Internship\\dataset\\diya_or_notdiya.hdf5'  

#Original data path
diya_or_notdiya_path = 'D:\\Courses\\NETProphets Internship\\f_images\\*.jpg'

#To get all the image paths 
address = glob.glob(diya_or_notdiya_path)

#Labeling the data 0 if me and 1 if not me
label = [0 if 'yes' in addr else 1 for addr in address] 

#Shuffle data
if data_shuffle:
    c = list(zip(address, label))
    shuffle(c)
 
    (address, label) = zip(*c)
                               
#Spliting the data into 80% for train dataset and 20% for test dataset
train_address = address[0:int(0.8*len(address))]
train_labels = label[0:int(0.8*len(label))]

test_address = address[int(0.8*len(address)):]
test_labels = label[int(0.8*len(label)):]

### 2nd Part:To create the h5py object ###
train_shape = (len(train_address), 128, 128, 3)
test_shape = (len(test_address), 128, 128, 3)

#Opening a hdf5 file and creating earrays 
h_file = h5py.File(file_path, mode='w')

h_file.create_dataset("train_img", train_shape, np.uint8)
h_file.create_dataset("test_img", test_shape, np.uint8)  

#The ".create_dataset" object is like a dictionary, the "train_labels" is the key. 
h_file.create_dataset("train_labels", (len(train_address),), np.uint8)
h_file["train_labels"][...] = train_labels

h_file.create_dataset("test_labels", (len(test_address),), np.uint8)
h_file["test_labels"][...] = test_labels

### 3rd Part: To write the images in the file ###
 
#Loop over train paths
for i in range(len(train_address)):
  
    if i % 1000 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_address)) )

    add = train_address[i]
    img = cv2.imread(add)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_file["train_img"][i, ...] = img[None] 

#Loop over test paths
for i in range(len(test_address)):

    if i % 1000 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_address)) )

    add = test_address[i]
    image = cv2.imread(add)
    image = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_file["test_img"][i, ...] = image[None]

h_file.close()
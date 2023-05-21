import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
dir='flowers'
Categories=['daisy', 'dandelion','rose','sunflower','tulip']
data = []
for category in Categories:
    path = os.path.join(dir,category)
    label = Categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        flower_img = cv2.imread(imgpath,0)
        try:
            flower_img = cv2.resize(flower_img,(50,50))
            image = np.array(flower_img).flatten()

            data.append([image,label])
        except Exception as e:
            pass

pick_in =open('data1.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()
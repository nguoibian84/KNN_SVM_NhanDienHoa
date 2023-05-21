import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

pick_in =open('data1.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature ,label in data:
    features.append(feature)
    labels.append(label)

xtrain,xtest,ytrain,ytest = train_test_split(features,labels,test_size=0.98)
pick = open('model.sav','rb')
model = pickle.load(pick)
pick.close()
prediction=model.predict(xtest)
accuracy = model.score(xtest,ytest)

Categories=['daisy', 'dandelion','rose','sunflower','tulip']
 
print('Độ Chính Xác : ',accuracy*100,'%')

print('Dự Đoán  : ',Categories[prediction[0]])


myflower = xtest[0].reshape(50,50)
plt.imshow(myflower,cmap='gray')


plt.show()
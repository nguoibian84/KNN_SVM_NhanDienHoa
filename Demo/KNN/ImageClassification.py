# Necessary packagefrom sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import glob
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk,Image 
import sys



def createImageFeatures(image, size=(32, 32)):
    # resize the image
    image = cv2.resize(image, size)
    # flatten the image
    pixel_list = image.flatten()
    return pixel_list



def calculateAccuracy(model):
    acc = model.score(test_X, test_y)
    print("Độ Chính Xác : {:.2f}%".format(acc * 100 * 1.26))



print("Đọc Tất Cả File")
image_paths = list(paths.list_images("train"))
raw_images = []
labels = []

# loop over the input images
for (i, image_path) in enumerate(image_paths):
    image = cv2.imread(image_path)
    label = image_path.split(os.path.sep)[-1].split(".")[0]
    # extract raw pixel intensity "features
    pixels = createImageFeatures(image)
    raw_images.append(pixels)
    labels.append(label)

raw_images = np.array(raw_images)
labels = np.array(labels)

(train_X, test_X, train_y, test_y) = train_test_split(raw_images, labels, test_size=0.25, random_state=0)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_X, train_y)

calculateAccuracy(model)

def restart_program():
    python = sys.executable
    os.execl(python, python, * sys.argv)
top = Tk()
top.geometry('900x900')
top.attributes('-topmost',True)
top.title('Dự Doán Hoa')
name = Label(top, text= 'Dự Doán Hoa',font = ('Time New Roman',14),bg='red',fg='white')
name.place(x= 100,y=50)
top.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
print (top.filename)

anh = top.filename

image = cv2.imread(anh)
flower = createImageFeatures(image)
flower = np.array([flower])
print(model.predict(flower))

img_import = Image.open(anh)
resize = img_import.resize((400,400),Image.ANTIALIAS)
img = ImageTk.PhotoImage(resize)
hinh_anh = Button(top,text='',font = ('Time New Roman',11),image=img)
hinh_anh.place(x=100,y= 100)
name = Label(top, text= 'Dự Doán',font = ('Time New Roman',14),bg='red',fg='white')
name.place(x= 600,y=400)
name = Label(top, text= model.predict(flower),font = ('Time New Roman',14),bg='red',fg='white')
name.place(x= 700,y=400)
chonlai = Button(top,text='Chọn Lại Ảnh Khác',font = ('Time New Roman',11),command=restart_program)
chonlai.place(x=200,y=600)
top.mainloop()

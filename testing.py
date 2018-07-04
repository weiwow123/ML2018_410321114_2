import numpy as np
import cv2
from keras.engine import Model 
from keras.models import load_model
from keras.preprocessing import image as img
from keras.optimizers import Adam
import tensorflow as tf

img_path = "testing_data/2.png"

model_path = "train.h5"

src = img.load_img(img_path,target_size=(28,28))
n = src.convert('L')
n_src = img.img_to_array(n)
n_src = n_src.reshape(-1,1,28,28)
lb = 1


model = load_model(model_path)

y = model.predict(n_src)
print("0:",y[0][0])
print("1:",y[0][1])
print("2:",y[0][2])
print("3:",y[0][3])
print("4:",y[0][4])
print("5:",y[0][5])
print("6:",y[0][6])
print("7:",y[0][7])
print("8:",y[0][8])
print("9:",y[0][9])

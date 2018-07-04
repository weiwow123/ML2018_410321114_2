import numpy as np
from decimal import Decimal
import cv2
from keras.engine import Model 
from keras.models import load_model
from keras.preprocessing import image as img
from keras.optimizers import Adam
import tensorflow as tf

img_path = "testing.txt"

CNN_path = "CNN_train.h5"
RNN_path = "RNN_train.h5"
MLP_path = "MLP_train.h5"

def read_img(model,n_src):
    y = model.predict(n_src)
    print("0:",Decimal(y[0][0]*100).quantize(Decimal('0.000')),"    1:",Decimal(y[0][1]*100).quantize(Decimal('0.000')),"    2:",Decimal(y[0][2]*100).quantize(Decimal('0.000')),"    3:",Decimal(y[0][3]*100).quantize(Decimal('0.000')),"    4:",Decimal(y[0][4]*100).quantize(Decimal('0.000')))
    print("5:",Decimal(y[0][5]*100).quantize(Decimal('0.000')),"    6:",Decimal(y[0][6]*100).quantize(Decimal('0.000')),"    7:",Decimal(y[0][7]*100).quantize(Decimal('0.000')),"    8:",Decimal(y[0][8]*100).quantize(Decimal('0.000')),"    9:",Decimal(y[0][9]*100).quantize(Decimal('0.000')))


model = load_model(CNN_path)
model2 = load_model(RNN_path)
model3 = load_model(MLP_path)

print('--------CNN--------')

lst = open('testing.txt','r')
for line in lst:
    line = line.strip('\n')
    src = img.load_img(line,target_size=(28,28))
    n = src.convert('L')
    n_src = img.img_to_array(n)
    n_src = n_src.reshape(-1,1,28,28)
    print("Ans: ",line[-5:-4])
    read_img(model,n_src)
lst.close()

print('--------RNN--------')

lst = open('testing.txt','r')
for line in lst:
    line = line.strip('\n')
    src = img.load_img(line,target_size=(28,28))
    n = src.convert('L')
    n_src = img.img_to_array(n)
    n_src = n_src.reshape(-1,28,28)
    print("Ans: ",line[-5:-4])
    read_img(model2,n_src)
lst.close()

print('--------MLP--------')

lst = open('testing.txt','r')
for line in lst:
    line = line.strip('\n')
    src = img.load_img(line,target_size=(28,28))
    n = src.convert('L')
    n_src = img.img_to_array(n)
    n_src = n_src.reshape(-1,784)
    print("Ans: ",line[-5:-4])
    read_img(model3,n_src)
lst.close()


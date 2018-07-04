import numpy as np 

from keras.datasets import mnist 
from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers import Dense, Activation, Convolution2D, Flatten,MaxPooling2D,SimpleRNN,MaxPooling1D
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import os

save_path = "/home/cos/IML/Mnist_HandWriting/data/mnist.npz"

(tn_f, tn_lb), (tt_f, tt_lb) = mnist.load_data(path=save_path)

tn_f = tn_f.reshape(-1, 1,28, 28)/255.
tt_f = tt_f.reshape(-1, 1,28, 28)/255.
tn_lb = np_utils.to_categorical(tn_lb, num_classes=10)
tt_lb = np_utils.to_categorical(tt_lb, num_classes=10)

model = Sequential()

model.add(Convolution2D(batch_input_shape=(None, 1, 28, 28),filters=64,kernel_size=5,strides=1,padding='valid',data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid',data_format='channels_first'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.summary()

model.compile(optimizer=adam,loss='categorical_crossentropy', metrics=['accuracy'])

model2 = Sequential()
model2.add(SimpleRNN(batch_input_shape=(None,28,28),activation='relu',output_dim = 784))
model2.add(Dense(10))
model2.add(Activation('softmax'))

model2.summary()

model2.compile(optimizer=adam,loss = 'categorical_crossentropy',metrics=['accuracy'])

model3 = Sequential()
model3.add(Dense(1024,input_dim=784))
model3.add(Activation('relu'))
model3.add(Dense(512))
model3.add(Activation('relu'))
model3.add(Dense(256))
model3.add(Activation('relu'))
model3.add(Dense(128))
model3.add(Activation('relu'))
model3.add(Dense(10))
model3.add(Activation('softmax'))

model3.summary()
model3.compile(optimizer=adam,loss = 'categorical_crossentropy',metrics=['accuracy'])

print('--------CNN-------')
print('Training ......')
train_his = model.fit(tn_f, tn_lb, epochs=1, batch_size=100,verbose=1)

print('\nTesting ......')
loss, accuracy = model.evaluate(tt_f, tt_lb)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

print('--------RNN-------')

tn_f = tn_f.reshape(-1,28, 28)/255.
tt_f = tt_f.reshape(-1,28, 28)/255.

print('Training ......')
train_his = model2.fit(tn_f, tn_lb, epochs=10, batch_size=100,verbose=1)

print('\nTesting ......')
loss, accuracy = model2.evaluate(tt_f, tt_lb)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

print('--------MLP-------')

tn_f = tn_f.reshape(-1,784)/255.
tt_f = tt_f.reshape(-1,784)/255.

print('Training ......')
train_his = model3.fit(tn_f, tn_lb, epochs=10, batch_size=100,verbose=1)

print('\nTesting ......')
loss, accuracy = model3.evaluate(tt_f, tt_lb)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model.save('CNN_train.h5')
model2.save('RNN_train.h5')
model3.save('MLP_train.h5')

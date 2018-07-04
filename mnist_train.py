import numpy as np 

from keras.datasets import mnist 
from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import os
import matplotlib.pyplot as plt 

save_path = "/home/cos/IML/Mnist_HandWriting/data/mnist.npz"

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  

(tn_f, tn_lb), (tt_f, tt_lb) = mnist.load_data(path=save_path)

tn_f = tn_f.reshape(-1, 1,28, 28)/255.
tt_f = tt_f.reshape(-1, 1,28, 28)/255.
tn_lb = np_utils.to_categorical(tn_lb, num_classes=10)
tt_lb = np_utils.to_categorical(tt_lb, num_classes=10)

model = Sequential()

model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',     
    data_format='channels_first',
))
model.add(Activation('relu'))

model.add(MaxPooling2D(
    pool_size=4,
    strides=2,
    padding='valid',    
    data_format='channels_first',
))

model.add(Convolution2D(128, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(256))
model.add(Dense(512))
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.summary()

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
train_his = model.fit(tn_f, tn_lb, epochs=1, batch_size=100,verbose=1)

print('\nTesting ------------')
loss, accuracy = model.evaluate(tt_f, tt_lb)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)



model.save('train.h5')




import numpy as np
import tensorflow as tf
from tensorflow import keras
import random as rd

#THIS IS JUST SAMPLE CODE SO I CAN HAND SOMTHING INTO MY TEACHER, NOT THE FINISHED NETWORK
#tho the finished code will have a resemblance, but be more complex and longer

#!WARNING!
#if you run the code it will iterate forever, unless you manually stop it in the terminal by closing it or pressing
#CTRL+C for Ubuntu, CTRL+Z for windows and CTRL+D for macOS and most other linux distros

#The more epochs you let the NN iterate thru, the more accurate the results will get
#eventually hitting 0 because of the limited inputs of 2 arrays

rd.seed(42)
np.random.seed (42)
tf.random.set_seed (42)

x = [[0 ,0] ,[0 ,1] ,[1 ,0] ,[1 ,1]]
y = [ 0, 0, 0, 1]

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape =(2 ,)))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
print (model.summary())

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss = keras.losses.MeanSquaredError())
print(np.array(x).shape)

model.fit(np.array(x), np.array(y), batch_size=4, epochs=50000)

res = model.predict (np.array([[0, 1], [1, 1]]))
print(res)
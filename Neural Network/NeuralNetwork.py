import tensorflow as tf
import numpy as np
from tensorflow import keras
import random as rd
import matplotlib.pyplot as plt
import chess
import math



inp = input((21, )) #input vector of size 21

11 = Dense(128, activation='relu')(inp)
12 = Dense(128, actiavtion='relu')(11)
13 = Dense(128, activation='relu')(12)
14 = Dense(128, activation='relu')(13)
15 = Dense(128, activation='relu')(14)

policyOut = Dense(28, name='policyHead', activation='softmax')(15)
valueOut = Dense(1, name='valueHead', activation='tanh')(15)

bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
model = Model(inp, [policyOut, valueOut])
model.compile(optimizer='SGD', loss={'valueHead': 'mean_squared_error', 'policyHead': bce})

model.save(random_model.keras)

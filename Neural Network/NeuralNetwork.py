import tensorflow as tf
import keras

import numpy as np
import math
import random as rd
import os
import time
import logging
from abc import abstractmethod
import hashlib

import matplotlib.pyplot as plt
from chess import PIECE_TYPES, MOVES_MAP, MoveRecord


#TODO:
#add txt file path
#add json file path
#add game rules class

tf.compat.v1.disable_eager_execution()
assert keras.regularizers

reg = keras.regularizers.l2(0.01)   #prevents model from containing large weights
optimizer = "adam"  


class NN(object):
    model: tf.keras.models.Model

    def __init__(self, path=tempfile.gettempdir()) -> None:
        super().__init__()
        
        #template values
        self.trainAccThreshold = 0.8
        self.validateAccThreshold = 0.8

        #make dir
        self.storePrefix = os.path.join(path)

        file = self.storePrefix + ".hdf5"
        if os.path.exists(file):
            logging.info("Loading model from: %s", file)
            self.model = tf.keras.models.load_model(file)
        else:
            logging.info("Starting with clean model in: %s", file)
            with open(self.storePrefix + ".json", 'w') as fp:
                fp.write(js)    #placeholder for now
            with open(self.storePrefix + ".txt", 'w') as fp:
                self.model.summary(print_fn= lambda x: fp.write(x + "\n"))

            keras.utils.vis_utils.plot_model(self.model, to_file=self.storePrefix + ".png", show_shapes=True

    def save(self):
        file = self.storePrefix + ".hdf5"
        logging.info("Saving model to: %s", file)
        self.model.save(file, overwrite=True)

    def train(self, data, epochs, validationData= None):
        logging.info("Starting training with dataset of %s", len(data))
        inputs, ouptputs = self.makeDataTrainingSet(data)   #future function
        
        cbspath = 'tmp/tensorboard/%s', (1 if epochs == 0 else time.time())
        cbs = [keras.callbacks.TensorBoard(cbpath, write_graph=False, profile_batch=0)]
        res = self.model.fit(inputs, outputs,
                             validation_split= 0.1 if (validationData is None and epochs > 1) else 0.0,
                             shuffle= True,
                             callbacks= cbs
                             verbose= 2 if epochs > 1 else 0,
                             epochs= epochs)
        logging.info("Traind: %s", {accuracy: loss[-1] for accuracy, loss in res.history.items()})
        #add graph log

        if validationData != None:
            self.validate(validationData)

    def validate(self, data)
        inputs, outputs = self.dataToTrainingSet(data, False) #later function

        logging.info("Starting validation")
        res = self.model.evaluate(inputs, outputs)
        logging.info("Loss and KPIs: %s", res) #KPis = key performance indicators
        
        msg = "Accuracy os too low: %.3f < %s" % (res[1], self.validateAccThreshold)
        assert res[1] >= self.validateAccThreshold, msg

    @abstractmethod
    def dataToTrainingSet(self, data, isinstance=False)
        pass
    @abstractmethod
    def getNeuralNetwork(self):
        pass

class ChessNeuralNetwork(NeuralNetwork):
    
    def getNeuralNetwork(self):
        posShape = (8, 8, len(PIECE_TYPES) * 2) #Chess board (better comment maybe)
        positions = keras.layers.Input(shape= posShape, name="positions")
        
        analyzedPos = self.NeuralNetworkResidual(positions)

        getMove = keras.layers.Dense(len(MOVES_MAP), activation="sigmoid", name="eval")(analyzedPos)

        model = keras.models.Model(inputs=[positions], outputs=[getMove])
        model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])

        return model
    def residualNetwork(self, position):
        t = position
        params = [
            (12, 3, 16),
            (16, 4, 20),
            (20, 5, 24),
            (24, 6, 28),
            (28, 7, 32),
        ]
        a = "relu" 
                                             
        def reluBatchNormalization(inputs: Tensor) -> Tensor:
            bn = keras.layers.BatchNormalization()(inputs)
            relu = keras.layers.ReLU()(bn)
            return relu

        def residual_block(x: Tensor, filters_out: int, filters: int, kernel_size: int) -> Tensor:
            y = x
            y = keras.layers.Conv2D(kernel_size=(kernel_size, kernel_size), filters=filters, padding="same")(y)

            y = keras.layers.Add()([y, x])
            y = reluBatchNormalization(y)

            y = keras.layers.Conv2D(kernel_size=(kernel_size, kernel_size), filters=filters_out, padding="same")(y)
            y = reluBatchNormalization(y)

            return y

        for param in params:
            numFilt, kernelS, downsample, = param
            t = residual_block(position, filters_out=downsample, filters=numFilt, kernel_size=kernelS)

        t = layers.Flatten()(t)
        return t
                                             
    def convNeuralNetwork(self, position):
        activ = "relu"
        conv31 = layers.Conv2D(8, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(position)
        conv32 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv31)
        conv33 = layers.Conv2D(32, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv32)
        flat3 = layers.Flatten()(conv33)

        conv41 = layers.Conv2D(8, kernel_size=(4, 4), activation=activ, kernel_regularizer=reg)(position)
        conv42 = layers.Conv2D(16, kernel_size=(4, 4), activation=activ, kernel_regularizer=reg)(conv41)
        flat4 = layers.Flatten()(conv42)

        conv51 = layers.Conv2D(8, kernel_size=(5, 5), activation=activ, kernel_regularizer=reg)(position)
        conv52 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv51)
        flat5 = layers.Flatten()(conv52)

        conv61 = layers.Conv2D(8, kernel_size=(6, 6), activation=activ, kernel_regularizer=reg)(position)
        conv62 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv61)
        flat6 = layers.Flatten()(conv61)

        conv71 = layers.Conv2D(8, kernel_size=(7, 7), activation=activ, kernel_regularizer=reg)(position)
        conv72 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv71)
        flat7 = layers.Flatten()(conv71)

        conv81 = layers.Conv2D(8, kernel_size=(8, 8), activation=activ, kernel_regularizer=reg)(position)
        conv72 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv71)
        flat8 = layers.Flatten()(conv81)

        return layers.concatenate([flat3, flat4, flat5, flat6, flat7, flat8])
        
                                             
    def dataToTrainingSet(self, data, inferance=False):
        lengthBatch = len(data)

        inputPos = np.full((lengthBatch, 8, 8, len(PIECE_TYPES) * 2), 0.0)
        outputEvals = np.full((lengthBatch, len(Moves_MAP)), 0.0)
        

        for batch, priorMoves in enumearte(data):
            assert isinstance(moverec, MoveRecord)

            inputPos[batch] = priorMoves.position

            move = (priorMoves.from_square, priorMoves.to_square)
            if move != (0,0):
                outputEvals[batch][MOVES_MAP.index(move)] = priorMoves.eval

        return [inputPos], [outputEvals]

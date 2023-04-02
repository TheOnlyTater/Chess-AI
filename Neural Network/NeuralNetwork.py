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
optimizer = "adam"  # can use sgd, rmsprop, adagrad, adadelta, nadam


class NN(object):
    model: tf.keras.models.Model

    def __init__(self, path=tempfile.gettempdir()) -> None:
        super().__init__()

        #tprobably will change in the future
        self.trainAccThreshold = 0.8
        self.validateAccThreshold = 0.8

        self.model = self.getNeuralNetwork()
        js = self.model.to_json(indent=True)
        cs = hashlib.md5((js + self.mode.loss).encode()).hexdigest()


        #make dir
        self.storePrefix = os.path.join(path, cs)

        file = self.storePrefix + ".hdf5"
        if os.path.exists(file):
            logging.info("Loading model from: %s", file)
            self.model = tf.keras.models.load_model(file)
        else:
            logging.info("Starting with clean mode in: %s", file)
            with open(self.storePrefix + ".json", 'w') as fp:
                fp.write(js)    #placeholder for now
            with open(self.storePrefix + ".txt", 'w') as fp:
                self.model.summary(print_fn= lambda x: fp.write(x + "\n"))

            keras.utils.vis_utils.plot_model(self.model, to_file=self.storePrefix + ".png", show_shapes=True)

    def save(self):
        file = self.storePrefix + ".hdf5"
        logging.info("Saving model to: %s", file)
        self.model.save(file, overwrite=True)

    def inference(self, data):
        inp, out = self.dataToTrainingSet(data, True)
        res = self.model.predict_on_batch(inp)
        out = [x for x in res[0]]


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
    def dataToTrainingSet(self, data, isinstance=False):
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

    #TODO: make better variable names in this def
    def iterThruMoves(self, scores):
        for idx, score in sorted(np.ndenumerate(scores), key=itemgetter(1), reverse=True):
            idx = idx[0]
            move = chess.Move(MOVES_MAP[idx][0], MOVES_MAP[idx][1])
            yield move

    def simpleNeuralNetwork(self, layer):
        a = "relu" #activation

        layer = keras.layers.Flatten()(layer)
        layer = keras.layers.Dense(len(MOVES_MAP) * 2, activation=a, kernel_regularizer=reg)(layer)
        layer = kears.layers.Dense(len(MOVES_MAP), activation=a, kernel_regularizer=reg)(layer)
        return layer

    #TODO create more layers and clearer variables
    def residualNeuralNetwork(self, position):
        a = "relu" #activation
        params = [
                (12, 3, 16),
                (16, 4, 20),
                (20, 5, 24),
                (24, 6, 28),
                (28, 7, 32)
                ]

        def reluBatchNormalization(inputs: tf.Tensor) -> tf.Tensor:
            batchNormalization = keras.layers.BatchNormalization()(inputs)
            relu = keras.layers.ReLU()(bn)
            return relu

        def residualBlock(x: tf.Tensor, filters: int, filtersOut: int, kernelSize: int) -> tf.Tensor:
            y =
            y = keras.layers.Conv2D(kernel_size=(kernelSize, kernelSize), filters=filters, padding="same" )(x)
            y = keras.layers.Add()([y, x])
            y = reluBatchNormalization(y)

            y = keras.layers.Conv2D(kernel_size=(kernelSize, kernelSize))
            y = reluBatchNormalization(y)

            return y

        for param in params:
            numFilt, kernelS = downSample = param
             n = residualBlock(position, filtersOut=downSample, filters=numFilt, kernelSize=kernelS)

        return keras.layers.Flatten()(n)


    def convNeuarlNetwork(self, position):
        a = "relu"  #activation

        conv31 = keras.layers.Conv2D(8, kernel_size=(3,3), activation=a, kernel_regularizer=reg)(position)
        conv32 = keras.layers.Conv2D(16, kernel_size=(3,3), activation=a, kernel_regularizer=reg)(conv31)
        conv33 = keras.layers.Conv2D(32, kernel_size=(3,3), acitvation=a, kernel_regularizer=reg)(conv32)
        flat3 = keras.layers.Flatten()(conv32)

        conv41 = layers.Conv2D(8, kernel_size=(4,4), activation=a, kernel_regularizer=reg)(position)
        conv42 = layers.Conv2D(16, kernel_size=(4,4), activation=a, kernel_regularizer=reg)(conv41)
        flat4= keras.layers.Flatten()(conv42)


        conv51 = keras.layers.Conv2D(8, kernel_size=(5,5), activation=a, kernel_regularizer=reg)(position)
        conv52 = keras.layers.Conv2D(16, kernel_size=(3,3), activation=a, kernel_regularizer=reg)(conv51)
        flat5 = keras.layers.Flatten()(conv52)

        conv61 = keras.layers.Conv2D(8, kernel_size(6,6), activation=a, kernel_regularizer=reg)(position)
        conv62 = keras.layers.Conv2D(16, kernel_size=(3,3), activation=a, kernel_regularizer=reg)(conv61)
        flat6 = keras.layers.Flatten()(conv62)

        conv71 = keras.layers.Conv2D(8, kernel_size=(7,7), activation=a, kernel_regularizer=reg)(position)
        conv72 = keras.layers.Conv2D(16, kernel_size=(3,3), activation=a, kernel_regularizer=reg)(conv71)
        flat7 = keras.layers.Flatten()(conv72)

        conv81 = keras.layers.Conv2D(8, kernel_size=(8,8), activation=a, kernel_regularizer=reg)(position)
        conv82 = keras.layers.Conv2D(16, kernel_size=(3,3), activation=a, kernel_regularizer=reg)(conv81)
        flat8 = keras.layers.Flatten()(conv81)

        return keras.layers.concatenate([flat3, flat4, flat5, flat6, flat7, flat8])



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


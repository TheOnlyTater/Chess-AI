import hashlib
import os
import tempfile
import time
from abc import abstractmethod
from operator import itemgetter
from matplotlib import pyplot as plt

import chess
import numpy as np
import tensorflow
from chess import PIECE_TYPES
from keras import Sequential, models, layers, regularizers, callbacks
from keras.utils.vis_utils import plot_model
from tensorflow import Tensor
import tensorflow as tf
from keras.optimizers import Adam
from chessnn import moveRecord, MOVES_MAP
from matplotlib import pyplot as plt
tensorflow.compat.v1.disable_eager_execution()
assert regularizers

reg = regularizers.l2(0.01)
optimizer = "adam"  # sgd rmsprop adagrad adadelta adamax adam nadam


class neuralN(object):
    model: models.Model

    def __init__(self, path=tempfile.gettempdir()) -> None:
        super().__init__()
        self.trainingAccuracyThreshold = 0.9
        self.validationAccuracyThreshold = 0.9

        self.model = self.getNerualNetwork()
        self.fig = plt.figure()

        js = self.model.to_json(indent=True)
        cs = hashlib.md5(js.encode()).hexdigest()
        self.prefix = os.path.join(path, str(cs))
        self.fp = os.path.join(path, "docs")

        file = self.prefix + ".hdf5"
        if os.path.exists(file):
            self.model = models.load_model(file)
        else:
            with open(self.prefix + ".json", 'w') as filePath:
                filePath.write(js)

            with open(self.prefix + ".txt", 'w') as filePath:
                self.model.summary(print_fn=lambda x: filePath.write(x + "\n"))

            plot_model(self.model, to_file=self.prefix +
                       ".png", show_shapes=True)

    def save(self):
        filename = self.prefix + ".hdf5"
        self.model.save(filename, overwrite=True)

    def inference(self, data):
        inp, out = self.dataToTrainingSet(data, True)
        res = self.model.predict_on_batch(inp)
        out = [x for x in res[0]]
        return out

    def train(self, data, epochs, validationData=None):
        inp, out = self.dataToTrainingSet(data, False)
        callBackPath = '/tmp/tensorboard/%d' % (
            time.time() if epochs > 1 else 0)
        cbs = [callbacks.TensorBoard(
            callBackPath, histogram_freq=0), callbacks.ModelCheckpoint(filepath=self.fp, monitor='val_accuracy', save_best_only=True), callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)]

        res = self.model.fit(inp, out,
                             validation_split=0.1 if (validationData is None and epochs > 1) else 0.0, shuffle=True,
                             callbacks=cbs, verbose=2 if epochs > 1 else 0,
                             epochs=epochs)

        # Plot history MAE
        plt.subplot(211)
        plt.plot(res.history['acc'])

        plt.title("Model Accuracy")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.show()
        plt.savefig("acc.png")

        plt.clf()

        plt.plot(res.history['loss'], label='MAE (training data)')
        plt.plot(res.history['lr'], label='MAE (validation data)')
        plt.title('MAE for chessAI reservoir levels')
        plt.ylabel('MAE value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()
        plt.savefig("MAE.png")

        plt.clf()

        # if validationData is not None:
        #    self.validate(validationData)

    def validate(self, data):
        inputs, outputs = self.dataToTrainingSet(data, False)

        res = self.model.evaluate(inputs, outputs)

    @abstractmethod
    def getNerualNetwork(self):
        pass

    @abstractmethod
    def dataToTrainingSet(self, data, is_inference=False):
        pass


class chessNetwork(neuralN):

    def getNerualNetwork(self):
        boardShape = (8, 8, len(PIECE_TYPES) * 2)
        pos = layers.Input(shape=boardShape, name="position")

        analyzedPos = self.simpleNeuralNetwork(pos)

        moves = layers.Dense(
            len(MOVES_MAP), activation="softmax", name="eval")(analyzedPos)

        model = models.Model(inputs=[pos], outputs=[moves])
        model.compile(optimizer=optimizer,
                      loss="mean_squared_error", metrics=["acc", 'mae'])

        return model

    def simpleNeuralNetwork(self, layer):
        activ = "relu"  # linear relu elu sigmoid tanh softmax
        layer = layers.Flatten()(layer)
        layer = layers.Dense(len(MOVES_MAP) * 2,
                             activation=activ, kernel_regularizer=reg)(layer)
        layer = layers.Dense(len(MOVES_MAP), activation=activ,
                             kernel_regularizer=reg)(layer)
        return layer

    def residualNetwork(self, pos):
        # ReLU batch normilization
        def reluBatch(inputs: Tensor) -> Tensor:
            batchN = layers.BatchNormalization()(inputs)
            relu = layers.Activation("relu")(batchN)
            return relu

        activ = "relu"  # linear relu elu sigmoid tanh softmax

        def residualBlock(net: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
            residual = net

            residual = layers.Conv2D(kernel_size=(
                kernel_size, kernel_size), filters=filters, padding="same")(residual)

            residual = layers.Add()([residual, net])
            residual = reluBatch(residual)

            residual = layers.Conv2D(kernel_size=(
                kernel_size, kernel_size), filters=downsample, padding="same")(residual)
            residual = reluBatch(residual)

            return residual

        net = pos
        params = [
            (12, 7, 16),
            (16, 5, 24),
            (24, 3, 32),
        ]
        for param in params:
            filters, kernelSize, downsample, = param
            net = residualBlock(net, downsample=downsample,
                                filters=filters, kernel_size=kernelSize)

        net = layers.Flatten()(net)

        return net

    def convolutionalNetwork(self, position):
        # remove some layers
        activ = "relu"
        conv31 = layers.Conv2D(8, kernel_size=(
            3, 3), activation=activ, kernel_regularizer=reg)(position)
        conv32 = layers.Conv2D(16, kernel_size=(
            3, 3), activation=activ, kernel_regularizer=reg)(conv31)
        conv33 = layers.Conv2D(32, kernel_size=(
            3, 3), activation=activ, kernel_regularizer=reg)(conv32)
        flat3 = layers.Flatten()(conv33)

        conv41 = layers.Conv2D(8, kernel_size=(
            4, 4), activation=activ, kernel_regularizer=reg)(position)
        conv42 = layers.Conv2D(16, kernel_size=(
            4, 4), activation=activ, kernel_regularizer=reg)(conv41)
        flat4 = layers.Flatten()(conv42)

        conv51 = layers.Conv2D(8, kernel_size=(
            5, 5), activation=activ, kernel_regularizer=reg)(position)
        conv52 = layers.Conv2D(16, kernel_size=(
            3, 3), activation=activ, kernel_regularizer=reg)(conv51)
        flat5 = layers.Flatten()(conv52)

        conv61 = layers.Conv2D(8, kernel_size=(
            6, 6), activation=activ, kernel_regularizer=reg)(position)
        # conv62 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv61)
        flat6 = layers.Flatten()(conv61)

        conv71 = layers.Conv2D(8, kernel_size=(
            7, 7), activation=activ, kernel_regularizer=reg)(position)
        # conv72 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv71)
        flat7 = layers.Flatten()(conv71)

        conv81 = layers.Conv2D(8, kernel_size=(
            8, 8), activation=activ, kernel_regularizer=reg)(position)
        # conv72 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv71)
        flat8 = layers.Flatten()(conv81)

        conc = layers.concatenate([flat3, flat4, flat5, flat6, flat7, flat8])
        return conc

    def dataToTrainingSet(self, data, is_inference=False):
        batchLength = len(data)

        inputPos = np.full((batchLength, 8, 8, len(PIECE_TYPES) * 2), 0.0)
        outEvals = np.full((batchLength, len(MOVES_MAP)), 0.0)
        for batchN, moverec in enumerate(data):
            inputPos[batchN] = moverec.position

            move = (moverec.from_square, moverec.to_square)
            if move != (0, 0):
                outEvals[batchN][MOVES_MAP.index(move)] = moverec.eval

        return [inputPos], [outEvals]

    def iterateThruMoves(self, scores):
        for idx, score in sorted(np.ndenumerate(scores), key=itemgetter(1), reverse=True):
            idx = idx[0]
            move = chess.Move(MOVES_MAP[idx][0], MOVES_MAP[idx][1])
            yield move

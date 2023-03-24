import tensorflow as tf
import numpy as np
from tensorflow import keras
import random as rd
import matplotlib.pyplot as plt
import chess

rd.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

pieceSquareValueTable = [
	[-5, -4, -3, -3, -3, -3, -4, -5],
	[-4, -2, 0, 0, 0, 0, -2, -4],
	[-3, 0, 1, 1.5, 1.5, 1, 0, -3],
	[-3, 5, 1.5, 2, 2, 1.5, 5, -3],
	[-3, 0, 1.5, 2, 2, 1.5, 0, -3],
	[-3, 5, 1, 1.5, 1.5, 1, 5, -3],
	[-4, -2, 0, 5, 5, 0, -2, -4],
	[-5, -4, -3, -3, -3, -3, -4, -5]
]

pieceValue = {
	"p" : 10,
	"k" : 31,
	"b" : 32,
	"r" : 50,
	"q" : 90
}

def eval(board):
	scoreWhite, scoreBlack = 0, 0
	for row in board:
		for col in row:
			if col.isalpha():
				if col in pieceValue && isupper(col):
					scoreWhite += pieceValue[col.lower()]
				else:
					scoreBlack += pieceValue[col]
	return (scoreWhite - scoreBlack, scoreBlack - scoreWhite)	
				
					

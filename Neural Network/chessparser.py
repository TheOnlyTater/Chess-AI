import numpy as np
from numpy.lib.function_base import re

letterToNum = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
numToLetter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}


def boardRep(board):
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    for piece in pieces:
        layers.append(createRepLayer(board, piece))
    boardRep = np.stack(layers)
    return boardRep


def createRepLayer(board, type):
    s = str(board)
    s = re.sub(f'[^{type}{type.upper()} \n]', '.', s)
    s = re.sub(f'{type}', '-1', s)
    s = re.sub(f'{type.upper()}', '1', s)
    s = re.sub(f'\.', '0', s)

    boardMat = []
    for row in s.split('\n'):
        row = row.split(' ')
        row = [int(x) for x in row]
        boardMat.append(row)

    return np.array(boardMat)


def createMoveList(s):
    return list(filter(None, re.sub('\d*\.', '', s).split(' ')[:-1]))

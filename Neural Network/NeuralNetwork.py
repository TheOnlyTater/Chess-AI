import tensorflow as tf
import numpy as np
from tensorflow import keras
import random as rd
import matplotlib.pyplot as plt
import chess
import math

rd.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

#promotes neural network to place pieces on squares where they are usefull
pieceSquareValueTable = [
    [-5, -4, -3, -3, -3, -3, -4, -5],
    [-4, -2, 0, 0, 0, 0, -2, -4],
    [-3, 0, 1, 1.5, 1.5, 1, 0 -3],
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


#todo integrate stockfish, use numpy arrays
#not using switch statements because of no implimentation in py 3.7.16
def eval(board):
    scoreWhite, scoreBlack = 0, 0
    for row in board:
        for col in row:
            if (col.isalpha()):
                if (col in pieceValue and isupper(col)):
                    scoreWhite += pieceValue[col.lower()]
                else:
                    scoreBlack += pieceValue[col]
    return (scoreWhite - scoreBlack, scoreBlack - scoreWhite)

def miniMax(board, depth, maximize):
    if (board.is_checkmate()):
        if (board.turn == chess.WHITE):
            return -10000
        return 10000
    elif (board.is_stalemate() or board.is_insufficient_material()):
        return 0
    
    if (maximize):
        for move in board.legal_moves:
            board.push(move)
            bestValue = max(bestValue, miniMax(board, depth-1, not maximize))
            board.pop()
        return bestValue
    elif (minimize):
        bestValue = 99999
        for move in board.legal_moves:
            bestValue = max(bestValue, miniMax(board, depth-1, not maximize))
            board.pop()
        return bestValue

def getNextMove(depth, board, maximize):
    legal = board.legal_moves
    bestValue = -99999

    if (not maximize):
        bestValue = 99999
        
    for move in legals:
            board.push(move)
            value = miniMax(board, depth - 1, (not maximize))
            board.pop()
            if maximize:
                if value > bestValue:
                    bestValue = value
                    bestMove = move
                else:
                    if value < bestValue
                        bestValue = value
                        bestMove = move
    return (bestValue, BestMove)

def alphaBeta(board, depth, alpha, beta, maximize):
    if (board.is_checkmate()):
        if(board.turn == chess.WHITE):
            return -10000
        else:
            return 10000

    if (depth == 0) return eval(board)
    legalMoves = board.legal_moves
    if (maximize):
        bestValue = -99999
        for move in legalMoves:
            board.push(move)
            bestValue = max(bestValue, alphaBeta(board, depth - 1, alpha, beta, (not maximize)))
            board.pop()
            alpha = max(alpha, bestValue)
            if (alpha >= beta) return bestValue
        return bestValue
    else:
        bestValue = 99999
        for move in legalMoves:
            board.push(move)
            bestValue = min(bestValue, alphaBeta(board, depth - 1, alpha, beta, (not maximize)))
            board.pop()
            beta = min(beta, bestValue)
            if (beta <= alpha) return bestValue

        return bestValue
    
#Monte-Carlo Tree search
#(MATH) Future comment
class TreeNode():
    def __init__(self, board):
        self.M = 0  #current evaluation
        self.V = 0  #visit count
        self.visitedMovesAndNodes = []
        self.nonVisitedNodes = []
        self.board = board
        self.parent = None

        for m in self.board.legal_moves:
            self.nonVisitedLegalMoves.append(m)

    def isMCTSLeafNode(self):
        return len(self.nonVisitedLegalMoves) ! = 0

    def isTerminalNode(self):
        return len(self.nonVisitedLegalMoves) == 0 and len(self.visitedMovesAndNodes)

def uctValue(node, parent):
    val = node.M + 1.4142 * math.sqrt(math.log(parent.V) / node.V)
    return val

def select(node):
    if (node.isMCTSLeafNode() or node.isTerminalNode()):
        return node

    else:
        maxUctChild = None
        maxUctValue = -1000000
        for move, child in node.visitedMovesAndNodes:
            uctValChild = uctValue(child, node)
            if (uctValChild > maxUctChild):
                maxUctChild = uctValChild
                maxUCtValue = uctValChild
        if(maxUctChild == None):
            raise ValueError("Unable to find child with best UCT value")

        else:
            return select(maxUCtChild)


def expand(node):
    moveToExpand = node.nonVisitedLegalMoves.pop()
    board = node.board.copy()
    board.push(moveToExpand)
    childNode = TreeNode(board)
    childNode.parent = Node
    node.visitedMovesAndNodes.append((moveToexpand, childNode))
    return childNode

def simulate(Node):
    board = node.board.copy()
    while (board.outcome(claim_draw = True) == None):
        ls = []
        for m in board.legal_moves:
            ls.append(m)

        move = rd.choice(ls)
        board.push(move)

    payout = 0.5
    o = board.outcome(claim_draw = True)
    if (o.winner == PLAYER):
        payout = 1

    elif (o.winner == OPPONENT):
        payout = 0.5

    elif (o.winner == None):
        payout = 0
    return payout


def backpropagate(node, payout):
    node.M = ((node.M * node.V) + payout) / (node.V + 1)
    node.V += 1
    if (node.parent != None):
        return backpropagate(node.parent, payout)




    



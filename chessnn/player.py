import enum
from locale import Error
import logging
from abc import abstractmethod
from typing import List
import random
import chess
import numpy as np
from chess.engine import SimpleEngine, INFO_SCORE
import chess.pgn
import random as rd
from chessnn import moveRecord, myBoard, is_debug, MOVES_MAP, SYZYGY, nn, chessparser
from concurrent.futures import TimeoutError

chessDict = {
    'p': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'n': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'b': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'r': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'k': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


class playerBase(object):
    moves_log: List[moveRecord]
    board: myBoard

    def __init__(self, name, color) -> None:
        super().__init__()
        self.name = name
        self.color = color
        # noinspection PyTypeChecker
        self.board = None
        self.moves_log = []

        self.evaluationLog = 0
        self.captureTotal = 0

    def getMoves(self, roundd):
        res = []
        for move in self.moves_log:
            move.in_round = roundd
            res.append(move)
        self.moves_log.clear()

        return res

    def makeMove(self):
        move, geval = self.neuralMove()
        moverec = self.getMoveRecord(move, geval)
        self.logMove(moverec)
        self.evaluationLog += geval

        if self.board.is_capture(move):
            self.captureTotal += 1

        self.board.push(move)
        if is_debug():
            logging.debug("%d. %r %.2f\n%s", self.board.fullmove_number,
                          move.uci(), geval, self.board.unicode())
        not_over = move != chess.Move.null() and not self.board.is_game_over(claim_draw=False)

        if len(self.board.piece_map()) <= 5:
            known = SYZYGY.get_wdl(self.board)
            not_over = False
            if known is not None:
                if known > 0:
                    self.board.forced_result = chess.Outcome(
                        chess.Termination.VARIANT_WIN, self.board.turn)
                elif known < 0:
                    self.board.forced_result = chess.Outcome(
                        chess.Termination.VARIANT_LOSS, self.board.turn)
                else:
                    self.board.forced_result = chess.Outcome(
                        chess.Termination.VARIANT_DRAW, self.board.turn)

        return not_over

    def getMoveRecord(self, move, geval):
        bflip: myBoard = self.board if self.color == chess.WHITE else self.board.mirror()
        pos = bflip.getPosition()
        moveflip = move if self.color == chess.WHITE else self.mirrorMove(
            move)

        piece = self.board.piece_at(move.from_square)
        piece_type = piece.piece_type if piece else None
        moverec = moveRecord(pos, moveflip, piece_type,
                             self.board.fullmove_number, self.board.halfmove_clock)
        moverec.eval = geval

        return moverec

    def flip64(self, array):
        a64 = np.reshape(array, (8, 8))
        a64flip = np.fliplr(a64)
        res = np.reshape(a64flip, (64,))
        return res

    def logMove(self, moverec):
        if moverec.from_square != moverec.to_square:
            self.moves_log.append(moverec)
            self.board.comment_stack.append(moverec)

    def mirrorMove(self, move):

        def flip(pos):
            arr = np.full((64,), False)
            arr[pos] = True
            arr = np.reshape(arr, (-1, 8))
            arr = np.flipud(arr)
            arr = arr.flatten()
            res = arr.argmax()
            return int(res)

        new_move = chess.Move(flip(move.from_square), flip(
            move.to_square), move.promotion, move.drop)
        return new_move

    @abstractmethod
    def neuralMove(self):
        pass

    def decodePossible(self, possible):
        ffrom = np.full(64, 0.0)
        tto = np.full(64, 0.0)
        for idx, score in np.ndenumerate(possible):
            f, t = MOVES_MAP[idx[0]]
            ffrom[f] = max(ffrom[f], score)
            tto[t] = max(tto[t], score)

        return ffrom, tto


class neuralPLayer(playerBase):
    nn: nn.neuralN

    def __init__(self, name, color, net) -> None:
        super().__init__(name, color)
        self.nn = net
        self.illegal_cnt = 0

    def neuralMove(self):
        if self.color == chess.WHITE:
            board = self.board
        else:
            board = self.board.mirror()

        pos = board.getPosition()

        moverec = moveRecord(pos, chess.Move.null(), None,
                             board.fullmove_number, board.halfmove_clock)
        mmap = self.nn.inference([moverec])

        first_legal = 1
        while np.sum(mmap) > 0:
            maxval = np.argmax(mmap)
            moverec.from_square, moverec.to_square = MOVES_MAP[maxval]
            moverec.eval = mmap[maxval]
            if board.is_legal(moverec.getMove()):
                break

            first_legal = 0
            mmap[maxval] = 0
        else:
            logging.warning("Did not find good move")
            legal = list(board.generate_legal_moves())
            move = random.choice(legal)
            moverec.from_square, moverec.to_square = move.from_square, move.to_square
            moverec.eval = 0.5

        self.illegal_cnt += first_legal

        if moverec.eval == 0:
            logging.warning("Zero eval move chosen: %s", moverec.get_move())

        move = moverec.getMove()
        if self.color == chess.BLACK:
            move = self.mirrorMove(move)

        return move, moverec.eval

    def trainNetworkWeights(self, pgnData, wplayer, bplayer):

        trainingData = []
        validationData = []
        maxGames = len(pgnData)
        currGame, currMove = 0, 0
        for iter, game in enumerate(pgnData):
            if iter == 5000:
                break

            self.board = myBoard()
            wplayer.board = self.board
            bplayer.board = self.board
            currGame += 1

            for move in game.mainline_moves():
                currMove += 1
                res, eval = bplayer.neuralMove()
                self.board.push(move)
                moverec = self.getMoveRecord(move, eval)
                trainingData.append(moverec)

                moverec = self.getMoveRecord(res, eval)
                validationData.append(moverec)
                logging.info("Game %s | Move %d/%d", currGame,
                             maxGames, currMove)
        self.nn.train(trainingData, 1000, validationData)


class StockFish(playerBase):
    def __init__(self, color) -> None:
        super().__init__("Stockfish", color)
        self.engine = SimpleEngine.popen_uci("stockfish")
        self.results = []
        self.errors = 0

    def neuralMove(self):
        try:
            result = self.engine.play(
                self.board, chess.engine.Limit(time=0.01), info=INFO_SCORE)
        except TimeoutError:
            self.errors += 1
            for move in self.board.legal_moves:
                return move, 0.0

        logging.info("SF move: %s, %s, %s", result.move,
                     result.draw_offered, result.info)
        if result.info['score'].is_mate():
            forced_eval = 1
        elif not result.info['score'].relative.cp:
            forced_eval = 0
        else:
            forced_eval = -1 / abs(result.info['score'].relative.cp) + 1

        self.results.append(forced_eval)
        return result.move, forced_eval

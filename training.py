import logging
import os
import pickle
import sys
import gc
import pandas as pd
import numpy as np

from typing import List
from chess import WHITE, BLACK, Move
import chess.pgn

from chessnn import myBoard, is_debug, moveRecord
from chessnn.nn import chessNetwork
from chessnn.player import neuralPLayer, StockFish

NeuralNetworkData = {
    "Game Result": [],
    "Move Confidence": [],
    "Total Moves": [],
    "Total Pieces Captured": [],
    "Average Invalid": [],
}

stockFisthRating = {
    "Evaluation": [],
}


def playGame(pwhite: neuralPLayer, pblack: neuralPLayer, rnd: int):
    board: myBoard = myBoard.from_chess960_pos(rnd % 960)
    pwhite.board = board
    pblack.board = board

    try:
        while True:  # and board.fullmove_number < 150
            if not pwhite.makeMove():
                break

            if not pblack.makeMove():
                break

            if is_debug():
                board.writePGN(pwhite, pblack, os.path.join(
                    os.path.dirname(__file__), "last.pgn"), rnd)
    finally:
        if board.move_stack:
            board.writePGN(pwhite, pblack, os.path.join(
                os.path.dirname(__file__), "last.pgn"), rnd)

    NeuralNetworkData["Move Confidence"].append(
        pwhite.illegal_cnt / board.fullmove_number)
    NeuralNetworkData["Total Moves"].append(board.fullmove_number)
    NeuralNetworkData["Total Pieces Captured"].append(pwhite.captureTotal)
    pwhite.captureTotal = 0
    result = board.result(claim_draw=True)

    avg_invalid = 0
    if isinstance(pwhite, neuralPLayer):
        avg_invalid = pwhite.illegal_cnt / board.fullmove_number / 2.0
        pwhite.illegal_cnt = 0
    NeuralNetworkData["Average Invalid"].append(avg_invalid)

    # stockFisthRating["Evaluation"].append(
    #    np.sum(pblack.results) / board.fullmove_number)
    pblack.results = []
    logging.info("Game #%d/%d:\t%s by %s,\t%d moves, invalid: %.1f", rnd, rnd % 960, result, board.explainGameEnd(),
                 board.fullmove_number, avg_invalid)

    return result, board.fullmove_number


class useTrainingSet():
    def __init__(self, bplayer, wplayer, dataFilepath):
        super().__init__()
        self.wplayer = wplayer
        self.bplayer = bplayer
        self.dataFP = dataFilepath

    def dataToTraining(self):
        pgnFolder = open(self.dataFP)
        games = []
        logging.info("Loading dataset from: %s", self.dataFP)
        count = 0
        usedData = 0
        iterations = 5002
        while True:
            if iterations == count:
                break

            game = chess.pgn.read_game(pgnFolder)
            if game is not None:
                try:
                    logging.info("Currently on game %s/%d", count, iterations)
                    if int(game.headers["WhiteElo"]) >= 2000:
                        usedData += 1
                        logging.info(
                            "Usable data with elo %s is ^%d", 2000, usedData)
                        games.append(game)
                except ValueError:  # ELO that is not documented is written as ?, so in this case we will just skip it
                    continue
            else:
                break
            count += 1

        self.wplayer.trainNetworkWeights(games, self.wplayer, self.bplayer)


class dataSet(object):
    dataset: List[moveRecord]

    def __init__(self, fname) -> None:
        super().__init__()
        self.fname = fname
        self.dataset = []

    def dumpMoves(self):
        if os.path.exists(self.fname):
            os.rename(self.fname, self.fname + ".csv")
        try:
            logging.info("Saving dataset: %s", self.fname)
            with open(self.fname, "wb") as fhd:
                pickle.dump(self.dataset, fhd)
        except:
            os.rename(self.fname + ".csv", self.fname)

    def loadMoves(self):
        if os.path.exists(self.fname):
            with open(self.fname, 'rb') as fhd:
                loaded = pickle.load(fhd)
                self.dataset.extend(loaded)

    def update(self, moves):
        lprev = len(self.dataset)
        for move in moves:
            if move.ignore:
                move.forced_eval = 0

        self.dataset.extend(moves)
        if len(self.dataset) - lprev < len(moves):
            logging.debug("partial increase")
        elif len(self.dataset) - lprev == len(moves):
            logging.debug("full increase")
        else:
            logging.debug("no increase")


def makeCsv(data, path=os.getcwd()):
    cvpath = os.path.join(path, "neuralNetResults/dataset.csv")
    sfPath = os.path.join(path, "neuralNetResults/datasetSF.csv")
    with open(cvpath, 'w') as fp:
        pass
    with open(sfPath, 'w') as fp:
        pass

    df = pd.DataFrame.from_dict(NeuralNetworkData)
    SF = pd.DataFrame.from_dict(stockFisthRating)

    df.to_csv(cvpath, sep=";", na_rep="ERROR",
              encoding='utf-8', float_format='%.4f')

    SF.to_csv(sfPath, sep=";", na_rep="ERROR",
              encoding='utf-8', float_format='%.4f')


def setToFile(draw, param):
    lines = ["%s\n" % item for item in draw]
    lines.sort()
    with open(param, "w") as fhd:
        fhd.writelines(lines)


def playWithScore(pwhite, pblack):
    results = dataSet("results.pkl")
    results.loadMoves()

    if results.dataset:
        pass

    rnd = 0
    try:
        while True:
            if not ((rnd + 1) % 96) and len(results.dataset):
                pass

            if iteration(pblack, pwhite, results, rnd) != 0:
                pass
            makeCsv(NeuralNetworkData)

            rnd += 1
            if rnd > 1:
                break
    finally:
        results.dumpMoves()


def iteration(pblack, pwhite, results, rnd) -> int:
    result, moveNum = playGame(pwhite, pblack, rnd)
    wmoves = pwhite.getMoves(rnd)
    bmoves = pblack.getMoves(rnd)

    if result == '1-0':
        for x, move in enumerate(wmoves):
            move.eval = 1  # 0.5 + 0.5 * x / len(wmoves)
            move.from_round = rnd
        for x, move in enumerate(bmoves):
            move.eval = 0  # 0.5 - 0.5 * x / len(bmoves)
            move.from_round = rnd

        results.update(wmoves)

        NeuralNetworkData["Game Result"].append(1)

        return 1

    elif result == '0-1':
        for x, move in enumerate(wmoves):
            move.eval = 0.5 - 0.5 * x / len(wmoves)
            move.from_round = rnd
        for x, move in enumerate(bmoves):
            move.eval = 1  # 0.5 + 0.5 * x / len(bmoves)
            move.from_round = rnd

        results.update(bmoves)

        NeuralNetworkData["Game Result"].append(-1)

        return -1

    else:
        for x, move in enumerate(wmoves):
            move.eval = 0.5 - 0.25 * x / len(wmoves)
            move.from_round = rnd
        for x, move in enumerate(bmoves):
            move.eval = 0.5 - 0.25 * x / len(bmoves)
            move.from_round = rnd

        NeuralNetworkData["Game Result"].append(0)

        return 0


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    _LOG_FORMAT = '[%(relativeCreated)d %(name)s %(levelname)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG if is_debug()
                        else logging.INFO, format=_LOG_FORMAT)

    # trainingSetFP = "trainingData/chess_games.pgn"

    nn = chessNetwork(os.path.join(os.path.dirname(__file__), "models"))
    nn.save()
    white = neuralPLayer("Peter", WHITE, nn)
    black = neuralPLayer("Kasper", BLACK, nn)

    # train = useTrainingSet(black, white, trainingSetFP)
    # train.dataToTraining()

    # print("TrainingDone")

    try:
        playWithScore(white, black)

    finally:
        if isinstance(black, StockFish):
            black.engine.quit()

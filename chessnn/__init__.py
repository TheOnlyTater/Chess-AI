from matplotlib import pyplot
import numpy as np

from chess.syzygy import open_tablebase, Tablebase
from chess import pgn, SquareSet, SQUARES, Outcome
import chess

from typing import List, Optional
from collections import Counter
import sys
import os.path
import logging
import json
import copy

import collections


mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# Values for caputring each piece
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 4,
    chess.ROOK: 6,
    chess.QUEEN: 10,
    chess.KING: 100,
}


class stringExporter(pgn.StringExporter):
    comm_stack: list

    def __init__(self, comments: list):
        super().__init__(headers=True, variations=True, comments=True)
        self.comm_stack = copy.copy(comments)

    def visit_move(self, board, move):
        if self.variations or not self.variation_depth:
            # Write the move number.
            if board.turn == chess.WHITE:
                self.write_token(str(board.fullmove_number) + ". ")
            elif self.force_movenumber:
                self.write_token(str(board.fullmove_number) + "... ")

            # Write the SAN.
            if self.comm_stack:
                log_rec = self.comm_stack.pop(0)
                if log_rec.ignore:
                    comm = "ign"
                else:
                    pass
                    comm = "%.2f" % (log_rec.get_eval())

                self.write_token(board.san(move) + " {%s} " % comm)
            else:
                self.write_token(board.san(move))

            self.force_movenumber = False

# Creates a board that keeps track of all prior and current moves avalible
class myBoard(chess.Board):
    move_stack: List[chess.Move]

    def __init__(self, fen=chess.STARTING_FEN, *, chess960=False):
        super().__init__(fen, chess960=chess960)
        self.forced_result = None
        self.illegal_moves = []
        self._fens = []
        self.comment_stack = []
        self.initial_fen = chess.STARTING_FEN

    def outcome(self, *, claim_draw: bool = False) -> Optional[Outcome]:
        if self.forced_result:
            return self.forced_result
        return super().outcome(claim_draw=claim_draw)
    
    # Sets the board to the inital fen
    def set_chess960_pos(self, sharnagl):
        super().set_chess960_pos(sharnagl)
        self.initial_fen = self.fen()
    
    # Writes a PGN to document a game
    def writePGN(self, wp, bp, fname, roundd):
        journal = pgn.Game.from_board(self)
        journal.headers.clear()
        if self.chess960:
            journal.headers["Variant"] = "Chess960"
        journal.headers["FEN"] = self.initial_fen
        journal.headers["White"] = wp.name
        journal.headers["Black"] = bp.name
        journal.headers["Round"] = roundd
        journal.headers["Result"] = self.result(claim_draw=True)
        journal.headers["Site"] = self.explainGameEnd()
        exporter = stringExporter(self.comment_stack)
        pgns = journal.accept(exporter)
        with open(fname, "w") as out:
            out.write(pgns)
            
    # Finds the reason for Chess game ending for the PGN exporter
    def explainGameEnd(self):
        if self.forced_result:
            comm = "SyzygyDB"
        elif self.is_checkmate():
            comm = "checkmate"
        elif self.can_claim_fifty_moves():
            comm = "50 moves"
        elif self.can_claim_threefold_repetition():
            comm = "threefold"
        elif self.is_insufficient_material():
            comm = "material"
        elif not any(self.generate_legal_moves()):
            comm = "stalemate"
        else:
            comm = "by other reason"
        return comm

    def can_claim_threefold_repetition1(self):
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 3

    
    def can_claim_threefold_repetition2(self):
        transposition_key = self._transposition_key()
        transpositions = collections.Counter()
        transpositions.update((transposition_key,))

        # Count positions.
        switchyard = []
        # noinspection PyUnresolvedReferences
        while self.move_stack:
            move = self.pop()
            switchyard.append(move)

            if self.is_irreversible(move):
                break

            transpositions.update((self._transposition_key(),))

        while switchyard:
            self.push(switchyard.pop())

        # Threefold repetition occured.
        if transpositions[transposition_key] >= 3:
            return True

        return False

    # Checks if the same move has been repeted 5 times in a row
    def is_fivefold_repetition1(self):
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 5
    
    # Claims a draw if there has been more then 100 moves or there could be draw
    def can_claim_draw1(self):
        return super().can_claim_draw() or self.fullmove_number > 100

    def push(self, move):
        super().push(move)
        self._fens.append(self.epd().replace(
            " w ", " . ").replace(" b ", " . "))

    def pop(self):
        self._fens.pop(-1)
        return super().pop()

    def getPosition(self):
        pos = np.full((8, 8, len(chess.PIECE_TYPES) * 2), 0)
        for square in chess.SQUARES:
            piece = self.piece_at(square)

            if not piece:
                continue

            int(piece.color)
            channel = piece.piece_type - 1
            if piece.color:
                channel += len(PIECE_VALUES)
            pos[chess.square_file(square)][chess.square_rank(
                square)][channel] = 1

        pos.flags.writeable = False
        return pos
    
    # Creates a matrice of every piece attacked and defended for the NN
    # Saves this infomation in MoveRecord
    def getAttackedDefended(self):
        attacked = np.full(64, 0.0)
        defended = np.full(64, 0.0)

        our = self.occupied_co[self.turn]
        their = self.occupied_co[not self.turn]

        for square in SquareSet(our):
            for our_defended in SquareSet(self.attacks_mask(square)):
                defended[our_defended] = 1.0

        for square in SquareSet(their):
            for our_attacked in SquareSet(self.attacks_mask(square)):
                attacked[our_attacked] = 1.0

        return attacked, defended
    
    # Creates a matrice of all the current moves on the board indicated by a 1
    def getPossibleMoves(self):
        res = np.full(len(MOVES_MAP), 0.0)
        for move in self.generate_legal_moves():
            res[MOVES_MAP.index((move.from_square, move.to_square))] = 1.0
        return res


class moveRecord(object):
    piece: chess.Piece

    def __init__(self, position, move: chess.Move, piece, move_number, fifty_progress) -> None:
        super().__init__()
        self.hash = None
        self.possible = None
        self.full_move = move_number
        self.fifty_progress = fifty_progress
        self.eval = None
        self.ignore = False

        self.position = position
        self.piece = piece

        self.from_round = 0

        self.to_square = move.to_square
        self.from_square = move.from_square

        self.attacked = None
        self.defended = None

    def __hash__(self) -> int:
        if self.hash is None:
            self.hash = hash(asTuple(self.position.tolist()))
        return self.hash

    def __str__(self) -> str:
        return json.dumps({x: y for x, y in self.__dict__.items() if x not in ('forced_eval', 'kpis')})

    def get_eval(self):
        if self.eval is not None:
            return self.eval

        return 0.0

    def getMoveNum(self):
        if self.from_square == self.to_square:
            return -1  # null move

        return MOVES_MAP.index((self.from_square, self.to_square))

    def getMove(self):
        return chess.Move(self.from_square, self.to_square)


def asTuple(x):
    if isinstance(x, list):
        return tuple(asTuple(y) for y in x)
    else:
        return x


def is_debug():
    return 'pydevd' in sys.modules or os.getenv("DEBUG")

# Creates a sorted set of all the current possible moves
# uses set to avoid duplicate moves
def possibleMoves():
    res = set()
    for f in SQUARES:
        for t in chess.SquareSet(chess.BB_RANK_ATTACKS[f][0]):
            res.add((f, t))

        for t in chess.SquareSet(chess.BB_FILE_ATTACKS[f][0]):
            res.add((f, t))

        for t in chess.SquareSet(chess.BB_DIAG_ATTACKS[f][0]):
            res.add((f, t))

        for t in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[f]):
            res.add((f, t))

    assert (10, 26) in res

    return list(sorted(res))


MOVES_MAP = possibleMoves()

try:
    SYZYGY = open_tablebase(os.path.join(os.path.dirname(
        __file__), "..", "syzygy", "3-4-5"), load_dtz=False)
except BaseException:
    SYZYGY = Tablebase()

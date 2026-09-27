"""Score decision graphs by playing chess, one graph execution per move.

A graph cannot name a move: there are thousands and which are legal changes every ply. It names
a policy instead -- take the best capture, get the king castled, save the piece that is hanging --
and the board turns that into a move, deterministically. The labels are the policies, so the same
rule language that presses buttons in Doom plays chess here.
"""

import random
from typing import Literal, Self

import chess
from pydantic import Field, model_validator

from decision_compose.graph import Plan, StrictModel, Text
from decision_compose.optimize import Outcome
from decision_compose.runtime import DecisionProgram, GraphExecutionError

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
# Material left on both sides, below which the game is played as an endgame.
ENDGAME = 24
# The edge that decides a game stopped at the move limit: a minor piece, not a pawn.
MARGIN = 3


def worth(piece: chess.Piece | None) -> int:
    return VALUES.get(piece.piece_type, 0) if piece else 0


def gain(board: chess.Board, move: chess.Move) -> int:
    """What a move wins in material, less the mover if the square can be taken back. An
    approximation of an exchange search: one recapture deep, which is what the labels need."""
    taken = worth(board.piece_at(move.to_square))
    if board.is_en_passant(move):
        taken = VALUES[chess.PAWN]
    mover = worth(board.piece_at(move.from_square))
    board.push(move)
    exposed = board.is_attacked_by(board.turn, move.to_square)
    board.pop()
    return taken - (mover if exposed else 0)


def centrality(square: int) -> int:
    """How near the middle a square is; ties between developing moves break on this."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    return -(abs(2 * file - 7) + abs(2 * rank - 7))


def hanging(board: chess.Board) -> list[tuple[int, int]]:
    """Own pieces the opponent attacks and this side does not defend, worst first."""
    loose = []
    for square, piece in board.piece_map().items():
        if piece.color != board.turn or piece.piece_type == chess.KING:
            continue
        if board.is_attacked_by(not board.turn, square) and not board.is_attacked_by(
            board.turn, square
        ):
            loose.append((worth(piece), square))
    return sorted(loose, reverse=True)


def quiet(board: chess.Board, move: chess.Move) -> bool:
    return not board.is_capture(move) and not board.gives_check(move)


def take(board: chess.Board) -> chess.Move | None:
    captures = [move for move in board.legal_moves if board.is_capture(move)]
    return max(captures, key=lambda move: (gain(board, move), move.uci()), default=None)


def mates(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    over = board.is_checkmate()
    board.pop()
    return over


def check(board: chess.Board) -> chess.Move | None:
    """A checking move, and if one of them ends the game, that one: a policy that says 'give
    check' and passes over mate in one is not doing what it says."""
    checks = [move for move in board.legal_moves if board.gives_check(move)]
    return max(checks, key=lambda move: (mates(board, move), gain(board, move), move.uci()), default=None)


def castle(board: chess.Board) -> chess.Move | None:
    castles = [move for move in board.legal_moves if board.is_castling(move)]
    return min(castles, key=lambda move: move.uci(), default=None)


def develop(board: chess.Board) -> chess.Move | None:
    """A knight or bishop off its own back rank, towards the middle."""
    home = 0 if board.turn == chess.WHITE else 7
    moves = [
        move
        for move in board.legal_moves
        if (piece := board.piece_at(move.from_square))
        and piece.piece_type in (chess.KNIGHT, chess.BISHOP)
        and chess.square_rank(move.from_square) == home
        and gain(board, move) >= 0
    ]
    return max(moves, key=lambda move: (centrality(move.to_square), move.uci()), default=None)


def advance(board: chess.Board) -> chess.Move | None:
    """A pawn forward: into the middle early, up the board once the pieces have gone."""
    pawns = [
        move
        for move in board.legal_moves
        if (piece := board.piece_at(move.from_square))
        and piece.piece_type == chess.PAWN
        and gain(board, move) >= 0
    ]
    ahead = 1 if board.turn == chess.WHITE else -1
    return max(
        pawns,
        key=lambda move: (
            move.to_square in CENTER,
            ahead * (chess.square_rank(move.to_square) - chess.square_rank(move.from_square)),
            centrality(move.to_square),
            move.uci(),
        ),
        default=None,
    )


def save(board: chess.Board) -> chess.Move | None:
    """Move the most valuable hanging piece somewhere it is not attacked."""
    for _, square in hanging(board):
        escapes = [
            move
            for move in board.legal_moves
            if move.from_square == square and gain(board, move) >= 0
        ]
        safe = [move for move in escapes if not exposes(board, move)]
        if safe or escapes:
            return max(
                safe or escapes, key=lambda move: (gain(board, move), centrality(move.to_square))
            )
    return None


def exposes(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    attacked = board.is_attacked_by(board.turn, move.to_square)
    board.pop()
    return attacked


def settle(board: chess.Board) -> chess.Move:
    """The move played when a policy has nothing to offer: a quiet move that keeps the piece it
    moves, else the one that loses least. It must not reach for a capture the policy did not
    ask for -- a fallback that grabs material makes every policy the same policy."""
    moves = sorted(board.legal_moves, key=lambda move: move.uci())
    calm = [move for move in moves if quiet(board, move) and not exposes(board, move)]
    if calm:
        return max(calm, key=lambda move: (centrality(move.to_square), move.uci()))
    return max(moves, key=lambda move: (gain(board, move), move.uci()))


def loose_for(board: chess.Board, color: bool) -> list[tuple[int, int]]:
    """Pieces of one color the other side attacks and that color does not defend, worst first."""
    found = []
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        if board.is_attacked_by(not color, square) and not board.is_attacked_by(color, square):
            found.append((worth(piece), square))
    return sorted(found, reverse=True)


def ring(board: chess.Board, color: bool) -> int:
    """How many squares around a king the other side attacks: pressure before it becomes check."""
    king = board.king(color)
    if king is None:
        return 0
    around = chess.SquareSet(chess.BB_KING_ATTACKS[king])
    return sum(1 for square in around if board.is_attacked_by(not color, square))


def passers(board: chess.Board, color: bool) -> list[int]:
    """Own pawns with no enemy pawn ahead of them on their file or the ones beside it, by how
    far up the board they are."""
    ahead = 1 if color == chess.WHITE else -1
    ranks = []
    for square in board.pieces(chess.PAWN, color):
        file, rank = chess.square_file(square), chess.square_rank(square)
        blocked = any(
            chess.square_file(other) in (file - 1, file, file + 1)
            and (chess.square_rank(other) - rank) * ahead > 0
            for other in board.pieces(chess.PAWN, not color)
        )
        if not blocked:
            ranks.append(rank if color == chess.WHITE else 7 - rank)
    return sorted(ranks, reverse=True)


def promote(board: chess.Board) -> chess.Move | None:
    """Push the passed pawn that is furthest up the board, promoting when it can."""
    ahead = 1 if board.turn == chess.WHITE else -1
    free = {square for square in board.pieces(chess.PAWN, board.turn)
            if (7 - chess.square_rank(square) if board.turn == chess.WHITE
                else chess.square_rank(square)) >= 0}
    pushes = [
        move
        for move in board.legal_moves
        if move.from_square in free
        and (piece := board.piece_at(move.from_square))
        and piece.piece_type == chess.PAWN
        and (chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) * ahead > 0
        and gain(board, move) >= 0
    ]
    if not pushes:
        return None
    reach = lambda move: chess.square_rank(move.to_square) if board.turn == chess.WHITE else 7 - chess.square_rank(move.to_square)
    return max(pushes, key=lambda move: (move.promotion or 0, reach(move), move.uci()))


POLICIES = {
    "CAPTURE": take,
    "CHECK": check,
    "CASTLE": castle,
    "DEVELOP": develop,
    "ADVANCE": advance,
    "SAVE": save,
    "PROMOTE": promote,
    "QUIET": lambda board: None,
}


def chosen(board: chess.Board, label: str) -> chess.Move:
    """The move a policy names in this position; every policy always yields a legal move."""
    if label not in POLICIES:
        raise ValueError(f"unknown policy {label!r}; the task allows {sorted(POLICIES)}")
    return POLICIES[label](board) or settle(board)


def observe(board: chess.Board) -> dict:
    """What the side to move can see, as numbers a read node can take by path."""
    mine = sum(worth(piece) for piece in board.piece_map().values() if piece.color == board.turn)
    theirs = sum(
        worth(piece) for piece in board.piece_map().values() if piece.color != board.turn
    )
    loose = hanging(board)
    theirs_loose = loose_for(board, not board.turn)
    mine_passed = passers(board, board.turn)
    best = take(board)
    checking = check(board)
    back = 0 if board.turn == chess.WHITE else 7
    return {
        "ply": board.ply(),
        "phase": "endgame" if mine + theirs <= ENDGAME else "opening" if board.ply() < 20 else "middlegame",
        "material": {"mine": mine, "theirs": theirs, "balance": mine - theirs},
        "king": {
            "in_check": int(board.is_check()),
            "attackers": len(board.attackers(not board.turn, board.king(board.turn))),
            "can_castle": int(bool(castle(board))),
            "castled": int(board.king(board.turn) not in (chess.E1, chess.E8)),
        },
        "capture": {
            "best_gain": gain(board, best) if best else 0,
            "count": sum(1 for move in board.legal_moves if board.is_capture(move)),
        },
        "check": {
            "available": int(bool(checking)),
            "gain": gain(board, checking) if checking else 0,
        },
        "hanging": {"count": len(loose), "worst": loose[0][0] if loose else 0},
        # What is loose on the other side is a reason to go looking, not only a capture to take.
        "target": {
            "loose_count": len(theirs_loose),
            "loose_worst": theirs_loose[0][0] if theirs_loose else 0,
            "king_ring": ring(board, not board.turn),
        },
        "pawns": {
            "passed": len(mine_passed),
            "furthest": mine_passed[0] if mine_passed else 0,
            "theirs_passed": len(passers(board, not board.turn)),
        },
        "safety": {"king_ring": ring(board, board.turn)},
        "development": {
            "minor_home": sum(
                1
                for square, piece in board.piece_map().items()
                if piece.color == board.turn
                and piece.piece_type in (chess.KNIGHT, chess.BISHOP)
                and chess.square_rank(square) == back
            )
        },
        "moves": {"count": board.legal_moves.count()},
    }


class ChessTask(StrictModel):
    game: Literal["chess"] = "chess"
    description: Text
    labels: list[Text] = Field(min_length=2, description="Policies the graph may name")
    opening_plies: int = Field(default=4, ge=0, description="Random legal moves opening a game")
    move_limit: int = Field(default=120, ge=2, description="Plies before the game is adjudicated")
    material_weight: float = Field(
        default=0.2,
        ge=0,
        le=0.5,
        description="Share of the score decided by material rather than by the result, so that "
        "a game nobody wins still says which graph stood better",
    )
    train: list[int] = Field(min_length=1)
    validation: list[int] = Field(min_length=1)
    test: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be distinct")
        unknown = set(self.labels) - set(POLICIES)
        if unknown:
            raise ValueError(f"unknown policies: {sorted(unknown)}; allowed {sorted(POLICIES)}")
        seeds = self.train + self.validation + self.test
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be distinct across splits")
        return self

    def baseline_plan(self) -> Plan:
        return Plan.constant(self.labels[0])

    def opening(self, seed: int) -> chess.Board:
        """A board a few random legal moves in, so that one seed is one game."""
        board = chess.Board()
        rng = random.Random(seed)
        for _ in range(self.opening_plies):
            moves = sorted(board.legal_moves, key=lambda move: move.uci())
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        return board


def played(task: ChessTask, white: DecisionProgram, black: DecisionProgram, seed: int) -> dict:
    """One game. Returns the result from white's side, with what each graph did along the way."""
    board = task.opening(seed)
    sides = {chess.WHITE: white, chess.BLACK: black}
    moves, models, error, loser = [], set(), None, None
    while not board.is_game_over() and board.ply() < task.move_limit:
        program = sides[board.turn]
        try:
            prediction = program(state=observe(board))
        except GraphExecutionError as failure:
            models.update(failure.prediction.models)
            error, loser = f"Graph execution failed: {failure}", board.turn
            break
        models.update(prediction.models)
        try:
            move = chosen(board, prediction.label)
        except ValueError as failure:
            error, loser = str(failure), board.turn
            break
        moves.append({"policy": prediction.label, "move": move.uci()})
        board.push(move)
    edge = material(board, chess.WHITE)
    if board.is_game_over():
        outcome = board.result(claim_draw=True)
    else:
        # Two graphs that cannot mate will shuffle to the move limit, and calling every one of
        # those a draw leaves search nothing to climb. A game stopped a piece down was lost.
        outcome = "1-0" if edge >= MARGIN else "0-1" if edge <= -MARGIN else "1/2-1/2"
    if loser is not None:
        outcome = "0-1" if loser == chess.WHITE else "1-0"
    return {
        "result": outcome,
        "moves": moves,
        "fen": board.fen(),
        "material": edge,
        "adjudicated": not board.is_game_over() and loser is None,
        "models": sorted(models),
        "error": error,
    }


def material(board: chess.Board, color: bool) -> int:
    ours = sum(worth(p) for p in board.piece_map().values() if p.color == color)
    theirs = sum(worth(p) for p in board.piece_map().values() if p.color != color)
    return ours - theirs


class Games:
    """Scores a graph by how it fares against a fixed opponent, one game per seed."""

    def __init__(self, task: ChessTask, opponent: DecisionProgram):
        self.task = task
        self.opponent = opponent

    def __call__(self, program: DecisionProgram, seed: int) -> Outcome:
        # Alternate colors by seed so a graph is scored from both sides of the same openings.
        mine = chess.WHITE if seed % 2 == 0 else chess.BLACK
        white, black = (program, self.opponent) if mine == chess.WHITE else (self.opponent, program)
        game = played(self.task, white, black, seed)
        points = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5, "*": 0.5}[game["result"]]
        if mine == chess.BLACK:
            points = 1 - points
        edge = game["material"] if mine == chess.WHITE else -game["material"]
        # Material only moves the score within the result, so winning always outranks holding on.
        share = self.task.material_weight
        score = (1 - share) * points + share * (0.5 + max(-15, min(15, edge)) / 30)
        said = "won" if points == 1 else "lost" if points == 0 else "drew"
        return Outcome(
            score=score if game["error"] is None else 0.0,
            record={
                "seed": seed,
                "color": "white" if mine == chess.WHITE else "black",
                "result": game["result"],
                "points": points,
                "material": edge,
                "plies": len(game["moves"]),
                "policies": {
                    policy: sum(1 for m in game["moves"] if m["policy"] == policy)
                    for policy in sorted({m["policy"] for m in game["moves"]})
                },
                "score": score,
                "error": game["error"],
            },
            trace={
                "Inputs": observe(self.task.opening(seed)),
                "Generated Outputs": {"moves": game["moves"][-8:]},
                "Feedback": game["error"]
                or (
                    f"Playing {'white' if mine == chess.WHITE else 'black'} from seed {seed} you "
                    f"{said} in {len(game['moves'])} plies, ending {edge:+d} in material "
                    f"(score {score:.3f}). Higher is better."
                ),
            },
            models=game["models"],
            error=game["error"],
        )

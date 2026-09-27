"""Chess as a task: a graph names a policy and the board turns it into a move."""

import chess
import pytest

from decision_compose.chess import POLICIES, ChessTask, Games, chosen, material, observe, played
from decision_compose.rules import compile_rules
from decision_compose.runtime import MISSING, DecisionProgram, resolve_node

POLICY_NAMES = sorted(POLICIES)
GREEDY = """read capture.best_gain in [-9, 9]
read hanging.worst in [0, 9]
read king.can_castle in [0, 1]
when capture.best_gain >= 1: CAPTURE
when hanging.worst >= 3: SAVE
when king.can_castle >= 1: CASTLE
always: DEVELOP"""


def task_data(**changes):
    return {
        "game": "chess",
        "description": "Win the game. The label is the policy that picks your next move.",
        "labels": POLICY_NAMES,
        "train": [0, 1],
        "validation": [2],
        "test": [3],
    } | changes


@pytest.fixture
def task():
    return ChessTask.model_validate(task_data())


def program(text):
    return DecisionProgram(compile_rules(text), None, "none")


@pytest.mark.parametrize("label", POLICY_NAMES)
@pytest.mark.parametrize("seed", [0, 5, 11])
def test_every_policy_names_a_legal_move_wherever_it_is_asked(task, label, seed):
    # A policy that cannot find its kind of move still has to move: the graph's label is the
    # whole decision, so there is nothing else to fall back on.
    board = task.opening(seed)
    while not board.is_game_over() and board.ply() < 24:
        move = chosen(board, label)
        assert move in board.legal_moves
        board.push(move)


def test_capture_takes_the_piece_that_is_free():
    # Black queen on d5 is defended by nothing; the pawn on e4 can take it.
    board = chess.Board("rnb1kbnr/ppp1pppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    assert chosen(board, "CAPTURE").uci() == "e4d5"


def test_capture_prefers_the_exchange_it_wins():
    # The bishop on c6 can take a pawn on d5 or the rook on a8. Both are undefended, so the
    # rook is worth four more.
    board = chess.Board("r3k3/8/2B5/3p4/8/8/8/4K3 w - - 0 1")
    assert chosen(board, "CAPTURE").uci() == "c6a8"


def test_save_moves_the_hanging_piece_out_of_reach():
    # White queen on h5 is attacked by the knight on f6 and defended by nothing.
    board = chess.Board("rnbqkb1r/pppppppp/5n2/7Q/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    move = chosen(board, "SAVE")
    assert move.from_square == chess.H5
    board.push(move)
    assert not board.is_attacked_by(chess.BLACK, move.to_square)


def test_the_example_the_proposer_sees_is_readable_by_a_read_node(task):
    # Search writes read paths by copying them out of the feedback example, so every path in it
    # has to resolve. The same defect cost a whole night of Doom searching.
    shown = Games(task, program("always: QUIET"))(program(GREEDY), 0).trace["Inputs"]
    for path in (["capture", "best_gain"], ["king", "can_castle"], ["material", "balance"]):
        node = compile_rules(
            f"read {'.'.join(path)} in [-99, 99]\nwhen {'.'.join(path)} >= 0: CAPTURE\nalways: QUIET"
        ).nodes
        read = next(n for n in node.values() if n.type == "read")
        assert resolve_node(read, shown, {}, {}) is not MISSING


def test_one_game_has_one_result_and_both_chairs_read_it_the_same(task):
    greedy, passive = program(GREEDY), program("always: QUIET")
    game = played(task, greedy, passive, 0)
    assert game["result"] in ("1-0", "0-1", "1/2-1/2", "*") and game["error"] is None
    # Whatever white ends up ahead by, black is behind by exactly that.
    board = chess.Board(game["fen"])
    assert material(board, chess.WHITE) == -material(board, chess.BLACK)


def test_a_graph_is_scored_from_both_chairs_across_seeds(task):
    # One seed is one opening; alternating color by seed keeps a graph from only ever
    # being measured as white.
    scorer = Games(task, program("always: QUIET"))
    assert scorer(program(GREEDY), 0).record["color"] == "white"
    assert scorer(program(GREEDY), 1).record["color"] == "black"


def test_taking_material_beats_never_taking_it(task):
    # The measurement Doom could not make: over a handful of games, is the better policy scored
    # higher? Neither of these can force mate, so the games end drawn however far ahead one side
    # is -- which is why the score is not the result alone. The board is what separates them.
    greedy, passive = program(GREEDY), program("always: QUIET")
    seeds = list(range(12))
    taking = [Games(task, passive)(greedy, seed).record for seed in seeds]
    leaving = [Games(task, greedy)(passive, seed).record for seed in seeds]
    assert sum(game["material"] for game in taking) > 0
    assert sum(game["score"] for game in taking) > sum(game["score"] for game in leaving)


def test_a_label_the_task_forbids_is_a_graph_error(task):
    with pytest.raises(ValueError, match="unknown policy"):
        chosen(chess.Board(), "CASTLE_QUEENSIDE")


def test_seeds_must_not_repeat_across_splits():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChessTask.model_validate(task_data(validation=[0]))


def test_promote_pushes_the_pawn_that_is_furthest_home():
    # A pawn on the seventh queens; the policy has to prefer it over the one on the third.
    board = chess.Board("8/1P6/8/8/8/2P3k1/8/6K1 w - - 0 1")
    move = chosen(board, "PROMOTE")
    assert move.uci() == "b7b8q" and move.promotion == chess.QUEEN


def test_a_graph_can_see_what_is_loose_on_the_other_side():
    # The black queen on d5 is attacked by the pawn on e4 and defended by nothing. Before this
    # reading a graph could only see its own hanging pieces and its own best capture.
    board = chess.Board("rnb1kbnr/ppp1pppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    seen = observe(board)
    assert seen["target"]["loose_count"] >= 1
    assert seen["target"]["loose_worst"] == 9


def test_passed_pawns_are_counted_for_the_side_to_move():
    # White's b-pawn has no black pawn on a, b or c ahead of it; black's g-pawn is not passed
    # because White's h-pawn stands beside its path.
    board = chess.Board("6k1/6p1/8/8/8/1P5P/8/6K1 w - - 0 1")
    assert observe(board)["pawns"]["passed"] == 1
    board.push_san("b4")
    assert observe(board)["pawns"]["passed"] == 0


def test_every_new_reading_is_a_number_a_read_node_can_take(task):
    shown = Games(task, program("always: QUIET"))(program(GREEDY), 0).trace["Inputs"]
    for group in ("target", "pawns", "safety"):
        assert group in shown
        for name, value in shown[group].items():
            assert isinstance(value, (int, float)), f"{group}.{name} is {type(value)}"

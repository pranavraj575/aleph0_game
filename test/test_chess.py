import pytest
import torch

from src.games import Chess2d

en_passant_tests = (
    [
        [
            [1, i],
            [3, i],  # I4
            [6, i],
            [5, i],  # I6
            [3, i],
            [4, i],  # I5
            [6, i + 1],
            [4, i + 1],  # (I+1)5
            [4, i],
            [5, i + 1],  # x(I+1)6
        ]
        for i in range(7)
    ]
    + [
        [
            [1, i],
            [3, i],  # I4
            [6, i],
            [5, i],  # I6
            [3, i],
            [4, i],  # I5
            [6, i - 1],
            [4, i - 1],  # (I-1)5
            [4, i],
            [5, i - 1],  # x(I-1)6
        ]
        for i in range(1, 8)
    ]
    + [
        [
            [0, 1],
            [2, 0],  # NA3
            [6, i],
            [4, i],  # I5
            [2, 0],
            [0, 1],  # NB1
            [4, i],
            [3, i],  # I4
            [1, i + 1],
            [3, i + 1],  # (I+1)4
            [3, i],
            [2, i + 1],  # x(I+1)5
        ]
        for i in range(7)
    ]
    + [
        [
            [0, 1],
            [2, 0],  # NA3
            [6, i],
            [4, i],  # I5
            [2, 0],
            [0, 1],  # NB1
            [4, i],
            [3, i],  # I4
            [1, i - 1],
            [3, i - 1],  # (I-1)4
            [3, i],
            [2, i - 1],  # x(I-1)5
        ]
        for i in range(1, 8)
    ]
)


def apply_actions(game, state, actions, render=False):
    if render:
        c = game.get_canvas()
        game.render(c, state)
    else:
        c = None
    reward, terminal = torch.zeros(game.num_agents()), False
    for action in actions:
        assert game.is_valid(state, action)
        state, reward, terminal, _ = game.step_weak_type(state, action)
        if render:
            game.render(c, state)
    return state, reward, terminal


@pytest.mark.parametrize("state", [Chess2d().init_state()])
@pytest.mark.parametrize("actions", en_passant_tests)
def test_apply_actions_chess2d(state, actions):
    game = Chess2d()
    apply_actions(game, state, actions)


def test_en_passant():
    game = Chess2d()
    s = game.init_state()
    actions = [
        [1, 0],
        [3, 0],  # A4
        [6, 0],
        [5, 0],  # A6
        [3, 0],
        [4, 0],  # A5
        [6, 1],
        [4, 1],  # B5
        [4, 0],
        [5, 1],  # xB6
    ]
    s, _, _ = apply_actions(game, s, actions)

    # piece was actually captured
    assert not game.is_valid(s, [4, 1])
    assert s.board[-1, 0, 4, 1] == 0


def test_castling():
    render = False
    game = Chess2d()
    s = game.init_state()
    actions = [
        [1, 4],
        [3, 4],  # E4
        [6, 4],
        [4, 4],  # E5
        [0, 6],
        [2, 5],  # NF3
        [7, 1],
        [5, 2],  # NC6
        [0, 5],
        [4, 1],  # BB5
        [7, 3],
        [6, 4],  # QE7
        [0, 4],
        [0, 6],  # O-O
        [6, 1],
        [5, 1],  # B6
        [0, 5],
        [0, 4],  # RE1
        [7, 2],
        [5, 0],  # BA3
        [4, 1],
        [3, 2],  # BC4
        [7, 4],
        [7, 2],  # O-O-O
    ]
    apply_actions(game, s, actions, render=render)

    # same except instead of BC4, its BxA3 at the last action
    #  this leads to castling being illegal
    #  in our representation, the bishop can capture a 'ghost' on E8, and win the game
    loss_actions = actions[:-4] + [
        [4, 1],
        [5, 0],  # BXA3
        [7, 4],
        [7, 2],  # O-O-O (illegal because of bishop on A3)
        [5, 0],
        [7, 2],  # BE8 (captures king)
    ]

    _, rwd, term = apply_actions(game, s, loss_actions, render=render)
    assert term
    assert rwd[0] == 1 and rwd[1] == -1


def test_castling_OO_failure():
    render = False
    game = Chess2d()
    s = game.init_state()
    actions = [
        [1, 4],
        [3, 4],  # E4
        [6, 4],
        [4, 4],  # E5
        [0, 6],
        [2, 7],  # NH3
        [7, 3],
        [5, 5],  # QF6
        [0, 5],
        [4, 1],  # BB5
        [5, 5],
        [1, 5],  # QxF2
        [0, 4],
        [0, 6],  # O-O (illegal because of queen
        [1, 5],  # move queen
    ]
    temp_s, _, _ = apply_actions(game, s, actions, render=render)
    for capture_squares in [[0, 4], [0, 5], [0, 6]]:
        _, reward, term, _ = game.step_weak_type(temp_s, capture_squares)
        assert term
        assert reward[0] == -1 and reward[1] == 1


def test_castling_OOO_failure():
    # now same for O-O-O

    render = False
    game = Chess2d()
    s = game.init_state()

    actions = [
        [1, 4],
        [3, 4],  # E4
        [6, 4],
        [4, 4],  # E5
        [0, 1],
        [2, 0],  # NA3
        [7, 3],
        [6, 4],  # QE7
        [0, 3],
        [2, 5],  # QF3
        [6, 4],
        [5, 3],  # QD6
        [1, 1],
        [2, 1],  # B3
        [7, 1],
        [5, 2],  # NC6
        [0, 2],
        [1, 1],  # BB2
        [5, 3],
        [1, 3],  # QXD2
        [0, 4],
        [0, 2],  # O-O-O (illegal because of Q on D2)
        [1, 3],  # move queen
    ]
    temp_s, _, _ = apply_actions(game, s, actions, render=render)
    for capture_squares in [[0, 4], [0, 3], [0, 2]]:
        _, reward, term, _ = game.step_weak_type(temp_s, capture_squares)
        assert term
        assert reward[0] == -1 and reward[1] == 1

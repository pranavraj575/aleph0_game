import pytest
import torch
from test_games import sample_from_action_mask

from src.games import Chess2d, Chess5d

en_passant_tests = (
    [
        [
            # I4
            [1, i],
            [3, i],
            # I6
            [6, i],
            [5, i],
            # I5
            [3, i],
            [4, i],
            # (I+1)5
            [6, i + 1],
            [4, i + 1],
            # x(I+1)6
            [4, i],
            [5, i + 1],
        ]
        for i in range(7)
    ]
    + [
        [
            # I4
            [1, i],
            [3, i],
            # I6
            [6, i],
            [5, i],
            # I5
            [3, i],
            [4, i],
            # (I-1)5
            [6, i - 1],
            [4, i - 1],
            # x(I-1)6
            [4, i],
            [5, i - 1],
        ]
        for i in range(1, 8)
    ]
    + [
        [
            # NA3
            [0, 1],
            [2, 0],
            # I5
            [6, i],
            [4, i],
            # NB1
            [2, 0],
            [0, 1],
            # I4
            [4, i],
            [3, i],
            # (I+1)4
            [1, i + 1],
            [3, i + 1],
            # x(I+1)5
            [3, i],
            [2, i + 1],
        ]
        for i in range(7)
    ]
    + [
        [
            # NA3
            [0, 1],
            [2, 0],
            # I5
            [6, i],
            [4, i],
            # NB1
            [2, 0],
            [0, 1],
            # I4
            [4, i],
            [3, i],
            # (I-1)4
            [1, i - 1],
            [3, i - 1],
            # x(I-1)5
            [3, i],
            [2, i - 1],
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
        # A4
        [1, 0],
        [3, 0],
        # A6
        [6, 0],
        [5, 0],
        # A5
        [3, 0],
        [4, 0],
        # B5
        [6, 1],
        [4, 1],
        # xB6
        [4, 0],
        [5, 1],
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
        # E4
        [1, 4],
        [3, 4],
        # E5
        [6, 4],
        [4, 4],
        # NF3
        [0, 6],
        [2, 5],
        # NC6
        [7, 1],
        [5, 2],
        # BB5
        [0, 5],
        [4, 1],
        # QE7
        [7, 3],
        [6, 4],
        # O-O
        [0, 4],
        [0, 6],
        # B6
        [6, 1],
        [5, 1],
        # RE1
        [0, 5],
        [0, 4],
        # BA3
        [7, 2],
        [5, 0],
        # BC4
        [4, 1],
        [3, 2],
        # O-O-O
        [7, 4],
        [7, 2],
    ]
    apply_actions(game, s, actions, render=render)

    # same except instead of BC4, its BxA3 at the last action
    #  this leads to castling being illegal
    #  in our representation, the bishop can capture a 'ghost' on E8, and win the game
    loss_actions = actions[:-4] + [
        # BXA3
        [4, 1],
        [5, 0],
        # O-O-O (illegal because of bishop on A3)
        [7, 4],
        [7, 2],
        # BE8 (captures king)
        [5, 0],
        [7, 2],
    ]

    _, rwd, term = apply_actions(game, s, loss_actions, render=render)
    assert term
    assert rwd[0] == 1 and rwd[1] == -1


def test_castling_OO_failure():
    render = False
    game = Chess2d()
    s = game.init_state()
    actions = [
        # E4
        [1, 4],
        [3, 4],
        # E5
        [6, 4],
        [4, 4],
        # NH3
        [0, 6],
        [2, 7],
        # QF6
        [7, 3],
        [5, 5],
        # BB5
        [0, 5],
        [4, 1],
        # QxF2
        [5, 5],
        [1, 5],
        # O-O (illegal because of queen
        [0, 4],
        [0, 6],
        # move queen
        [1, 5],
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
        # E4
        [1, 4],
        [3, 4],
        # E5
        [6, 4],
        [4, 4],
        # NA3
        [0, 1],
        [2, 0],
        # QE7
        [7, 3],
        [6, 4],
        # QF3
        [0, 3],
        [2, 5],
        # QD6
        [6, 4],
        [5, 3],
        # B3
        [1, 1],
        [2, 1],
        # NC6
        [7, 1],
        [5, 2],
        # BB2
        [0, 2],
        [1, 1],
        # QXD2
        [5, 3],
        [1, 3],
        # O-O-O (illegal because of Q on D2)
        [0, 4],
        [0, 2],
        # move queen
        [1, 3],
    ]
    temp_s, _, _ = apply_actions(game, s, actions, render=render)
    for capture_squares in [[0, 4], [0, 3], [0, 2]]:
        _, reward, term, _ = game.step_weak_type(temp_s, capture_squares)
        assert term
        assert reward[0] == -1 and reward[1] == 1


@pytest.mark.parametrize("seed", list(range(3)))
def test_chess5d_playthrough_and_start_turn_method(seed, depth=250):
    game = Chess5d()
    torch.random.manual_seed(seed)
    s = game.init_state()
    terminal = False

    start_turn_board = s.board
    start_turn_center = s.center_timeline

    bd, center = game.start_turn_board_and_center(s)
    assert torch.equal(bd, start_turn_board)
    assert center == start_turn_center
    while depth >= 0 and not terminal:
        mask = game.action_mask(s)
        action = sample_from_action_mask(game, mask)
        game.agent_observe(s)
        game.critic_observe(s)
        assert game.is_valid(s, action)
        s_prime, _, terminal, _ = game.step(s, action)
        if game.player(s) != game.player(s_prime):
            # END_TURN was played
            start_turn_board = s_prime.board
            start_turn_center = s_prime.center_timeline

        bd, center = game.start_turn_board_and_center(s_prime)
        assert torch.equal(bd, start_turn_board)
        assert center == start_turn_center

        s = s_prime
        depth -= 1

    game.render(game.get_canvas(), s)

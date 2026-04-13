import itertools

import pytest
import torch
from test_games import sample_from_action_mask

from src.games import Chess2d, Chess5d
from src.games.chess5d import DAG_subgraphs_w_at_most_one_outgoing_edge, all_subsets

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


def test_check():
    render = True
    game = Chess2d()
    s = game.init_state()
    actions = [
        # E4
        [1, 4],
        [3, 4],
        # E5
        [6, 4],
        [4, 4],
        # QH5
        [0, 3],
        [4, 7],
        # D5
        [6, 3],
        [4, 3],
        # QxF3
        [4, 7],
        [6, 5],
    ]
    temp_s, _, _ = apply_actions(game, s, actions, render=render)
    assert game.player_in_check(temp_s)


@pytest.mark.parametrize("seed", list(range(3)))
def test_chess5d_undo_turn(seed, depth=250):
    game = Chess5d()
    torch.random.manual_seed(seed)
    s = game.init_state()
    terminal = False
    prev_start_turn_state = None
    start_turn_state = s
    temp_state = game.undo_player_turn(s, prev_turn=False)
    assert torch.equal(temp_state.board, start_turn_state.board)
    assert temp_state.center_timeline == start_turn_state.center_timeline
    assert torch.equal(temp_state.start_turn_board_mask, start_turn_state.start_turn_board_mask)
    while depth >= 0 and not terminal:
        mask = game.action_mask(s)
        action = sample_from_action_mask(game, mask)
        game.agent_observe(s)
        game.critic_observe(s)
        assert game.is_valid(s, action)
        s_prime, _, terminal, _ = game.step(s, action)
        if game.player(s) != game.player(s_prime):
            # END_TURN was played
            prev_start_turn_state = start_turn_state
            start_turn_state = s_prime

        temp_state = game.undo_player_turn(state=s_prime, prev_turn=False)
        assert torch.equal(temp_state.board, start_turn_state.board)
        assert temp_state.center_timeline == start_turn_state.center_timeline
        assert torch.equal(temp_state.start_turn_board_mask, start_turn_state.start_turn_board_mask)
        assert torch.equal(temp_state.prev_start_turn_board_mask, start_turn_state.prev_start_turn_board_mask)

        if prev_start_turn_state is not None:
            temp_state = game.undo_player_turn(state=s_prime, prev_turn=True)
            assert torch.equal(temp_state.board, prev_start_turn_state.board)
            assert temp_state.center_timeline == prev_start_turn_state.center_timeline
            assert torch.equal(temp_state.start_turn_board_mask, prev_start_turn_state.start_turn_board_mask)
        s = s_prime
        depth -= 1

    game.render(game.get_canvas(), s)


edge_lists = []
torch.random.manual_seed(0)
for n in range(3, 9):
    for _ in range(4 * (8 - n) + 3):
        el = dict()
        for i in range(n):
            el[i] = [
                j
                for j in range(n)
                if j != i
                and torch.rand(
                    1,
                )
                < 0.5
            ]
        edge_lists.append(el)


@pytest.mark.parametrize("edge_list", edge_lists)
def test_dag_subgraph(edge_list):
    all_graphs = [g for g, _, _ in DAG_subgraphs_w_at_most_one_outgoing_edge(edge_list=edge_list)]

    # assert it is a DAG (and in reeverse topological order, where each edge (a,b) is made before any (x,a))
    # also assert that each vertex is the source of at most one edge
    def is_sorted_DAG_with_property(graph):
        used_vertices = set()
        for a, b in graph:
            if a in used_vertices:
                return False
            used_vertices = used_vertices.union({a, b})
        return True

    for graph in all_graphs:
        assert is_sorted_DAG_with_property(graph)

    all_edges = []
    for s in edge_list:
        all_edges.extend([(s, sink) for sink in edge_list[s]])
    if len(all_edges) <= 10:
        for subset in all_subsets(all_edges):
            for graph in itertools.permutations(subset):
                if is_sorted_DAG_with_property(graph):
                    assert graph in all_graphs


@pytest.mark.parametrize("seed", list(range(3)))
@pytest.mark.parametrize(
    "depth",
    [
        5,
        10,
        15,
        25,
    ],
)
def test_all_turns(seed, depth):
    game = Chess5d()
    torch.random.manual_seed(seed)
    s = game.init_state()
    terminal = False
    while depth >= 0 and not terminal:
        mask = game.action_mask(s)
        action = sample_from_action_mask(game, mask)

        assert game.is_valid(s, action)
        s_prime, _, terminal, _ = game.step(s, action)

        s = s_prime
        depth -= 1
    if terminal:
        return
    # make sure game is at start of turn
    s = game.undo_player_turn(s)
    turns = list(game.get_all_possible_turns(state=s))
    og_len = len(turns)
    # game.render(game.get_canvas(),s)
    if len(turns) > 2000:
        turns = [turns[i] for i in torch.randperm(len(turns))[:1000]]
    print(f"testing {len(turns)} out of {og_len} generated turns")
    for i, turn in enumerate(turns):
        print(f"{i + 1}/{len(turns)}", end="\r")
        s_prime = s
        for source_idx, sink_idx in turn:
            source_idx[1] = source_idx[1] + s_prime.center_timeline
            sink_idx[1] = sink_idx[1] + s_prime.center_timeline
            action = (source_idx, torch.tensor(-1))
            # game.render(None,game.step(s_prime,action)[0])
            assert game.is_valid(s_prime, action)
            s_prime, _, _, _ = game.step(s_prime, action)
            action = (sink_idx, torch.tensor(-1))
            assert game.is_valid(s_prime, action)
            s_prime, _, _, _ = game.step(s_prime, action)
        assert game.is_valid(s_prime, (-torch.ones(4, dtype=torch.int), torch.tensor(0)))

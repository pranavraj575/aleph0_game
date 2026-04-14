import itertools
import os

import pytest
import torch
from test_games import sample_from_action_mask

from aleph0_game.games import Chess2d, Chess5d
from aleph0_game.games.chess5d import DAG_subgraphs_w_at_most_one_outgoing_edge, all_subsets

en_passant_tests = (
    [
        [
            # i4
            [1, i],
            [3, i],
            # i6
            [6, i],
            [5, i],
            # i5
            [3, i],
            [4, i],
            # (i+1)5
            [6, i + 1],
            [4, i + 1],
            # x(i+1)6
            [4, i],
            [5, i + 1],
        ]
        for i in range(7)
    ]
    + [
        [
            # i4
            [1, i],
            [3, i],
            # i6
            [6, i],
            [5, i],
            # i5
            [3, i],
            [4, i],
            # (i-1)5
            [6, i - 1],
            [4, i - 1],
            # x(i-1)6
            [4, i],
            [5, i - 1],
        ]
        for i in range(1, 8)
    ]
    + [
        [
            # Na3
            [0, 1],
            [2, 0],
            # i5
            [6, i],
            [4, i],
            # Nb1
            [2, 0],
            [0, 1],
            # i4
            [4, i],
            [3, i],
            # (i+1)4
            [1, i + 1],
            [3, i + 1],
            # x(i+1)5
            [3, i],
            [2, i + 1],
        ]
        for i in range(7)
    ]
    + [
        [
            # Na3
            [0, 1],
            [2, 0],
            # i5
            [6, i],
            [4, i],
            # Nb1
            [2, 0],
            [0, 1],
            # i4
            [4, i],
            [3, i],
            # (i-1)4
            [1, i - 1],
            [3, i - 1],
            # x(i-1)5
            [3, i],
            [2, i - 1],
        ]
        for i in range(1, 8)
    ]
)


def apply_actions(game, state, actions, render=False, assert_valid=True):
    if render:
        c = game.get_canvas()
        game.render(c, state)
    else:
        c = None
    reward, terminal = torch.zeros(game.num_agents()), False
    for action in actions:
        if assert_valid:
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
        # a4
        [1, 0],
        [3, 0],
        # a6
        [6, 0],
        [5, 0],
        # a5
        [3, 0],
        [4, 0],
        # b5
        [6, 1],
        [4, 1],
        # xb6
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
        # e4
        [1, 4],
        [3, 4],
        # e5
        [6, 4],
        [4, 4],
        # Nf3
        [0, 6],
        [2, 5],
        # Nc6
        [7, 1],
        [5, 2],
        # Bb5
        [0, 5],
        [4, 1],
        # Qe7
        [7, 3],
        [6, 4],
        # O-O
        [0, 4],
        [0, 6],
        # b6
        [6, 1],
        [5, 1],
        # Re1
        [0, 5],
        [0, 4],
        # Ba3
        [7, 2],
        [5, 0],
        # Bc4
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
        # BXa3
        [4, 1],
        [5, 0],
        # O-O-O (illegal because of bishop on A3)
        [7, 4],
        [7, 2],
        # Be8 (captures king)
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
        # e4
        [1, 4],
        [3, 4],
        # e5
        [6, 4],
        [4, 4],
        # Nh3
        [0, 6],
        [2, 7],
        # Qf6
        [7, 3],
        [5, 5],
        # Bb5
        [0, 5],
        [4, 1],
        # Qxf2
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
        # e4
        [1, 4],
        [3, 4],
        # e5
        [6, 4],
        [4, 4],
        # Na3
        [0, 1],
        [2, 0],
        # Qe7
        [7, 3],
        [6, 4],
        # Qf3
        [0, 3],
        [2, 5],
        # Qd6
        [6, 4],
        [5, 3],
        # b3
        [1, 1],
        [2, 1],
        # Nc6
        [7, 1],
        [5, 2],
        # Bb2
        [0, 2],
        [1, 1],
        # QXd2
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
    render = False
    game = Chess2d()
    s = game.init_state()
    actions = [
        # e4
        [1, 4],
        [3, 4],
        # e5
        [6, 4],
        [4, 4],
        # Qh5
        [0, 3],
        [4, 7],
        # d5
        [6, 3],
        [4, 3],
        # Qxf3
        [4, 7],
        [6, 5],
    ]
    temp_s, _, _ = apply_actions(game, s, actions, render=render)
    assert game.player_in_check(temp_s)


def assert_state_equality(s1, s2, check_timestep=False):
    assert torch.equal(s1.board, s2.board)
    assert s1.center_timeline == s2.center_timeline
    assert torch.equal(s1.board_spawn_timestep, s2.board_spawn_timestep)
    if check_timestep:
        assert (s1.timestep, s1.start_turn_timestep, s1.prev_start_turn_timestep) == (
            s2.timestep,
            s2.start_turn_timestep,
            s2.prev_start_turn_timestep,
        )


@pytest.mark.parametrize("seed", list(range(3)))
def test_chess5d_undo_turn(seed, depth=250):
    game = Chess5d()
    torch.random.manual_seed(seed)
    s = game.init_state()
    terminal = False
    prev_start_turn_state = None
    start_turn_state = s
    temp_state = game.undo_player_turn(s, prev_turn=False)
    assert_state_equality(temp_state, start_turn_state, check_timestep=True)
    while depth >= 0 and not terminal:
        mask = game.action_mask(s)
        action = sample_from_action_mask(game, mask)
        game.agent_observe(s)
        game.critic_observe(s)
        s_prime, _, terminal, _ = game.step(s, action)
        if game.player(s) != game.player(s_prime):
            # END_TURN was played
            prev_start_turn_state = start_turn_state
            start_turn_state = s_prime

        temp_state = game.undo_player_turn(state=s_prime, prev_turn=False)
        assert_state_equality(temp_state, start_turn_state, check_timestep=True)

        if prev_start_turn_state is not None:
            temp_state = game.undo_player_turn(state=s_prime, prev_turn=True)

            assert_state_equality(temp_state, prev_start_turn_state, check_timestep=False)

        s = s_prime
        depth -= 1


@pytest.mark.parametrize("seed", list(range(3, 6)))
def test_chess5d_lossless_history(seed, depth=250):
    game = Chess5d()
    torch.random.manual_seed(seed)
    s = game.init_state()
    start_turn_states = [s]
    terminal = False

    while depth >= 0 and not terminal:
        mask = game.action_mask(s)
        action = sample_from_action_mask(game, mask)
        game.agent_observe(s)
        game.critic_observe(s)
        s, _, terminal, _ = game.step(s, action)
        if s.player != start_turn_states[-1].player:
            start_turn_states.append(s)
        depth -= 1
    s = game.undo_player_turn(s, prev_turn=False, lossless=True)
    sp = start_turn_states.pop()
    assert_state_equality(s, sp, check_timestep=True)
    while start_turn_states:
        s = game.undo_player_turn(s, prev_turn=True, lossless=True)
        sp = start_turn_states.pop()
        assert_state_equality(s, sp, check_timestep=True)


edge_lists = []
torch.random.manual_seed(6)
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


@pytest.mark.parametrize("seed", list(range(7, 10)))
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


# downloaded games from https://www.chessgames.com/
def clean_pgm(fn):
    with open(fn, "r") as f:
        s = f.read()
        s = " ".join(t for t in s.split("\n") if not (t.startswith("[")))
        s = s.replace(".", ". ").replace("  ", " ").strip().split(" ")
    output = [m for m in s if "." not in m]
    if output[-1] in ("1-0", "0-1", "1/2-1/2"):
        # remove result
        output = output[:-1]
    return output


test_dir = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "algebraic_actions",
    [
        ["e4", "e5", "Na3"],
        ["e4", "d5", "c4", "b5", "exd5"],
        ["e4", "d5", "c4", "b5", "cxd5"],
        ["e4", "e5", "Qh5", "Na6", "Bc4", "Nb8", "Qxf7#"],
    ]
    + [clean_pgm(os.path.join(test_dir, "notation_tests", fn)) for fn in os.listdir(os.path.join(test_dir, "notation_tests"))],
)
def test_algebraic_notation(algebraic_actions):
    game = Chess2d()
    s = game.init_state()
    for action in algebraic_actions:
        pick, place = game.from_algebraic_notation(state=s, action=action)
        alg = game.to_algebraic_notation(state=s, pick_action=pick, place_action=place)
        assert action == alg
        s, _, _, _ = game.step_weak_type(state=s, action=action)

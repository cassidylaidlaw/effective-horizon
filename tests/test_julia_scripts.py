import json
import os
import random
import subprocess

import numpy as np
import pytest


def assert_structure_close(a, b, eps=1e-8):
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a.keys()) == set(b.keys())
        for k in a:
            assert_structure_close(a[k], b[k], eps=eps)
    elif isinstance(a, list):
        assert isinstance(b, list)
        assert len(a) == len(b)
        for a_i, b_i in zip(a, b):
            assert_structure_close(a_i, b_i, eps=eps)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
        assert a.shape == b.shape
        assert np.all(np.abs(a - b) < eps)
    else:
        assert (a == b) or abs(a - b) < eps


TEST_MDPS = [
    ("data/mdps/MiniGrid-KeyCorridorS3R1-v0/consolidated.npz", 100),
    ("data/mdps/freeway_10_fs30/consolidated.npz", 10),
    ("data/mdps/maze_easy_l0_30_fs1/consolidated.npz", 30),
    (
        "data/mdps_with_exploration_policy_sb3/freeway_10_fs30/consolidated_framestack.npz",
        10,
    ),
    ("data/mdps_with_exploration_policy_sb3/maze_easy_l0_30_fs1/consolidated.npz", 30),
]


@pytest.mark.uses_julia
def test_construct_atari_mdp(tmp_path):
    subprocess.check_call(
        [
            "julia",
            "--threads=2",
            "--project=EffectiveHorizon.jl",
            "EffectiveHorizon.jl/src/construct_mdp.jl",
            "--rom",
            "freeway",
            "-o",
            tmp_path,
            "--horizon",
            "10",
            "--frameskip",
            "30",
            "--done_on_life_lost",
            "--save_screens",
            "--save_states",
        ]
    )
    for mdp_fname, expected_num_states in [
        ("mdp.npz", 399),
        ("consolidated.npz", 197),
        ("consolidated_ignore_screen.npz", 33),
    ]:
        mdp = np.load(tmp_path / mdp_fname)
        assert mdp["transitions"].shape == (expected_num_states, 3)
        assert mdp["screen_mapping"].shape == (expected_num_states,)
        state_lengths = mdp["state_lengths"]
        assert state_lengths.shape == (expected_num_states,)
        assert mdp["states"].shape == (expected_num_states, state_lengths.max())


@pytest.mark.uses_julia
def test_construct_procgen_mdp(tmp_path):
    subprocess.check_call(
        [
            "julia",
            "--threads=2",
            "--project=EffectiveHorizon.jl",
            "EffectiveHorizon.jl/src/construct_mdp.jl",
            "--env_name",
            "maze",
            "--distribution_mode",
            "easy",
            "--level",
            "0",
            "-o",
            tmp_path,
            "--horizon",
            "30",
            "--frameskip",
            "1",
            "--save_screens",
            "--save_states",
        ]
    )
    for mdp_fname, expected_num_states in [
        ("mdp.npz", 2227),
        ("consolidated.npz", 243),
        ("consolidated_ignore_screen.npz", 134),
    ]:
        mdp = np.load(tmp_path / mdp_fname)
        assert mdp["transitions"].shape == (expected_num_states, 9)
        assert mdp["screen_mapping"].shape == (expected_num_states,)
        state_lengths = mdp["state_lengths"]
        assert state_lengths.shape == (expected_num_states,)
        assert mdp["states"].shape == (expected_num_states, state_lengths.max())


@pytest.mark.uses_julia
def test_construct_minigrid_mdp(tmp_path):
    subprocess.check_call(
        [
            "julia",
            "--project=EffectiveHorizon.jl",
            "EffectiveHorizon.jl/src/construct_mdp.jl",
            "--minigrid",
            "--env_name",
            "BRIDGE/MiniGrid-KeyCorridorS3R1-v0",
            "-o",
            tmp_path,
            "--horizon",
            "100000000",
            "--frameskip",
            "1",
            "--save_screens",
            "--save_states",
        ]
    )
    for mdp_fname, expected_num_states in [
        ("mdp.npz", 169),
        ("consolidated.npz", 168),
        ("consolidated_ignore_screen.npz", 168),
    ]:
        mdp = np.load(tmp_path / mdp_fname)
        assert mdp["transitions"].shape == (expected_num_states, 6)
        assert mdp["screen_mapping"].shape == (expected_num_states,)
        state_lengths = mdp["state_lengths"]
        assert state_lengths.shape == (expected_num_states,)
        assert mdp["states"].shape == (expected_num_states, state_lengths.max())


@pytest.mark.uses_julia
def test_analyze_mdp(tmp_path, randomize_julia_scripts):
    test_mdps = TEST_MDPS
    if randomize_julia_scripts:
        test_mdps = random.sample(test_mdps, 1)

    for mdp_fname, horizon in test_mdps:
        mdp_type = mdp_fname.split("/")[1]
        mdp_name = mdp_fname.split("/")[-2]
        out_dir = tmp_path / mdp_type / mdp_name
        os.makedirs(out_dir, exist_ok=True)
        analyze_command = [
            "julia",
            "--project=EffectiveHorizon.jl",
            "EffectiveHorizon.jl/src/analyze_mdp.jl",
            "--mdp",
            mdp_fname,
            "--horizon",
            str(horizon),
            "-o",
            out_dir / "consolidated_analyzed.json",
        ]
        if mdp_type == "mdps_with_exploration_policy_sb3":
            analyze_command.extend(
                [
                    "--exploration_policy",
                    os.path.dirname(mdp_fname) + "/exploration_policy.npy",
                ]
            )
        subprocess.check_call(analyze_command)
        with open(out_dir / "consolidated_analyzed.json") as results_file:
            results = json.load(results_file)
        with open(mdp_fname[:-4] + "_analyzed.json") as expected_results_file:
            expected_results = json.load(expected_results_file)
        assert_structure_close(results, expected_results)

        value_dists = np.load(out_dir / "consolidated_analyzed_value_dists_1.npy")
        expected_value_dists = np.load(mdp_fname[:-4] + "_analyzed_value_dists_1.npy")
        assert_structure_close(value_dists, expected_value_dists)


@pytest.mark.uses_julia
def test_analyze_sticky_actions(tmp_path, randomize_julia_scripts):
    test_mdps = TEST_MDPS
    if randomize_julia_scripts:
        test_mdps = random.sample(test_mdps, 1)

    for mdp_fname, horizon in test_mdps:
        mdp_type = mdp_fname.split("/")[1]
        if mdp_type == "mdps_with_exploration_policy_sb3":
            # We don't analyze exploration policies for sticky actions.
            continue

        mdp_name = mdp_fname.split("/")[-2]
        out_dir = tmp_path / mdp_type / mdp_name
        os.makedirs(out_dir, exist_ok=True)
        analyze_command = [
            "julia",
            "--project=EffectiveHorizon.jl",
            "EffectiveHorizon.jl/src/analyze_sticky_actions.jl",
            "--mdp",
            mdp_fname,
            "--horizon",
            str(horizon),
            "-o",
            out_dir / "consolidated_analyzed_sticky_0.25.json",
        ]
        subprocess.check_call(analyze_command)
        with open(out_dir / "consolidated_analyzed_sticky_0.25.json") as results_file:
            results = json.load(results_file)
        with open(
            mdp_fname[:-4] + "_analyzed_sticky_0.25.json"
        ) as expected_results_file:
            expected_results = json.load(expected_results_file)
        assert_structure_close(results, expected_results)


@pytest.mark.uses_julia
def test_compute_gorp_bounds(tmp_path, randomize_julia_scripts):
    test_mdps = TEST_MDPS
    if randomize_julia_scripts:
        test_mdps = random.sample(test_mdps, 1)

    for mdp_fname, horizon in test_mdps:
        # MiniGrid environments take forever because of the long horizon and
        # because almost all states are optimal, so we skip them.
        if "MiniGrid" in mdp_fname:
            continue

        mdp_type = mdp_fname.split("/")[1]
        mdp_name = mdp_fname.split("/")[-2]
        out_dir = tmp_path / mdp_type / mdp_name
        os.makedirs(out_dir, exist_ok=True)

        with open(mdp_fname[:-4] + "_gorp_bounds.json") as expected_results_file:
            expected_results = json.load(expected_results_file)
        max_k = len(expected_results["k_results"])

        gorp_bounds_command = [
            "julia",
            "--threads=2",
            "--project=EffectiveHorizon.jl",
            "EffectiveHorizon.jl/src/compute_gorp_bounds.jl",
            "--mdp",
            mdp_fname,
            "--use_value_dists",
            "--horizon",
            str(horizon),
            "--max_k",
            str(max_k),
            "-o",
            out_dir / "consolidated_gorp_bounds.json",
        ]
        if mdp_type == "mdps_with_exploration_policy_sb3":
            gorp_bounds_command.extend(
                [
                    "--exploration_policy",
                    os.path.dirname(mdp_fname) + "/exploration_policy.npy",
                ]
            )
        subprocess.check_call(gorp_bounds_command)

        with open(out_dir / "consolidated_gorp_bounds.json") as results_file:
            results = json.load(results_file)
        assert_structure_close(results, expected_results)


@pytest.mark.uses_julia
def test_run_gorp_and_ucb(tmp_path):
    for mdp_fname, horizon in TEST_MDPS:
        mdp_type = mdp_fname.split("/")[1]
        mdp_name = mdp_fname.split("/")[-2]
        out_dir = tmp_path / mdp_type / mdp_name
        os.makedirs(out_dir, exist_ok=True)

        value_dists = np.load(mdp_fname[:-4] + "_analyzed_value_dists_1.npy")
        initial_value_dist = value_dists[:, :, 0]
        optimal_return = initial_value_dist[0].max()

        alg = random.choice(["ucb", "gorp"])

        command = [
            "julia",
            "--threads=2",
            "--project=EffectiveHorizon.jl",
            f"EffectiveHorizon.jl/src/run_{alg}.jl",
            "--mdp",
            mdp_fname,
            "--horizon",
            str(horizon),
            "--max_sample_complexity",
            "1000000",
            "--num_runs",
            "5",
            "-o",
            out_dir / f"consolidated_{alg}.json",
            "--optimal_return",
            str(optimal_return),
        ]
        if alg == "gorp":
            k = np.random.randint(1, 3)
            command.extend(["--k", str(k)])
        if mdp_type == "mdps_with_exploration_policy_sb3" and alg == "gorp":
            command.extend(
                [
                    "--exploration_policy",
                    os.path.dirname(mdp_fname) + "/exploration_policy.npy",
                ]
            )
        subprocess.check_call(command)

        with open(out_dir / f"consolidated_{alg}.json") as results_file:
            results = json.load(results_file)

        if alg == "gorp":
            assert results.keys() == {
                "actions",
                "mean_rewards",
                "final_returns",
                "sample_complexities",
                "median_final_return",
                "median_sample_complexity",
                "timesteps_per_iteration",
            }
        else:
            assert results.keys() == {
                "actions",
                "final_returns",
                "sample_complexities",
                "median_final_return",
                "median_sample_complexity",
            }

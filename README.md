# Bridging RL Theory and Practice with the Effective Horizon

This repository contains code and data for the paper [Bridging RL Theory and Practice with the Effective Horizon](https://arxiv.org/abs/2304.09853). It includes the BRIDGE dataset and the programs used to construct and analyze it.

Part of the code is written in Python and part in Julia. We used Julia for the programs that construct and analyze the tabular representations of the MDPs in BRIDGE, due to its speed and native support for multithreading. We used Python and [RLlib](https://www.ray.io/rllib) to run the deep RL experiments.

## Installation

1. Install [Python](https://www.python.org/) 3.8, 3.9, or 3.10. Python 3.11 is currently unsupported because [MiniGrid](https://minigrid.farama.org/) does not support it. If you want to run any of the Julia scripts, install [Julia](TODO) 1.8 or later (earlier versions may work but are untested).
2. Clone the repository:

        git clone https://github.com/cassidylaidlaw/effective-horizon.git
        cd effective-horizon

3. Install pip requirements:

        pip install -r requirements.txt

4. Install Julia requirements:

        julia --project=EffectiveHorizon.jl -e "using Pkg; Pkg.instantiate()"

## Data and Pretrained Models

Coming soon!

## Usage

This section explains how to get started with using the code and how to run the experiments from the paper.

### Environments in BRIDGE

All of the environments in BRIDGE are made available as gym environments.

**Atari:** Atari environments follow the naming convention `BRIDGE/$ROM_$HORIZON_fs$FRAMESKIP-v0`. For instance, Pong with a horizon of 50 and frameskip of 30 can be instantiated via `gym.make("BRIDGE/pong_50_fs30-v0")`.

**Procgen:** Procgen environments follow the naming convention `BRIDGE/$GAME_$DIFFICULTY_l$LEVEL_$HORIZON_fs$FRAMESKIP-v0`. For instance, to instantiate CoinRun easy level 0 with a horizon of 10 and frameskip of 8, run `gym.make("BRIDGE/coinrun_easy_l0_10_fs8-v0")`.

**MiniGrid:** MiniGrid environments follow the naming convention `BRIDGE/MiniGrid-$ENV-v0`. For instance, run `gym.make("BRIDGE/MiniGrid-Empty-5x5-v0")` or `gym.make("BRIDGE/MiniGrid-KeyCorridorS3R1-v0")`. There are also versions of the MiniGrid environments with shaped reward functions. There are three shaping functions used in the paper: `Distance`, `OpenDoors`, and `Pickup`. To use the environments with these shaping functions, add the shaping functions and then `Shaped` after the environment name. For instance, run `gym.make("BRIDGE/MiniGrid-Empty-5x5-DistanceShaped-v0")` or `gym.make("BRIDGE/MiniGrid-UnlockPickup-OpenDoorsPickupShaped-v0")`.

### Deep RL training

**PPO:** To train PPO on the environments in GORP, run:

    python -m effective_horizon.experiments.train with env_name="BRIDGE/pong_50_fs30-v0" \
    run=PPO train_batch_size=10000 rollout_fragment_length=100 sgd_minibatch_size=1000 \
    num_sgd_iter=10 num_training_iters=500 seed=0

We tuned `train_batch_size` from the choices of 1,000, 10,000, and 100,000; the `num_training_iters` was set appropriately to 5,000, 500, or 50 respectively so that the total number of environment steps over the course of training was 5 million.

**DQN:** To train DQN on the environments in GORP, run:

    python -m effective_horizon.experiments.train with env_name="BRIDGE/pong_50_fs30-v0" \
    run=FastDQN train_batch_size=10000 rollout_fragment_length=100 sgd_minibatch_size=1000 \
    num_training_iters=1000000 stop_on_timesteps=5000000 seed=0 epsilon_timesteps="$EPSILON_TIMESTEPS" \
    dueling=True double_q=True prioritized_replay=True learning_starts=0 simple_optimizer=True

We tuned `epsilon_timesteps` from the choices of 500,000 and 5,000,000.

For both PPO and DQN, the level of parallelism can be chosen by adding `num_workers=N` to the command, which will start N worker processes collecting rollouts in parallel.

### Constructing and analyzing tabular MDPs

Coming soon!

## Citation

If you find this repository useful for your research, please cite our paper as follows:

    @inproceedings{laidlaw2023effectivehorizon,
      title={Bridging RL Theory and Practice with the Effective Horizon},
      author={Laidlaw, Cassidy and Russell, Stuart and Dragan, Anca},
      booktitle={arXiv preprint},
      year={2023}
    }

## Contact

For questions about the paper or code, please contact cassidy_laidlaw@berkeley.edu.

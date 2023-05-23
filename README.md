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

**PPO:** To train PPO on the environments in BRIDGE, run:

    python -m effective_horizon.experiments.train with env_name="BRIDGE/pong_50_fs30-v0" \
    run=PPO train_batch_size=10000 rollout_fragment_length=100 sgd_minibatch_size=1000 \
    num_sgd_iter=10 num_training_iters=500 seed=0

We tuned `train_batch_size` from the choices of 1,000, 10,000, and 100,000; the `num_training_iters` was set appropriately to 5,000, 500, or 50 respectively so that the total number of environment steps over the course of training was 5 million.

**DQN:** To train DQN on the environments in BRIDGE, run:

    python -m effective_horizon.experiments.train with env_name="BRIDGE/pong_50_fs30-v0" \
    run=FastDQN train_batch_size=10000 rollout_fragment_length=100 sgd_minibatch_size=1000 \
    num_training_iters=1000000 stop_on_timesteps=5000000 seed=0 epsilon_timesteps="$EPSILON_TIMESTEPS" \
    dueling=True double_q=True prioritized_replay=True learning_starts=0 simple_optimizer=True

We tuned `epsilon_timesteps` from the choices of 500,000 and 5,000,000.

For both PPO and DQN, the level of parallelism can be chosen by adding `num_workers=N` to the command, which will start N worker processes collecting rollouts in parallel.

### Constructing and analyzing tabular MDPs

There are a number of scripts, mostly written in Julia, that we used to construct and analyze the tabular representations of the MDPs in the BRIDGE dataset. All of the Julia scripts can be sped up by running them with multiple threads via the flag `--threads`, e.g., `julia --threads=16 --project=EffectiveHorizon.jl ...`. Before running any of these, make sure to set your PYTHONPATH to include the `effective_horizon` package via `export PYTHONPATH=$(pwd)`. Each script is listed below.

**Constructing tabular MDPs:** the `construct_mdp.jl` script constructs tabular representations of the MDPs in BRIDGE.

  * To construct the tabular representation of an Atari MDP, run
   
        julia  --project=EffectiveHorizon.jl EffectiveHorizon.jl/src/construct_mdp.jl \
            --rom freeway \
            -o path/to/store/mdp \
            --horizon 10 \
            --frameskip 30 \
            --done_on_life_lost \
            --save_screens
    
    For the `skiing_10_fs30` MDP, additionally use the option `--noops_after_horizon 200`.
    
  * For a Procgen MDP, run

        julia --project=EffectiveHorizon.jl EffectiveHorizon.jl/src/construct_mdp.jl \
            --env_name maze \
            --distribution_mode easy \
            --level 0 \
            -o path/to/store/mdp \
            --horizon 30 \
            --frameskip 1 \
            --save_screens

  * For a MiniGrid MDP, run

        julia --project=EffectiveHorizon.jl EffectiveHorizon.jl/src/construct_mdp.jl \
            --minigrid \
            --env_name BRIDGE/MiniGrid-KeyCorridorS3R1-v0 \
            -o path/to/store/mdp \
            --horizon 100000000 \
            --frameskip 1

See the appendices of the paper for the exact values of horizon and frameskip used for each Atari/Procgen MDP in BRIDGE. We used the above values for all MiniGrid MDPs, since it is actually possible to enumerate every state in the MiniGrid environments.

This script outputs three files to the directory specified after `-o`:

  * `mdp.npz`: the full tabular representation of the MDP with all states that differ at all in their internal environment representations.
  * `consolidated.npz`: in this representation, states which are indistinguishable (i.e., every sequence of actions leads to the same sequence of rewards and screens) are combined. This is what we used for analysis in the paper.
  * `consolidated_ignore_screen.npz`: similar to `consolidated.npz`, except that we do not consider screens for determinining indistinguishability. That is, states are combined if every sequence of actions leads to the same sequence of rewards.

**Analyzing MDPs:** the `analyze_mdp.jl` script performs various analyses of MDPs, including those used to calculate many of the sample complexity bounds in our paper. You can run it with the command

    julia --project=EffectiveHorizon.jl EffectiveHorizon.jl/src/analyze_mdp.jl \
        --mdp path/to/mdp/consolidated.npz \
        --horizon 10

Replace the horizon with the appropriate value for the environment. You can also specify an exploration policy with the option `--exploration_policy path/to/exploration_policy.npy` (see exploration policy section below for more details). This will output a few files:

  * `consolidated_analyzed.json`: contains various analysis metrics for the environment, including:
      - `min_k`: minimum value of $k$ for which the MDP is $k$-QVI-solvable.
      - `epw`: the effective planning window $W$.
      - `effective_horizon_results`:
          * `effective_horizon`: a bound on the effective horizon using Theorem 5.4.
      - `min_occupancy_results`: used to calculate the covering length $L$. It can be bounded above by `log(2 * num_states * num_actions) / min_state_action_occupancy`.
  * `consolidated_analyzed_value_dists_*.npy`: these numpy arrays (one for each timestep) contain the full distribution over rewards-to-go when following the exploration policy from each state.

**Computing bounds on the effective horizon:** the `compute_gorp_bounds.jl` script uses the techniques in Appendix C to give more precise bounds on the effective horizon. It can be run with the command

    julia --project=EffectiveHorizon.jl EffectiveHorizon.jl/src/compute_gorp_bounds.jl \
    --mdp path/to/mdp/consolidated.npz \
    --use_value_dists \
    --horizon 10 \
    --max_k 1 \
    -o path/to/output.json

The `--use_value_dists` option relies on the outputs of the `analyze_mdp.jl` script to give tighter bounds. The `--max_k` option specifies the maximum value of $k$ for which a bound on $H_k$ will be calculated. Higher values of $k$ take exponentially longer to run so we recommend starting with a small value (1-3) and increasing it if the bounds are not satisfactory. This script also takes the `--exploration_policy` option, similar to `analyze_mdp.jl`.

The will produce an output JSON file with the following results:

  * `sample_complexity`: the best bound on the sample complexity of GORP over all values of $k$.
  * `effective_horizon`: the best bound on the effective horizon over all values of $k$.
  * `k_results`: bounds on the sample complexity and effective horizon for specific values of $k$.

### Running GORP empirically

To use GORP to learn a policy for an environment, we provide both a gym-compatible Python implementation and a faster, parallelized Julia implementation which can run on the tabular MDPs in BRIDGE.

**Using Python:** To train GORP, run:

    python -m effective_horizon.experiments.train with env_name="BRIDGE/pong_50_fs30-v0" \
    run=GORP episodes_per_action_seq=M seed=0

Replace M with the desired parameter $m$ for running GORP. This implementation always uses $k = 1$.

To replicate the long-horizon Atari experiments, run

    python -m effective_horizon.experiments.train with run=GORP \
    env_name=atari horizon=27000 frameskip=4 deterministic=True rom_file=pong \
    reward_scale=1 done_on_life_lost=True gamma=0.99 num_training_iters=1000000 seed=0

Set `rom_file` to the desired game and `reward_scale` appropriately according to the reward scales given in the appendix.

**Using Julia:** To train GORP with Julia, run:

    julia --project=EffectiveHorizon.jl EffectiveHorizon.jl/src/run_gorp.jl \
    --mdp path/to/mdp/consolidated.npz \
    --horizon 10 \
    --max_sample_complexity 100000000 \
    --num_runs 101 \
    --optimal_return OPTIMAL_RETURN \
    --k K \\
    -o path/to/output.json

This script works a bit differently from the Python one—given a value of $k$ and the optimal return for the MDP, it searches for the minimum value of $m$ such that GORP finds an optimal policy at least half the time.

It also takes the `--exploration_policy` option, similarly to `analyze_mdp.jl` and `compute_gorp_bounds.jl`.

### Exploration policies

This section describes our experiments on using the effective horizon to understand initializing deep RL with a pretrained policy. We trained exploration policies for many of the Atari and Procgen environments.

**Training an exploration policy for Atari:** we trained these policies using behavior cloning on the [Atari-HEAD dataset](https://arxiv.org/abs/1903.06754). After downloading the data from [Zenodo](https://zenodo.org/record/3451402), convert it to RLlib format by running

    python -m effective_horizon.experiments.convert_atari_head_data with \
    data_dir=path/to/atari_head/freeway out_dir=path/to/rllib_data

Then, run this command on the output of the previous command to filter the actions to the minimal set for each Atari game:

    python -m effective_horizon.experiments.filter_to_minimal_actions with \
    data_dir=path/to/rllib_data rom=freeway out_dir=path/to/rllib_minimal_action_data

Finally, train a behavior cloned policy:

    python -m rl_theory.experiments.train_bc with env_name=mdps/freeway_10_fs30-v0 \
    input=path/to/rllib_minimal_action_data log_dir=data/logs num_workers=0 \
    num_workers=0 entropy_coeff=0.1

This will create a number of checkpoint files under `data/logs`; we use `checkpoint-100` for the exploration policy.

**Training an exploration policy for Procgen:** we trained these policies using PPO on a disjoint set of Procgen levels from those contained in BRIDGE. To train an exploration policy for Procgen, run

    python -m effective_horizon.experiments.train with env_name=procgen procgen_env_name=maze \
    run=PPO train_batch_size=10000 rollout_fragment_length=100 sgd_minibatch_size=1000 \
    num_sgd_iter=10 num_training_iters=2500 seed=0 log_dir=data/logs entropy_coeff=0.1

Again, this will create a number of checkpoint files under `data/logs`; we use `checkpoint-2500`.

**Generating tabular versions of exploration policies:** the policies created by the above commands cannot immediately be used for analysis since they are represented by neural networks rather than tabularly. To convert the neural network policies to tabular representations, take the following steps:

  1. For Atari MDPs, construct a "framestack" version of the MDP. Since neural network policies for Atari typically take in the last few frames in addition to the current one, we must construct a new tabular representation with states based on multiple frames instead of just one. Run

        python -m effective_horizon.experiments.convert_atari_mdp_to_framestack with mdp=path/to/mdp/consolidated.npz horizon=10 out=path/to/mdp/consolidated_framestack.npz

  2. Now, construct the tabular policy. Run

        python -m effective_horizon.experiments.construct_tabular_policy with mdp=path/to/mdp/consolidated.npz checkpoint=path/to/checkpoint horizon=10 run=RUN out=path/to/mdp/exploration_policy.npy
    
    Replace the horizon with the appropriate horizon and specify `run=BC` for Atari and `run=PPO` for Procgen. The resulting `exploration_policy.npy` file can be passed to the analysis scripts as described above.

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

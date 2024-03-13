# BRIDGE dataset

This directory contains the BRIDGE dataset (Bridging the RL Interdisciplinary Divide with Grounded Environments), as introduced in the paper [Bridging RL Theory and Practice with the Effective Horizon](https://arxiv.org/abs/2304.09853).

## Contents

The data is organized as follows:

 * `summary.csv`: this contains a table summarizing the properties of the MDPs in the dataset, with one MDP per row. The columns are as follows:
    * `mdp_name`: the MDP name, which corresponds to the folder under which its data is stored as well as the ID of the corresponding `gym` environment.
    * `origin`: the collection of environments from which the MDP was taken: either `atari`, `procgen`, or `minigrid`.
    * `type`: either `base` for the main MDPs or `reward_shaping` for the reward-shaped versions.
    * `base_mdp_name`: for reward-shaped MDPs, the name of the unshaped version.
    * `horizon`: the horizon of the MDP.
    * `num_actions`: number of actions.
    * `num_state_consolidated`: the number of states in the consolidated version of the MDP, which is the version included in the dataset.
    * `reward_min`: minimum possible reward in the MDP.
    * `reward_exploration`: the reward achieved by the exploration policy (the random policy).
    * `reward_optimal`: the maximum possible reward.
    * `reward_optimal_probability`: the probability of achieving the maximum possible reward in a single episode while following the exploration policy.
    * `reward_factor`: when running deep RL algorithms, we divide rewards by this factor to make them more uniform across environments. See Appendix F.1 for details.
    * `k_min`: the minimum value of k for which the MDP is k-QVI-solvable.
    * `epw`: the effective planning window (W) for the MDP.
    * `min_state_action_occupancy` and `min_sum_state_action_occupancy`: used in the calculation of covering length; see `analyze_mdp.jl` for how this is computed.
    * `effective_horizon`: our best bound on the effective horizon of the MDP. This is the minimum of `effective_horizon_simple` and the (often tighter) bounds computed based on the methods in Appendix C of the paper.
    * `effective_horizon_simple`: a bound on the effective horizon using Theorem 5.4 in the paper.
    * `reward_ppo`, `reward_dqn`, and `reward_gorp`: the maximum reward achieved by PPO, DQN, and GORP when run in the MDP. PPO and DQN are run for 5 million timesteps; GORP is run for 100 million.
    * `reward_gorp_k_1`: the maximum reward achieved by GORP when only using k = 1.
    * `sample_complexity_ppo`, `sample_complexity_dqn`, `sample_complexity_gorp`, `sample_complexity_gorp_k_1`: the empirical sample complexity of each of the above RL algorithms, which is the median timestep at which the algorithm first found an optimal policy, or `inf` if it never did.
 * `mdps` directory: this contains the 155 base MDPs in the BRIDGE dataset along with analysis results. Each MDP is in a single subdirectory which contains the following files:
    * `consolidated.npz`: the tabular representation of the MDP with consolidated states. States are consolidated if any sequence of actions from them will always lead to the same sequence of observations and rewards. The file format is a NumPy NPZ archive with two arrays, `transitions` and `rewards`, each of shape `(num_states, num_actions)`. `transitions[state, action]` gives the index of the next state reached by taking action `action` in state `state`, or -1 if the next state is terminal; `rewards[state, action]` gives the reward accompanying the aforementioned transition. The intial state has index 0.
    * `consolidated_analyzed.json`: the output of running `analyze_mdp.jl` on the consolidated MDP.
    * `consolidated_gorp_bounds.json`: the output of `compute_gorp_bounds.jl`, which uses the techniques in Appendix C of the paper to give tigher bounds on the effective horizon. The dictionary under `k_results` gives a bound on $H_k$ for each value of k.
    * `gorp_1.json`, `gorp_2.json`, etc.: the output of `run_gorp.jl`, which runs the GORP algorithm on the tabular representation of the MDP. `gorp_1.json` contains the results of running GORP with k = 1, `gorp_2.json` with k = 2, and so on. The keys are as follows:
       * `median_sample_complexity`: median sample complexity across all random seeds.
       * `median_final_return`: median total reward across all random seeds. If the median sample complexity is finite, this should be equal to the optimal return.
       * `sample_complexities` and `final_returns`: individual sample complexities and final returns for each random seed.
       * `actions`: the action sequences learned by GORP for each random seed. The first list consists of all the first actions, the second element consists of all the second actions, and so on.
 * `mdps_with_reward_shaping` directory: this contains the reward shaped versions of the MiniGrid MDPs. The files in each subdirectory are analogous to those in the `mdps` directory.

## Datasheet

We provide a datasheet, as proposed by [Gebru et al.](https://arxiv.org/abs/1803.09010), for the BRIDGE dataset.

### Motivation

#### For what purpose was the dataset created?
We have described the purpose extensively in the paper: we aim to bridge the theory-practice gap in RL. BRIDGE allows this by providing tabular representations of popular deep RL benchmarks such that instance-dependent bounds can be calculated and compared to empirical RL performance.

#### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?
The dataset was created by Cassidy Laidlaw as part of his PhD at the University of California, Berkeley.

#### Who funded the creation of the dataset?
Cassidy Laidlaw was supported by an National Defense Science and Engineering Graduate (NDSEG) Fellowship from the U.S. Department of Defense. Experiments were run on computers purchased using a grant from Open Philanthropy.

#### Any other comments?
No.

### Composition

#### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?
The instances are Markov Decision Processes (MDPs).

#### How many instances are there in total (of each type, if appropriate)?
There are 155 MDPs in BRIDGE. They include 67 MDPs based on Atari games from the Arcade Learning Environment, 55 MDPs based on Procgen games, and 33 MDPs based on MiniGrid gridworlds.

#### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?
The MDPs in BRIDGE are based on a small subset of the many environments that are used for empirically evaluating RL algorithms. We aimed to cover a range of the most popular environments. To make our analysis possible, we excluded environments that were not deterministic or did not have discrete action spaces. We also reduced the horizon of many of the environments to make it tractable to compute their tabular representations.

#### What data does each instance consist of?
For each MDP, we provide the following data:
 * A transition function and a reward function, which are represented as a matrix with an entry for each state-action pair in the MDP.
 * A corresponding `gym` environment that can be used to train policies for the MDP with various RL algorithms.
 * Properties of the MDP that are calculated from its tabular representation, including the effective planning window, bounds on the effective horizon, bounds on the covering length, etc.
 * For MiniGrid MDPs, there are additional versions of the MDP with shaped reward functions  which also include all of the above data.

#### Is there a label or target associated with each instance?
In the paper, we aim to bound and/or estimate the empirical sample complexity of RL algorithms, so these could be considered targets for each instance.

#### Is any information missing from individual instances?
There is no information missing.

#### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?
No.

#### Are there recommended data splits (e.g., training, development/validation, testing)?
No.

#### Are there any errors, sources of noise, or redundancies in the dataset?
We do not believe there are errors or sources of noise in the dataset. The tabular representations of the MDPs have been carefully tested for correspondence with the environments they are based on. There is some redundancy, as many Atari games are represented more than once with varying horizons.

#### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?
The dataset is mostly self-contained, except that the `gym` environments rely on external libraries. There are archival versions of these available through package managers like PyPI.

#### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)?
No.

#### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?
No.

### Collection process

#### How was the data associated with each instance acquired?
The data was collected using open-source implementations of each environment.

#### What mechanisms or procedures were used to collect the data (e.g., hardware apparatuses or sensors, manual human curation, software programs, software APIs)?
We developed a software tool to construct the tabular representations of the MDPs in BRIDGE. We validated the correctness of the tabular MDPs through extensive testing to ensure they corresponded exactly with the `gym` implementations of the environments.

#### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?
The MDPs in BRIDGE were selected from three collections of commonly used RL environments: the Arcade Learning Environment, ProcGen, and MiniGrid. We chose these three collections to represent a broad set of deterministic environments with discrete action spaces. Within each collection, the environments were further filtered based on the criteria described in Appendix F of the paper.

#### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?
Only the authors were involved in the data collection process.

#### Over what timeframe was the data collected?
The dataset was assembled between February 2022 and January 2023. The RL environments from which the MDPs in BRIDGE were constructed were created prior to this; see the original papers where each collection of environments was introduced for more details.

#### Were any ethical review processes conducted (e.g., by an institutional review board)?
No.

### Preprocessing/cleaning/labeling

#### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?
Yes, various preprocessing and analysis was done. See Appendix F.2 in the paper for details.

#### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?
Yes, this is included with the dataset.

#### Is the software that was used to preprocess/clean/label the data available?
Yes, this is available with the rest of our code.

#### Any other comments?
No.

### Uses

#### Has the dataset been used for any tasks already?
The dataset has thus far only been used to validate our theory of the effective horizon in this paper.

#### Is there a repository that links to any or all papers or systems that use the dataset?
There is not. However, we require that any uses of the dataset cite this paper, allowing one to use tools like Semantic Scholar or Google Scholar to find other papers which use the BRIDGE dataset.

#### What (other) tasks could the dataset be used for?
We hope that the BRIDGE dataset is used for further efforts to bridge the theory-practice gap in RL. The dataset could be used to identify other properties or assumptions that hold in common environments, or to calculate instance-dependent sample complexity bounds and compare them to the empirical sample complexity of RL algorithms.

#### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?
As we have already mentioned, BRIDGE is restricted to deterministic MDPs with discrete action spaces and relatively short horizons. This could mean that analyses of the dataset like ours do not generalize to the broader space of RL environments that may have continuous action spaces, stochastic transitions, and/or long horizons. We have included some experiments, like those in Appendix H.1, to show that our theory of the effective horizon generalizes beyond the MDPs in BRIDGE. We encourage others to do the same and we hope to address some of these limitations in the future with extensions to BRIDGE.

#### Are there tasks for which the dataset should not be used?
We do not foresee any particular tasks for which the dataset should not be used.

#### Any other comments?
No.

### Distribution

#### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?
Yes, we distribute the dataset publicly.

#### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?
The dataset is distributed through Zenodo.

#### When will the dataset be distributed?
The dataset is publicly available as of August 2023.

#### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?
It is distributed under CC-BY-4.0.

#### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?
The Atari ROMs used to construct the Atari MDPs in BRIDGE are copyrighted by the original creators of the games. However, they are widely used throughout the reinforcement learning literature and to our knowledge the copyright holders have not complained about this. Since we are not legal experts, we do not know if releasing our dataset violates their copyright, but we do not believe that we are harming them since the tabular representations in BRIDGE are only useful for research purposes and cannot be used to play the games in any meaningful way.

#### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?
No.

#### Any other comments?
No.

### Maintenance

#### Who will be supporting/hosting/maintaining the dataset?
We (the authors) will support and maintain the dataset.

#### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?
Cassidy Laidlaw can be conctacted at cassidy_laidlaw@berkeley.edu or cassidy256@gmail.com.

#### Is there an erratum?
We will record reports of any errors in the dataset and release new versions with descriptions of what was fixed as necessary.

#### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?
We will release new versions of the dataset to correct any reported errors as described above. We may also expand the dataset in the future with more MDPs or new kinds of MDPs, such as stochastic or continuous-action-space MDPs. Any updates will be available on Zenodo.

#### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)?
No.

#### Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers.
We hope to find a host for the dataset that will retain older versions of the dataset. We only plan to maintain the latest version of the dataset, however. We will note this policy in the dataset's description.

#### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?
There is no predefined mechanism to contribute to the dataset, but we will consider external contributions on a case-by-case basis. We encourage others to extend and build on the dataset.

#### Any other comments?
No.


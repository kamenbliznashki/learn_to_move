# NeurIPS Learn to Move: Walk Around Challenge Solution

Sixth place solution for the [NeurIPS Learn to Move: Walk Around Challenge](https://www.aicrowd.com/challenges/neurips-2019-learning-to-move-walk-around) part of the conference competition [track](https://neurips.cc/Conferences/2019/CallForCompetitions).

See my [blog post]() for more information.

The repo `master` contains the code that performed best in the end: learning algorithm is Soft Actor Critic (cf [paper](https://arxiv.org/abs/1801.01290)) with automatic/learnable entropy adjustment (cf [paper](https://arxiv.org/abs/1812.11103)) and a BNAF normalizing flow (cf [paper](https://arxiv.org/abs/1904.04676)) on top of the policy; exploration algorithm is Self-supervised Exploration via Disagreement (cf [paper](https://arxiv.org/abs/1906.04161)).

I worked on several approaches concurrently, which I branched out. The branches are named after the approach implemented, e.g. `modelbased` implementing a 'hybrid' model-based RL, `deepexpl` implementing ideas from Deep Exploration using Prior Functions, `nstep_returns`, etc. The code in each branch is as it were. Bottom line is, code outside of the master and algorithm used above is staler (e.g. I ran various experiments with DDPG until halfway through the competition and used TD3 only at the very beginning, thus running TD3 on the final codebase may require some tweaking).

### Dependencies
* Competition simulation platform -- opensim 4.0 and osim-rl 3.0.2 (cf http://osim-rl.stanford.edu)
* gym 0.13.1
* python 3.6
* tensorflow 1.14
* numpy 1.16
* scipy 1.3
* mpi4pi 3.0.2 (+ working MPI; only needed if running distributed DDPG)

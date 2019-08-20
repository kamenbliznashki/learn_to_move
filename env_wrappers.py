from abc import ABC, abstractmethod
import contextlib
import multiprocessing as mp
import os
import os.path as osp
import time
import csv

import numpy as np

import gym
from collections import deque

from osim.env import L2M2019Env, OsimModel



class L2M2019EnvBaseWrapper(L2M2019Env):
    """ Wrapper to move certain class variable to instance variables """
    def __init__(self, **kwargs):
        self._model = kwargs.pop('model', '3D')
        stepsize = kwargs.pop('stepsize', 0.01)
        self._visualize = kwargs.get('visualize', False)
        self._osim_model = None
        super().__init__(**kwargs)  # NOTE -- L2M2019Env at init calls get_model_key() which pulls self.model; -> setting _model here initializes the model to 2D or 3D
        self.osim_model.stepsize = stepsize

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def change_model(self, model):
        # overwrite method so as to remove arguments `difficulty` and `seed` in the parent change_model method
        if self.model != model:
            self.model = model
            self.load_model(self.model_paths[self.get_model_key()])

    @property
    def osim_model(self):
        return self._osim_model

    @osim_model.setter
    def osim_model(self, model):
        self._osim_model = model

    @property
    def visualize(self):
        return self._visualize

    @visualize.setter
    def visualize(self, new_state):
        assert isinstance(new_state, bool)
        self._visualize = new_state

    # match evaluation env
    def step(self, action):
        return super().step(action, project=True, obs_as_dict=True)

    def reset(self, **kwargs):
        obs_as_dict = kwargs.pop('obs_as_dict', True)
        return super().reset(obs_as_dict=obs_as_dict, **kwargs)

class Obs2VecEnv(gym.Wrapper):
    def __init__(self, env=None, **kwargs):
        super().__init__(env)
        obs_dim = env.observation_space.shape[0] - 2*11*11
        self.observation_space = gym.spaces.Box(np.zeros(obs_dim), np.zeros(obs_dim))

    def obs2vec(self, obs_dict):
        # Augmented environment from the L2R challenge
        res = []

        # target velocity field (in body frame)
#        res += obs_dict['v_tgt_field'].flatten().tolist()

        res.append(obs_dict['pelvis']['height'])
        res.append(obs_dict['pelvis']['pitch'])
        res.append(obs_dict['pelvis']['roll'])
        res.append(obs_dict['pelvis']['vel'][0]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][1]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][2]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][3])
        res.append(obs_dict['pelvis']['vel'][4])
        res.append(obs_dict['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])
        return res

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return self.obs2vec(o), r, d, i

    def reset(self, **kwargs):
        o = self.env.reset(**kwargs)
        return self.obs2vec(o)

class RandomPoseInitEnv(gym.Wrapper):
    def reset(self, **kwargs):
        pose = np.array([np.random.uniform(0,1),          # forward speed
                         0,#np.random.uniform(-1,1),          # righward speed
                         np.random.uniform(0.94, 0.96),    # pelvis_height
                         0,#np.random.uniform(-0.25, 0.25),   # trunk lean
                         0*np.pi/180,                      # [right] hip adduct
                         np.random.uniform(-1,1),          # hip flex
                         np.random.uniform(-1,0),          # knee extend
                         np.random.uniform(-0.935,0.935),  # ankle flex
                         0*np.pi/180,                      # [left] hip adduct
                         np.random.uniform(-1,1),          # hip flex
                         np.random.uniform(-1,0),          # knee extend
                         np.random.uniform(-0.935,0.935)]) # ankle flex
        return self.env.reset(init_pose=pose, **kwargs)

class ZeroOneActionsEnv(gym.Wrapper):
    """ transform action from tanh policies in (-1,1) to (0,1) """
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.env.action_space.low -= 1

    def step(self, action):
        # input in [-1,1], output in [0,1]
        return self.env.step((action + 1)/2)

class PoolVTgtEnv(gym.Wrapper):
    def __init__(self, env=None, **kwargs):
        super().__init__(env, **kwargs)
        obs_dim = env.observation_space.shape[0] - 2*11*11 + 2*3*3
        self.observation_space = gym.spaces.Box(np.zeros(obs_dim), np.zeros(obs_dim))

    def pool_vtgt(self, obs):
        pooled_vtgt = obs['v_tgt_field'].reshape(2,11,11)[:,::2,::2].reshape(2,3,2,3,2).mean((2,4))  # subsample and 2x2 avg pool
        obs['v_tgt_field'] = pooled_vtgt
        return obs

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return self.pool_vtgt(o), r, d, i

    def reset(self, **kwargs):
        o = self.env.reset(**kwargs)
        return self.pool_vtgt(o)

class FallPenaltyEnv(gym.Wrapper):
    def step(self, action):
        o, r, d, i = self.env.step(action)

        # if pelvis below 0.6, OsimEnv returns done; penalize this
        if o['pelvis']['height'] < 0.6:
            r -= 10

        return o, r, d, i

class RewardAugEnv(gym.Wrapper):
    def step(self, action):
        o, r, d, i = self.env.step(action)

        rewards = {}
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # gyroscope -- l2 penalize pitch and roll
        eps = 0.01
        pitch = o['pelvis']['pitch']**2
        roll = o['pelvis']['roll']**2
        yaw = abs(self.get_state_desc()['joint_pos']['ground_pelvis'][2])
#        yaw_vel = o['pelvis']['vel'][5]**2
        rewards['pitch'] = 0.1 if pitch < eps else -pitch
        rewards['roll'] = 0.1 if roll < eps  else -roll
        rewards['yaw'] = 0.1 if yaw < 0.3 else -sigmoid(yaw)
#        rewards['yaw_vel'] = - sigmoid(yaw_vel - 5)

#        # distance from vtgt min
#        def distance_from_min(grid):
#            x, y = np.nonzero(grid == np.min(grid))
#            x = x[0] - grid.shape[0]//2
#            y = y[0] - grid.shape[1]//2
#            return np.sqrt(x**2 + y**2)
#
#        x, y = np.asarray(o['v_tgt_field'])
#        dist = np.mean([distance_from_min(x), distance_from_min(y)])
#        rewards['dist'] = 0.1 * dist

        speed = np.sqrt(o['pelvis']['vel'][0]**2 + o['pelvis']['vel'][1]**2)
        rewards['speed'] = sigmoid(speed - 5)

        # if pelvis below 0.6, OsimEnv returns done; penalize this
        rewards['height'] = 0.1 if o['pelvis']['height'] > 0.6 else -1

#        print('Augmented rewards: {}'.format(rewards))
        r += sum(rewards.values())
        return o, r, d, i

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, n_skips=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self.obs_buffer = deque(maxlen=4)
        self.n_skips       = n_skips

    def step(self, action):
        total_reward = 0
        done = None
        for _ in range(self.n_skips):
            obs, reward, done, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        # TODO  -- max on all elements of the state space?; perhaps on the actuation strenght?
        #       -- test with/without max
        max_frame = obs#np.max(np.stack(self.obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self.obs_buffer.clear()
        obs = self.env.reset(**kwargs)
        self.obs_buffer.append(obs)
        return obs



# --------------------
# Monitoring wrappers
#
#   Modified from OpenAI Baselines https://github.com/openai/baselines/blob/master/baselines/bench/monitor.py
#       Changes: -- merged ResultsWriter class; removed args for reset keywords and info keywords
#
# --------------------

class Monitor(gym.core.Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename):
        super().__init__(env)
        self.tstart = time.time()
        if filename:
            if osp.isdir(filename):
                filename = osp.join(filename, self.EXT)
            else:
                filename = filename + '.' + self.EXT
            # write to csv
            header = {'t_start': time.strftime("%Y-%m-%d_%H-%M-%S"), 'model': env.model}
            header = '# {} \n'.format(header)
            self.f = open(filename, "at")
            if not osp.exists(filename):
                self.f.write(header)
            self.logger = csv.DictWriter(self.f, fieldnames=('return', 'length', 'time'))
            self.logger.writeheader()
            self.f.flush()
        else:
            self.f = None
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()

    def reset(self, **kwargs):
        self.rewards = []
        self.needs_reset = False
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return (ob, rew, done, info)

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            # write to log
            epinfo = {"return": round(eprew, 6), "length": eplen, "time": round(time.time() - self.tstart, 6)}
            self.write_row(epinfo)

        self.total_steps += 1

    def close(self):
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times


# --------------------
# Multiprocessing wrappers
#
#   Modified from OpenAI Baselines https://github.com/openai/baselines/blob/master/baselines/common/
#       Changes -- in VecEnv -- removed render, get_images, get_viewer, metadata (no support in Osim)
#               -- in DummyVecEnv -- removed obs in dict dtype returned by stepping an env (osim env returns a list)
#               -- in SubprocVecEnv -- removed get_images; rendering from pipe commands in worker fn in SubprocVecEnv
#
# --------------------

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that each observation becomes an batch of observations,
    and expected action is a batch of actions to be applied per-environment.
    """
    closed = False

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of observations, or a dict of observation arrays.
        If step_async is still doing work, that work will be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step with the given actions.
        Call step_wait() to get the results of the step. You should not call this if a step_async run is already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict ofarrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """ Clean up the  extra resources, beyond what's in this base class. Only runs when not self.closed. """
        pass

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """ Step the environments synchronously. This is available for backwards compatibility. """
        self.step_async(actions)
        return self.step_wait()

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is, the step and reset commands are send to one environment
    at a time. Useful when debugging and when num_env == 1 (in the latter case, avoids communication overhead)
    """
    def __init__(self, env_fns):
        """
        Arguments:
        env_fns: iterable of callables functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)

        self.buf_obs   = np.zeros((self.num_envs,) + env.observation_space.shape, dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs, 1), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs, 1), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

    def step_async(self, actions):
        assert len(actions) == self.num_envs, 'cannot match actions {} to {} environments'.format(actions, self.num_envs)
        self.actions = actions

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self.buf_obs[e] = np.asarray(obs)
        return np.copy(self.buf_obs), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self.buf_obs[e] = np.asarray(obs)
        return np.copy(self.buf_obs)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, [reward], [done], info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send((env.observation_space, env.action_space, env.spec))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()

@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default. If the child process has MPI environment variables,
    MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are
    starting multiprocessing Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_', 'PMIX_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)

class CloudpickleWrapper:
    """ Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle) """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(VecEnv):
    """ VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes. """
    def __init__(self, env_fns, context='spawn'):
        """
        Arguments:
        env_fns: iterable of callables functions that build environments
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                    for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, 'Trying to operate on a SubprocVecEnv after calling close()'

    def __del__(self):
        if not self.closed:
            self.close()


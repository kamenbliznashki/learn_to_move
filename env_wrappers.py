from abc import ABC, abstractmethod
import contextlib
import multiprocessing as mp
import os
import os.path as osp
import time
import csv

import numpy as np
from scipy.signal import fftconvolve

import gym
from collections import deque
import tabulate

from osim.env import L2M2019Env



class L2M2019EnvBaseWrapper(L2M2019Env):
    """ Wrapper to move certain class variable to instance variables """
    def __init__(self, **kwargs):
        self._model = kwargs.pop('model', '3D')
        stepsize = kwargs.pop('stepsize', 0.01)
        self._visualize = kwargs.get('visualize', False)
        self._osim_model = None
        super().__init__(visualize=kwargs['visualize'],
                         integrator_accuracy=kwargs['integrator_accuracy'],
                         difficulty=kwargs['difficulty'])  # NOTE -- L2M2019Env at init calls get_model_key() which pulls self.model; -> setting _model here initializes the model to 2D or 3D
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

class L2M2019ClientWrapper:
    # L2M variables
    LENGTH0 = 1 # leg length

    # gym env variables -- conform to gym env, so can be wrapped later
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    obs_dim = 339
    act_dim = 22
    observation_space = gym.spaces.Box(np.zeros(obs_dim), np.zeros(obs_dim))
    action_space = gym.spaces.Box(np.zeros(act_dim), np.ones(act_dim))

    def __init__(self, client=None):
        self.client = client

    def step(self, action):
        return self.client.env_step(action.tolist())

    def reset(self):
        return self.client.env_reset()

    def create(self):
        return self.client.env_create()

    def submit(self):
        return self.client.submit()

class Obs2VecEnv(gym.Wrapper):
#    def __init__(self, env=None, **kwargs):
#        super().__init__(env)
#        # NOTE NOTE NOTE -- if removing the vtgt from the obs vector below; need to update the dims here;
#        #                    the previous env wrapper PoolVtgtEnv adjusted for the pooling operation
#        #                    NOTE this should match what the state predictors exploration models take in as offset to indices selected
#        obs_dim = env.observation_space.shape[0] - 2*3 #2*11*11
#        self.observation_space = gym.spaces.Box(np.zeros(obs_dim), np.zeros(obs_dim))

    def obs2vec(self, obs_dict):
        # Augmented environment from the L2R challenge
        res = []

        # target velocity field (in body frame)
        res += obs_dict['v_tgt_field'].flatten().tolist()

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
        if not o:  # submission client returns False at the end
            return o
        return self.obs2vec(o)

    def create(self):
        return self.obs2vec(self.env.create())

class RandomPoseInitEnv(gym.Wrapper):
    def __init__(self, env=None, anneal_start_step=1000, anneal_end_step=2000, **kwargs):
        # if avg episode length starts at 20 steps then annealing begins after about 20k model steps; 
        #   if episode length goes to 100 steps in between resets;
        #   then by 2k resets we'll be between 20*2k=40k and 100*2k=200k model steps
        # on training restart from a loaded model, annealing restarts (this leads to some instability when loading a model)
        super().__init__(env)
        # anneal pose to zero-pose
        self.anneal_start_step = anneal_start_step
        self.anneal_end_step = anneal_end_step
        self.anneal_step = 0

    def reset(self, **kwargs):
        seed = kwargs.get('seed', None)
        if seed is not None:
            state = np.random.get_state()
            np.random.seed(seed)

        # construct random pose
        #  init pose vector is:
        #       [forward speed
        #        rightward speed
        #        pelvis height
        #        trunk lean
        #        [right] hip adduct
        #        hip flex
        #        knee extend
        #        ankle flex
        #        [left] hip adduct  == - inward / + outward
        #        hip flex           == - in forward (ie knee up in front) direction; + in backward (knee behind) direction
        #        knee extend        == - normal bent knee / + knee bents forward over the knee cap
        #        ankle flex]        == - toes point up / + extend toes away

        x_vel = np.clip(np.abs(np.random.normal(0, 1.5)), a_min=None, a_max=3.5)
        y_vel = np.random.uniform(-0.15, 0.15)
        # foot in the air
        leg1 = [np.random.uniform(0, 0.1), np.random.uniform(-1, 0.3), np.random.uniform(-1.3, -0.5), np.random.uniform(-0.9, -0.5)]
        # foot on the ground
        leg2 = [np.random.uniform(0, 0.1), np.random.uniform(-0.25, 0.05), np.random.uniform(-0.5, -0.25), -0.25]

        pose = [x_vel,
                y_vel,
                0.94,
                np.random.uniform(-0.15, 0.15)]

        if y_vel > 0:
            pose += leg1 + leg2
        else:
            pose += leg2 + leg1

        pose = np.asarray(pose)

        pose *= 1 - np.clip((self.anneal_step - self.anneal_start_step)/(self.anneal_end_step - self.anneal_start_step), 0, 1)
        pose[2] = 0.94

        self.anneal_step += 1

        if seed is not None:
            np.random.set_state(state)

        return self.env.reset(init_pose=pose, **kwargs)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=8):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = np.zeros_like(self.env.action_space.low)

    def reset(self, **kwargs):
        self.env.reset()
        # NOTE -- sample noops
        noops = self.noop_max#np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            o, _, _, _ = self.env.step(self.noop_action)
        return o


class ActionAugEnv(gym.Wrapper):
    """ transform action from tanh policies in (-1,1) to (0,1) """
#    def __init__(self, env, **kwargs):
#        super().__init__(env)
#        action_dim = self.env.action_space.shape[0] - 2
#        # policy actions in [-1,1]
#        self.env.action_space = gym.spaces.Box(-1 * np.ones(action_dim), np.ones(action_dim))

    def step(self, action):
#        # input in [-1,1], output in [0,1]
#        # zero out L/R hip adductors -- let's see if it prevents leg crossing / is necessery to talk
#        action = np.insert(action, [1, 12], -1)  # insert -1 in positions 1 and 12 ie output is (22,) and pos 1 and 12 are -1 (the hip adductors are inactive)
        return self.env.step((action + 1)/2)

class PoolVTgtEnv(gym.Wrapper):
    def __init__(self, env=None, **kwargs):
        super().__init__(env)
        # v_tgt_field pooling; output size = pooled_vtgt + scale of x vel
        self.v_tgt_field_size = 4   # v_tgt_field pooled size for x and y
        self.v_tgt_field_size += 1  # distance to vtgt sink
        # adjust env reference dims
        obs_dim = env.observation_space.shape[0] - 2*11*11 + self.v_tgt_field_size
        self.observation_space = gym.spaces.Box(np.zeros(obs_dim), np.zeros(obs_dim))

    def pool_vtgt(self, obs):
        # transpose and flip over x coord to match matplotliv quiver so easier to interpret
        vtgt = obs['v_tgt_field'].swapaxes(1,2)[:,::-1,:]
        # pool v_tgt_field to (3,3)
        pooled_vtgt = vtgt.reshape(2,11,11)[:,::2,::2].reshape(2,3,2,3,2).mean((2,4))
        # pool each coordinate
        x_vtgt = pooled_vtgt[0].mean(0)  # pool dx over y coord
        y_vtgt = np.abs(pooled_vtgt[1].mean(1))  # pool dy over x coord and return one hot indicator of the argmin
        # y turning direction (yaw tgt) = [left, straight, right]
        y_vtgt_onehot = np.zeros_like(y_vtgt)
        y_vtgt_argsort = y_vtgt.argsort()
        # if target is behind (x_vtgt is negative and y_vtgt is [0, 1, 0] ie argmin is 1, then choose second to argmin to force turn
        y_vtgt_onehot[y_vtgt_argsort[1] if (y_vtgt[1] < 1 and y_vtgt_argsort[0] == 1) else y_vtgt_argsort[0]] = 1
        # distance to vtgt sink
        goal_dist = np.sqrt(x_vtgt[1]**2 + y_vtgt[1]**2)
        # x speed tgt = [stop, go]
        x_vtgt_onehot = (goal_dist > 0.3)
#        print('dx {:.2f}; dy {:.2f}; dxdy {:.2f}; dx_tgt {:.2f}'.format(
#            x_vtgt[1], y_vtgt[1], np.sqrt(x_vtgt[1]**2 + y_vtgt[1]**2), dx_tgt))
        obs['v_tgt_field'] = np.hstack([x_vtgt_onehot, y_vtgt_onehot, goal_dist])
        return obs

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return self.pool_vtgt(o), r, d, i

    def reset(self, **kwargs):
        o = self.env.reset(**kwargs)
        if not o:  # submission client returns False at end
            return o
        return self.pool_vtgt(o)

    def create(self):
        return self.pool_vtgt(self.env.create())

class RewardAugEnv(gym.Wrapper):
    @staticmethod
    def compute_rewards(x_vtgt_onehot, goal_dist, height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw, rf, rl, ru, lf, ll, lu):
        """ note this operates on scalars (when called by env) and vectors (when called by state predictor models) """
        # NOTE -- should be left right symmetric if using symmetric memory

        rewards = {}

        # goals -- v_tgt_field sink
        rewards['vtgt_dist'] = np.clip(np.tanh(1 / np.clip(goal_dist, 0.1, None) - 0.5), 0, None)  # e.g. [0. , 0.165, 0.462, 0.905, 0.999] for goal dist [2. , 1.5, 1. , 0.5, 0.1]
        rewards['vtgt_goal'] = np.where(goal_dist < 0.3, 5 * np.ones_like(goal_dist), np.zeros_like(goal_dist))

        # stability -- penalize pitch and roll
        rewards['pitch'] = - 1 * np.clip(pitch * dpitch, 0, float('inf')) # if in different direction ie counteracting ie diff signs, then clamped to 0, otherwise positive penalty
        rewards['roll']  = - 1 * np.clip(roll * droll, 0, float('inf'))

        # velocity -- reward dx; penalize dy and dz
        rewards['dx'] = np.where(x_vtgt_onehot == 1, 3 * np.tanh(dx), 2 * (1 - np.tanh(5*dx)**2))
        rewards['dy'] = - 2 * np.tanh(2*dy)**2
#        rewards['dz'] = - np.tanh(dz)**2

        # footsteps -- penalize ground reaction forces outward/lateral/(+) (ie agent pushes inward and crosses legs)
#        if ll is not None:
#            rewards['grf_ll'] = - 0.5 * np.tanh(10*ll)
#            rewards['grf_rl'] = - 0.5 * np.tanh(10*rl)

        # falling
        rewards['height'] = np.where(height > 0.70, np.zeros_like(height), -5 * np.ones_like(height))

        return rewards

    def step(self, action):
        yaw_old = self.get_state_desc()['joint_pos']['ground_pelvis'][2]

        o, r, d, i = self.env.step(action)

        # extract data
        x_vtgt_onehot, y_vtgt_onehot, goal_dist = np.split(o['v_tgt_field'], [1, self.v_tgt_field_size - 1], axis=-1)  # split to produce 3 arrays of shape (n,), (n,) and (1,)  where n is half the pooled v_tgt_field
        height, pitch, roll, [dx, dy, dz, dpitch, droll, dyaw] = o['pelvis'].values()
        yaw_new = self.get_state_desc()['joint_pos']['ground_pelvis'][2]
        rf, rl, ru = o['r_leg']['ground_reaction_forces']
        lf, ll, lu = o['l_leg']['ground_reaction_forces']
        # convert to array for compute_rewards fn
        goal_dist = np.asarray(goal_dist)
        height = np.asarray(height)
        dx = np.asarray(dx)

        # compute rewards
        rewards = self.compute_rewards(x_vtgt_onehot, goal_dist, height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw, rf, rl, ru, lf, ll, lu)

        # turning -- reward turning towards the v_tgt_field sink;
        # yaw target is relative to current yaw; so reward if the change in yaw is in the direction and magnitude of the tgt
        delta_yaw = yaw_new - yaw_old
        yaw_tgt = 0.025 * np.array([1, 0, -1]) @ y_vtgt_onehot   # yaw is (-) in the clockwise direction
        rewards['yaw_tgt'] = 2 * (1 - np.tanh(100*(delta_yaw - yaw_tgt))**2)

#        print('grf ll: {:.3f}; grf rl: {:.3f}'.format(ll, rl))
#        print('delta_yaw: ', np.round(delta_yaw, 3), '; yaw_tgt: ', np.round(yaw_tgt, 3))
#        print('dx_tgt: ', dx_tgt, '; y_vtgt: ', y_vtgt_onehot)
#        print('Augmented rewards:\n {}'.format(tabulate.tabulate(rewards.items())))
        i['rewards'] = float(sum(rewards.values()))
        return o, r, d, i

class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, n_skips=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self.n_skips = n_skips

    def step(self, action):
        total_reward = 0
        aug_reward = 0
        for _ in range(self.n_skips):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if 'rewards' in info:
                aug_reward += info['rewards']
                info['rewards'] = aug_reward
            if done:
                break

        return obs, total_reward, done, info


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
            exists = osp.exists(filename)
            self.f = open(filename, "at")
            self.logger = csv.DictWriter(self.f, fieldnames=('return', 'length', 'time'))
            if not exists:
                self.f.write(header)
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

    def __init__(self, num_envs, observation_space, action_space, v_tgt_field_size):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.v_tgt_field_size = v_tgt_field_size

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
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space, getattr(env, 'v_tgt_field_size', 0))

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
                remote.send((env.observation_space, env.action_space, getattr(env, 'v_tgt_field_size', 0), env.spec))
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
        observation_space, action_space, v_tgt_field_size, self.spec = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space, v_tgt_field_size)

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
            try:
                results = [remote.recv() for remote in self.remotes]
            except EOFError:  # nothing to receive / closed by garbage collection
                pass
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, 'Trying to operate on a SubprocVecEnv after calling close()'

    def __del__(self):
        if not self.closed:
            self.close()


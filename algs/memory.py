from collections import namedtuple
import numpy as np


Transition = namedtuple('Transition', ['obs', 'actions', 'rewards', 'dones', 'next_obs'])

class Memory:
    def __init__(self, max_size, observation_shape, action_shape, dtype='float32'):
        self.observation_shape = observation_shape
        self.v_tgt_field_size = observation_shape[0] - (339 - 2*11*11)
        self.action_shape = action_shape
        self.dtype = dtype

        self.obs      = np.zeros((max_size, *self.observation_shape)).astype(dtype)
        self.actions  = np.zeros((max_size, *self.action_shape)).astype(dtype)
        self.rewards  = np.zeros((max_size, 1)).astype(dtype)
        self.dones    = np.zeros((max_size, 1)).astype(dtype)
        self.next_obs = np.zeros((max_size, *self.observation_shape)).astype(dtype)

        self.max_size = max_size
        self.pointer = 0
        self.size = 0
        self.current_obs = None

    def store_transition(self, obs, actions, rewards, dones, next_obs, training=True):
        """ add transition to memory; overwrite oldest if memory is full """
        if not training:
            return

        # assert 2d arrays coming in
        obs, actions, rewards, dones, next_obs = np.atleast_2d(obs, actions, rewards, dones, next_obs)

        B = obs.shape[0]  # batched number of environments
        idxs = np.arange(self.pointer, self.pointer + B) % self.max_size
        self.obs[idxs] = obs
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.dones[idxs] = dones
        self.next_obs[idxs] = next_obs

        # step buffer pointer and size
        self.pointer = (self.pointer + B) % self.max_size
        self.size = min(self.size + B, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, batch_size)
        return Transition(*np.atleast_2d(self.obs[idxs], self.actions[idxs], self.rewards[idxs], self.dones[idxs], self.next_obs[idxs]))

    def initialize(self, env, n_prefill_steps=1000, training=True, policy=None):
        if not training:
            return

        # prefill memory using uniform exploration or policy provided
        if self.current_obs is None:
            self.current_obs = env.reset()

        for _ in range(n_prefill_steps):
            actions = np.random.uniform(-1, 1, (env.num_envs,) + self.action_shape) if policy is None else policy.get_actions(self.current_obs)[0]
            next_obs, r, done, info = env.step(actions)
            r_aug = np.vstack([i.get('rewards', 0) for i in info])
            self.store_transition(self.current_obs, actions, r + r_aug, done, next_obs, training)
            self.current_obs = next_obs

        print('Memory initialized.')

class SymmetricMemory(Memory):

    def compute_mirrored_obs(self, obs):
        # batch_size
        B = obs.shape[0]

        #   copy obs then edit in place
        m_obs = obs.copy()

        # body obs start after v tgt field; pelvis is first 9 components of body obs
        vtgt_obs, body_obs = np.split(m_obs, [self.v_tgt_field_size], axis=-1)
        pelvis_obs, leg_obs = np.split(body_obs, [9], axis=-1)

        #   flip legs
        leg_obs = np.flipud(leg_obs.reshape(B,2,-1)).reshape(B,-1)

        #   3. flip sign of pelvis roll (right->left), y-velocity, roll, yaw
        pelvis_obs[:,[2,4,7,8]] *= -1

        #   4. flip v tgt (NOTE excluding speed scalar -- sync with PoolVtgt for output of the v tgt field)
        y_vtgt_onehot, dx_scale = np.split(vtgt_obs, [self.v_tgt_field_size - 1], axis=-1)
        y_vtgt_onehot = np.fliplr(y_vtgt_onehot)

        # join all flipped elements into mirrored obs
        return np.hstack([y_vtgt_onehot, dx_scale, pelvis_obs, leg_obs])


    # NOTE --- assumes reward is left-right symmetric

    def store_transition(self, obs, actions, rewards, dones, next_obs, training=True):
        # add experience to buffer
        super().store_transition(obs, actions, rewards, dones, next_obs, training)

        # batch_size
        B = obs.shape[0]

        # add mirrored `experience` to buffer -- state and action mirrored along left-right symmetry
        #   flip body obs - legs, roll/yaw, y vel and roll/yaw angular velocities -- for indexing from the observations dict to list see L2M2019Env.get_observation()

        #   1. flip actions for muscles on left and right leg
        m_actions = actions.copy()
        m_actions = np.flipud(m_actions.reshape(B,2,11)).reshape(B,-1)   # eg [0,1,2,3] -> [[0,1],[2,3]] -> [[2,3],[0,1]] -> [2,3,0,1]

        #   2. flip obs vectors
        m_obs = self.compute_mirrored_obs(obs)
        m_next_obs = self.compute_mirrored_obs(next_obs)

        super().store_transition(m_obs, m_actions, rewards, dones, m_next_obs, training)
        del m_actions, m_obs



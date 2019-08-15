from collections import namedtuple
import numpy as np


Transition = namedtuple('Transition', ['obs', 'actions', 'rewards', 'dones', 'next_obs'])

class Memory:
    def __init__(self, max_size, observation_shape, action_shape, reward_scale, dtype='float32'):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.reward_scale = reward_scale
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
        self.rewards[idxs] = rewards * self.reward_scale
        self.dones[idxs] = dones
        self.next_obs[idxs] = next_obs

        # step buffer pointer and size
        self.pointer = (self.pointer + B) % self.max_size
        self.size = min(self.size + B, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, batch_size)
        return Transition(*np.atleast_2d(self.obs[idxs], self.actions[idxs], self.rewards[idxs], self.dones[idxs], self.next_obs[idxs]))

    def initialize(self, env, n_prefill_steps=100):
        # prefill memory using uniform exploration
        if self.current_obs is None:
            self.current_obs = env.reset()

        for _ in range(n_prefill_steps):
            actions = np.random.uniform(-1, 1, (env.num_envs,) + self.action_shape)
            next_obs, r, done, _ = env.step(actions)
            self.store_transition(self.current_obs, actions, r, done, next_obs)
            self.current_obs = next_obs


class SymmetricMemory(Memory):
    def store_transition(self, obs, actions, rewards, dones, next_obs, training=True):
        # add experience to buffer
        super().store_transition(obs, actions, rewards, dones, next_obs, training)

        # add mirrored `experience` to buffer -- state and action mirred along left-right symmetry
        #   flip actions for muscles on left and right leg
        m_actions = actions.copy()
        m_actions = np.flipud(m_actions.reshape(2,11)).flatten()   # ie [0,1,2,3] --> [2,3,0,1]
        #   flip obs for left and right leg -- for indexing from the observations dict to list see L2M2019Env.get_observation()
        m_obs = obs.copy()
        m_next_obs = next_obs.copy()
        leg_obs = m_obs[251:]
        leg_next_obs = m_next_obs[251:]
        m_obs[251:] = np.flipup(leg_obs.reshape(2,-1)).flatten()
        m_next_obs[251:] = np.flipup(leg_next_obs.reshape(2,-1)).flatten()
        #   flip sign of pelvis roll (right->left)
        m_obs[244] = - np.sign(m_obs[244]) * np.abs(m_obs[244])
        #   flip sign of pelvis y-axis velocity
        m_obs[246] = - np.sign(m_obs[246]) * np.abs(m_obs[246])
        #   flip sign of pelvis roll and yaw angular velocities
        m_obs[249:251] = - np.sign(m_obs[249:251]) * np.abs(m_obs[249:251])
        #   save mirred to buffer
        super().store_transition(m_obs, m_actions, rewards, dones, m_next_obs, training)
        del m_actions, m_obs



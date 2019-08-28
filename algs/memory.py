from collections import namedtuple, deque
import numpy as np


Transition = namedtuple('Transition', ['obs', 'actions', 'rewards', 'dones', 'next_obs'])

class Memory:
    def __init__(self, max_size, observation_shape, action_shape, reward_scale, n_step_returns, dtype='float32'):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.reward_scale = reward_scale
        self.n_step_returns = n_step_returns
        self.dtype = dtype

        self.obs      = np.zeros((max_size, *self.observation_shape)).astype(dtype)
        self.actions  = np.zeros((max_size, *self.action_shape)).astype(dtype)
        self.rewards  = np.zeros((max_size, n_step_returns)).astype(dtype)
        self.dones    = np.zeros((max_size, 1)).astype(dtype)
        self.next_obs = np.zeros((max_size, *self.observation_shape)).astype(dtype)

        self.max_size = max_size
        self.pointer = 0
        self.size = 0
        self.current_obs = None

        self.obs_buffer      = deque(maxlen=n_step_returns)
        self.actions_buffer  = deque(maxlen=n_step_returns)
        self.rewards_buffer  = deque(maxlen=n_step_returns)
        self.dones_buffer    = deque(maxlen=n_step_returns)

    def store_transition(self, obs, actions, rewards, dones, next_obs, training=True):
        """ add transition to memory; overwrite oldest if memory is full """
        if not training:
            return

        # assert 2d arrays coming in
        obs, actions, rewards, dones, next_obs = np.atleast_2d(obs, actions, rewards, dones, next_obs)

        # store in buffers
        self.obs_buffer.append(obs)
        self.actions_buffer.append(actions)
        self.rewards_buffer.append(rewards)
        self.dones_buffer.append(dones)

        # start saving n-steps to memory when buffer is sufficient size
        if len(self.obs_buffer) == self.n_step_returns:
            # prep objects to save
            # NOTE  n-step q value is ∑gamme^i * r + gamma^N * q(x_N, a_N) * (1 - dones);
            #       => dones are at n-th step and we are excluding all inputs in buffers where done=1 appears during the n steps
            obs = self.obs_buffer[0]
            actions = self.actions_buffer[0]
            rewards = np.stack(self.rewards_buffer, axis=1).squeeze(2)  # (n_env, n_steps)

            # choose inputs which are not `done` within of the n-step window. e.g. exclude dones = [0,1,0] for n=3
            #   stack the dones_buffer to (n_envs, n_step_returns, 1), select steps 1: N-1 and exclude env idx if done=1 appears
            load_idxs = np.where(np.stack(self.dones_buffer, 2).sum(-1) != 1)[0]
            # save indices for the batched number of environments excluding inputs where done=1 within the n-step window
            B = len(load_idxs)
            save_idxs = np.arange(self.pointer, self.pointer + B) % self.max_size

            self.obs[save_idxs] = obs[load_idxs]
            self.actions[save_idxs] = actions[load_idxs]
            self.rewards[save_idxs] = rewards[load_idxs] * self.reward_scale
            self.dones[save_idxs] = dones[load_idxs]
            self.next_obs[save_idxs] = next_obs[load_idxs]

            # step memory pointer and size
            self.pointer = (self.pointer + B) % self.max_size
            self.size = min(self.size + B, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, batch_size)
        return Transition(*np.atleast_2d(self.obs[idxs], self.actions[idxs], self.rewards[idxs], self.dones[idxs], self.next_obs[idxs]))

    def initialize(self, env, n_prefill_steps=1000, training=True):
        if not training:
            return
        # prefill memory using uniform exploration
        if self.current_obs is None:
            self.current_obs = env.reset()

        for _ in range(n_prefill_steps):
            actions = np.random.uniform(-1, 1, (env.num_envs,) + self.action_shape)
            next_obs, r, done, _ = env.step(actions)
            self.store_transition(self.current_obs, actions, r, done, next_obs, training)
            self.current_obs = next_obs

        # NOTE -- clearing n-step buffers since env is reset after memory init
        self.clear_nstep_buffers()
        print('Memory initialized.')

    def clear_nstep_buffers(self):
        self.obs_buffer.clear()
        self.actions_buffer.clear()
        self.rewards_buffer.clear()
        self.dones_buffer.clear()

class SymmetricMemory(Memory):

    # NOTE --- assumes reward is left-right symmetric

    def store_transition(self, obs, actions, rewards, dones, next_obs, training=True):
        # add experience to buffer
        super().store_transition(obs, actions, rewards, dones, next_obs, training)

        # batch_size
        B = obs.shape[0]

        # add mirrored `experience` to buffer -- state and action mirred along left-right symmetry
        #   flip body obs - legts, roll/yaw, y vel and roll/yaw angular velocities -- for indexing from the observations dict to list see L2M2019Env.get_observation()

        #   1. flip actions for muscles on left and right leg
        m_actions = actions.copy()
        m_actions = np.flipud(m_actions.reshape(B,2,11)).reshape(B,-1)   # [0,1,2,3] -> [[0,1],[2,3]] -> [[2,3],[0,1]] -> [2,3,0,1]

        #   copy obs then edit in place
        m_obs = obs.copy()
        m_next_obs = next_obs.copy()

        # NOTE -- if state contains the v_tgt_field, body obs start at index 242
        body_obs = m_obs#[242:]
        body_next_obs = m_next_obs#[242:]

        pelvis_obs = body_obs[:,:9]
        pelvis_next_obs = body_next_obs[:,:9]
        leg_obs = body_obs[:,9:]
        leg_next_obs = body_next_obs[:,9:]

        #   2. flip legs
        leg_obs = np.flipud(leg_obs.reshape(B,2,-1)).flatten()
        leg_next_obs = np.flipud(leg_next_obs.reshape(B,2,-1)).flatten()

        #   3. flip sign of pelvis roll (right->left), y-velocity, roll, yaw
        pelvis_obs[:,[2,4,7,8]] *= -1
        pelvis_next_obs[:,[2,4,7,8]] *= -1

        #   save mirred to buffer
        super().store_transition(m_obs, m_actions, rewards, dones, m_next_obs, training)
        del m_actions, m_obs



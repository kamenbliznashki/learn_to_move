from collections import namedtuple, deque
import numpy as np


Transition = namedtuple('Transition', ['obs', 'actions', 'rewards', 'dones', 'next_obs'])

class Memory:
    def __init__(self, max_size, observation_shape, action_shape, n_step_returns, dtype='float32'):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
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
        """ add transition to memory; overwrite oldest if memory is full 
        N-step transition is (s_t, a_t, [r_t,...t_tpN], d_N, s_N)
        """
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
            rewards = np.hstack(self.rewards_buffer)  # (n_env, n_steps)

            # choose inputs which are not `done` within of the n-step window. e.g. exclude dones = [0,1,0] for n=3
            #   stack the dones_buffer to (n_envs, n_step_returns, 1), select steps 1: N-1 and exclude env idx if done=1 appears
            #   this is only relevant the first time we store a transition, since after transitions are stored on every step so dones will only occur at the end of the buffer
#            load_idxs = np.where(np.hstack(self.dones_buffer)[:,:-1].sum(-1) != 1)[0]
#            # save indices for the batched number of environments excluding inputs where done=1 within the n-step window
#            B = len(load_idxs)
            B = obs.shape[0]
            idxs = np.arange(self.pointer, self.pointer + B) % self.max_size

            # store (s_t, a_t, [r_t,...t_tpN], d_N, s_N)
            self.obs[idxs] = obs
            self.actions[idxs] = actions
            self.rewards[idxs] = rewards
            self.dones[idxs] = dones
            self.next_obs[idxs] = next_obs

            # step memory pointer and size
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
            actions = np.random.uniform(-1, 1, (env.num_envs,) + self.action_shape) if policy is None else policy.get_actions(self.current_obs)
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

class EnsembleMemory:
    def __init__(self, n_heads, *args, **kwargs):
        self.heads = [Memory(*args, **kwargs) for _ in range(n_heads)]

    def __getitem__(self, k):
        return self.heads[k]

    def store_transition(self, obs, actions, rewards, dones, next_obs, training=True):
        for head in self.heads:
            m = np.random.randint(2)  # double or nothing bootstrap
            if m == 1:
                head.store_transition(obs, actions, rewards, dones, next_obs, training)

    def initialize(self, env, n_prefill_steps, training=True, policy=None):
        for i, head in enumerate(self.heads):  # call store_transition in each Memory class (ie no bootstrap)
            head.initialize(env, n_prefill_steps, training, policy[i] if policy is not None else None)


class BootstrappedMemory(Memory):
    def store_transition(self, obs, actions, rewards, dones, next_obs, training=True):
        m = np.random.randint(2)  # double or nothing bootstrap
        if m == 1:
            super().store_transition(obs, actions, rewards, dones, next_obs, training)


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
        leg_obs = np.flipud(leg_obs.reshape(B,2,-1)).reshape(B,-1)
        leg_next_obs = np.flipud(leg_next_obs.reshape(B,2,-1)).reshape(B,-1)

        #   3. flip sign of pelvis roll (right->left), y-velocity, roll, yaw
        pelvis_obs[:,[2,4,7,8]] *= -1
        pelvis_next_obs[:,[2,4,7,8]] *= -1

        #   save mirrored to buffer -- NOTE need to add mirrored vtgt here
        m_obs = np.hstack([pelvis_obs, leg_obs])
        m_next_obs = np.hstack([pelvis_next_obs, leg_next_obs])
        super().store_transition(m_obs, m_actions, rewards, dones, m_next_obs, training)
        del m_actions, m_obs



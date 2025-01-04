# import gymnasium as gym
import gym
import numpy as np
from collections import deque
import cv2


class Atari_Env(gym.Wrapper):
    """
    We modify the Atari environment to accelerate the training with some tricks:
        Episode termination: Make end-of-life == end-of-episode, but only reset on true game over. Done by DeepMind for the DQN and co. since it helps value estimation.
        Frame skipping: Return only every `skip`-th frame.
        Observation resize: Warp frames from 210x160 to 84x84 as done in the Nature paper and later work.
        Frame Stacking: Stack k last frames. Returns lazy array, which is much more memory efficient.
    Args:
        env_id: The environment id of Atari, such as "Breakout-v5", "Pong-v5", etc.
        env_seed: The random seed to set up the environment.
        obs_type: This argument determines what observations are returned by the environment. Its values are:
                    ram: The 128 Bytes of RAM are returned
                    rgb: An RGB rendering of the game is returned
                    grayscale: A grayscale rendering is returned
        frame_skip: int or a tuple of two ints. This argument controls stochastic frame skipping, as described in the section on stochasticity.
        num_stack: int, the number of stacked frames if you use the frame stacking trick.
        image_size: This argument determines the size of observation image, default is [210, 160].
        noop_max: max times of noop action for env.reset().
    """

    def __init__(self, env_id, env_seed, render_mode, obs_type, frameskip):
        self.env = gym.make(env_id,
                            render_mode=render_mode,
                            obs_type=obs_type,
                            frameskip=frameskip
                            )
        #obs_type="grayscale",frameskip=4
        self.env.action_space.seed(seed=env_seed)
        self.env.unwrapped.reset(seed=env_seed)
        self.max_episode_steps = self.env._max_episode_steps
        super(Atari_Env, self).__init__(self.env)
        # self.env.seed(config.env_seed)
        self.num_stack = 4
        self.obs_type = "grayscale"
        self.frames = deque([], maxlen=self.num_stack)
        self.image_size = [84, 84]
        self.noop_max = 30
        self.lifes = self.env.unwrapped.ale.lives()
        self.was_real_done = True
        self.grayscale, self.rgb = False, False
        if self.obs_type == "rgb":
            self.rgb = True
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.image_size[0], self.image_size[1], 3 * self.num_stack), dtype=np.uint8)
        elif self.obs_type == "grayscale":
            self.grayscale = True
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.image_size[0], self.image_size[1], self.num_stack), dtype=np.uint8)
        else:  # ram type
            self.observation_space = self.env.observation_space
        # assert self.env.unwrapped.get_action_meanings()[0] == "NOOP"
        # assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
        # assert len(self.env.unwrapped.get_action_meanings()) >= 3
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self._render_mode = render_mode
        self._episode_step = 0

    def close(self):
        self.env.close()

    def render(self, render_mode):
        return self.env.unwrapped.render(render_mode)

    def reset(self):
        info = {}
        if self.was_real_done:
            self.env.unwrapped.reset()
            # Execute NoOp actions
            num_noops = np.random.randint(0, self.noop_max)
            for _ in range(num_noops):
                obs, _, done, _ = self.env.unwrapped.step(0)
                if done:
                    self.env.unwrapped.reset()
            # try to fire
            obs, _, done, _ = self.env.unwrapped.step(1)
            if done:
                obs = self.env.unwrapped.reset()
            # stack reset observations
            for _ in range(self.num_stack):
                self.frames.append(self.observation(obs))

            self._episode_step = 0
        else:
            obs, _, done, _ = self.env.unwrapped.step(0)
            for _ in range(self.num_stack):
                self.frames.append(self.observation(obs))

        self.lifes = self.env.unwrapped.ale.lives()
        self.was_real_done = False
        return self._get_obs(), info

    def step(self, actions):
        observation, reward, terminated, info = self.env.unwrapped.step(actions)
        self.frames.append(self.observation(observation))
        lives = self.env.unwrapped.ale.lives()
        # avoid environment bug
        if self._episode_step >= self.max_episode_steps:
            terminated = True
        self.was_real_done = terminated
        if (lives < self.lifes) and (lives > 0):
            terminated = True
        truncated = self.was_real_done
        self.lifes = lives
        self._episode_step += 1
        # return self._get_obs(), self.reward(reward), terminated, truncated, info
        return self._get_obs(), self.reward(reward), terminated, info

    def _get_obs(self):
        assert len(self.frames) == self.num_stack
        return LazyFrames(list(self.frames))

    def observation(self, frame):
        if self.grayscale:
            return np.expand_dims(cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA), -1)
        elif self.rgb:
            return cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
        else:
            return frame

    def reward(self, reward):
        return np.sign(reward)


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
    This object should only be converted to numpy array before being passed to the model.
    """

    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]


if __name__ == "__main__":
    import time
    env = Atari_Env("ALE/Breakout-v5", env_seed=1, render_mode='rgb_array',obs_type='grayscale',frameskip=4)
    for i in range(10000):
        env.reset()
        done = False
        while not done:
            actions = env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            time.sleep(0.01)
            env.render("human")
            if done:
                print("Rewards: ", reward)

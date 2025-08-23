import gym
from gym import spaces
from ple.games.pong import Pong as PLEPong
from ple import PLE
import numpy as np


class Pong(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.name = "Pong"
        self.max_steps = config.get("max_steps", 1000)
        players_speed_ratio = config.get("players_speed_ratio", 0.5)
        cpu_speed_ratio = config.get("cpu_speed_ratio", 0.5)
        max_score = config.get("max_score", 1000)

        # Initialize PLE Pong
        self.game = PLEPong(cpu_speed_ratio=cpu_speed_ratio, players_speed_ratio = players_speed_ratio,  MAX_SCORE=max_score)
        self.env = PLE(
            self.game,
            fps=30,
            display_screen=False,
            force_fps=False
        )
        self.env.init()

        # Action space (discrete)
        self.action_set = self.env.getActionSet()
        self.action_space = spaces.Discrete(len(self.action_set))

        # Get feature keys and sizes
        state_dict = self.env.getGameState()
        self.state_keys = list(state_dict.keys())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.state_keys),),
            dtype=np.float32
        )

        # Episode bookkeeping
        self.episode_step = 0
        self.done = False

    def _get_obs(self):
        state = self.env.getGameState()
        return np.array([state[k] for k in self.state_keys], dtype=np.float32)

    def step(self, action):
        reward = self.env.act(self.action_set[action])
        obs = self._get_obs()

        # Reward shaping
        if reward == -6:
            reward = 0
        elif reward == 6:
            reward = 1

        self.episode_step += 1
        self.done = self.env.game_over()
        truncated = self.is_episode_done()

        if truncated:
            reward = 0.5

        return obs, reward, self.done, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset_game()
        self.episode_step = 0
        self.done = False
        obs = self._get_obs()
        return obs, {}

    def is_episode_done(self):
        return self.episode_step >= self.max_steps

    def get_frame(self, grayscale=True, rotate=True):
        """Return the current screen frame from the environment."""
        frame = self.env.getScreenRGB()

        # Rotate so the game is horizontal like classic Pong
        if rotate:
            frame = np.rot90(frame, k=3)   # 90 degrees counter-clockwise

        if grayscale:
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        return frame

    def render(self, mode="human"):
        if mode == "rgb_array":
            return self.env.getScreenRGB()
        elif mode == "human":
            self.env.display_screen = True

    def close(self):
        self.env.close()

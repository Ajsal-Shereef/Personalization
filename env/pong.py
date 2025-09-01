import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ple import PLE
from ple.games.pong import Pong as PongEnv


class Pong(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: dict, render_mode=None):
        super().__init__()

        self.config = config
        self.render_mode = render_mode
        self.name = "Pong"

        # Construct Pong with proper defaults, allow overrides from config
        game = PongEnv(
            MAX_SCORE=config.get("max_score", 1),
            players_speed_ratio=config.get("players_speed_ratio", 0.4),
            cpu_speed_ratio=config.get("cpu_speed_ratio", 0.6),
            ball_speed_ratio=config.get("ball_speed_ratio", 0.75),
        )

        self.p = PLE(game, fps=config.get("fps", 30), display_screen=(render_mode == "human"))
        self.p.init()
        self.game = game

        # Config values
        self.max_steps = config.get("max_steps", 1000)
        self.current_step = 0

        # Action space: UP / DOWN / NOOP
        self.actions = self.p.getActionSet()
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space: 7 continuous features
        low = np.array([0, -np.inf, 0, 0, 0, -np.inf, -np.inf], dtype=np.float32)
        high = np.array([
            game.height,   # player_y
            np.inf,        # player_velocity
            game.height,   # cpu_y
            game.width,    # ball_x
            game.height,   # ball_y
            np.inf,        # ball_velocity_x
            np.inf         # ball_velocity_y
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p.reset_game()
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        reward = self.p.act(self.actions[action])
        if reward == -6:
            reward = -1
        elif reward == +6:
            reward = +1
        terminated = self.p.game_over()
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        if truncated:
            reward = 0.5

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def get_frame(self):
        """Return the current rendered frame as an RGB array rotated 90° clockwise."""
        frame = self.p.getScreenRGB()
        return np.rot90(frame, k=-1)
    
    def get_legal_actions(self):
        return list(range(len(self.p.getActionSet())))

    def close(self):
        self.p.quit()

    def _get_obs(self):
        state = self.p.getGameState()
        return np.array([
            state["player_y"],
            state["player_velocity"],
            state["cpu_y"],
            state["ball_x"],
            state["ball_y"],
            state["ball_velocity_x"],
            state["ball_velocity_y"],
        ], dtype=np.float32)

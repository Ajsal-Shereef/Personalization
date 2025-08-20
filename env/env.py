import sys
import gymnasium
import numpy as np
sys.modules["gym"] = gymnasium
import highway_env
from gymnasium.core import ObservationWrapper

class Highway(ObservationWrapper):
    "This class creates the Highway environment with a specified number of lanes and maximum steps."
    def __init__(self, config):
        self.env = gymnasium.make('highway-fast-v0')
        self.name = "Highway"
        env_config = {
                    "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "features_range": {
                                        "x": [-100, 100],
                                        "y": [-100, 100],
                                        "vx": [-20, 20],
                                        "vy": [-20, 20]
                                    },
                    "absolute": False,
                    "order": "sorted",
                    "show_trajectories": True,
                    }
                }
        self.env.unwrapped.configure(env_config)
        self.env.unwrapped.config["duration"] = config["max_steps"]
        self.env.unwrapped.config["right_lane_reward"] = 0
        self.env.unwrapped.config['lanes_count'] = config["lane_count"]
        super().__init__(self.env)
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(np.prod(self.env.observation_space.shape) + 1,), dtype=np.float32)
        
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        lane = self.env.unwrapped.vehicle.lane_index[-1]
        next_state = np.concatenate((next_state.flatten(order = 'C'), np.array([lane])))
        return next_state, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        lane = self.env.unwrapped.vehicle.lane_index[-1]
        state = np.concatenate((state.flatten(order = 'C'), np.array([lane])))
        return state, info
        
    
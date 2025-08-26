import sys
import gymnasium
import numpy as np
sys.modules["gym"] = gymnasium
import highway_env
from gymnasium.core import ObservationWrapper

class Highway(ObservationWrapper):
    "This class creates the Highway environment with a specified number of lanes and maximum steps."
    def __init__(self, config):
        self.env = gymnasium.make('highway-fast-v0', render_mode="rgb_array")
        self.config = config
        self.name = "Highway"
        env_config = {
                    "observation": {
                    "type": "Kinematics",
                    "vehicles_count":config["vehicles_count"],
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "collision_reward": -1,
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
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(config["vehicles_count"]*5 + 1,), dtype=np.float32)
        
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
    
    def get_frame(self):
        return self.env.render()
    
    def get_legal_actions(self):
        vehicle = self.env.unwrapped.vehicle
        lane = vehicle.lane_index[-1]
        if lane == 0:
            return [1,2,3,4]
        elif lane == self.config["lane_count"]-1:
            return [0,1,3,4]
        else:
            return [0,1,2,3,4]
    
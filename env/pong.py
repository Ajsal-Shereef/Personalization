import sys
import gymnasium

class Pong(gymnasium.Wrapper):
    def __init__(self, config):
        self.name = "Pong"
        self.env = gymnasium.make('Pong-PLE-v0', MAX_SCORE = config["max_steps"], cpu_speed_ratio=config["cpu_speed_ratio"], players_speed_ratio=config["players_speed_ratio"])
        super().__init__(self.env)

    def step(self, action):
        observation, reward, self.done, info = self.env.step(action)
        if reward == -6:
            reward = 0
        elif reward == 6:
            reward = 1
        self.episode_step += 1
        truncated = self.is_episode_done()
        if truncated:
            reward = 0.5
        return observation, reward, self.done, truncated, info
    
    def reset(self):
        self.episode_step = 0
        reset = self.env.reset()
        return reset, {}
    
    def is_episode_done(self):
        max_step_criteria = self.episode_step == self.max_steps
        return max_step_criteria
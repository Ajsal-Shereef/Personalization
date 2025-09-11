class HumanOracleHighway():
    """This class is a simmulated human which gives the safety feedback"""
    def __init__(self, env, mode, speed_feedback):
        self.env = env
        self.mode = mode
        self.speed_feedback = speed_feedback
        self.reset_arrays()
        self.reset_episode_count()

    def get_episodic_feedback(self):
        if self.mode == 'preference':
            return self.episode_right_lane 
        if self.mode == 'avoid':
            return -self.episode_left_lane 
        if self.mode == 'both':
            return -self.episode_left_lane + self.episode_right_lane 
        
    def get_human_feedback(self, observation = None):
        vehicle = self.env.unwrapped.vehicle
        lane = vehicle.lane_index[-1]
        value = 0
        if lane == 2 and (self.mode == "preference" or self.mode == "both"):
            value = 1
        if lane == 0 and (self.mode == "avoid" or self.mode == "both"):
            value = -1
        if self.speed_feedback:
            if vehicle.speed > 25:
                value -= 1
        return value
    
    def get_statistics(self, episode_step):
        if self.mode == 'preference':
            return [self.episode_right_lane/episode_step, self.episode_hitting, self.speed_violations/episode_step]
        if self.mode == 'avoid':
            return [self.episode_left_lane/episode_step, self.episode_hitting, self.speed_violations/episode_step]
        if self.mode == 'both':
            return [self.episode_right_lane/episode_step, self.episode_left_lane/episode_step, self.episode_hitting, self.speed_violations/episode_step]
        
    def get_statistics_description(self):
        if self.mode == 'preference':
            return ["Right lane %", "Hitting", "Speed violations %"]
        if self.mode == 'avoid':
            return ["Left lane %", "Hitting", "Speed violations %"]
        if self.mode == 'both':
            return ["Right lane %", "Left lane %", "Hitting", "Speed violations %"]
        
    def update_counts(self, info):
        vehicle = self.env.unwrapped.vehicle
        lane = vehicle.lane_index[-1]
        if lane == 2:
            self.episode_right_lane += 1
            self.cummulative_right_lane += 1
        if lane == 0:
            self.episode_left_lane += 1
            self.cummulative_left_lane += 1
        if info["crashed"]:
            self.episode_hitting += 1
        if self.speed_feedback:
            if vehicle.speed > 25:
                 self.speed_violations += 1
        
    def reset_episode_count(self):
        self.episode_right_lane = 0
        self.episode_left_lane = 0
        self.episode_hitting = 0
        self.speed_violations = 0
    
    def reset_arrays(self):
        self.cummulative_right_lane = 0
        self.cummulative_left_lane = 0
        
class HumanOraclePong():
    def __init__(self, env, mode):
        self.mode = mode
        self.env = env
        self.reset_arrays()
        self.reset_episode_count()

    def get_episodic_feedback(self):
        if self.mode == 'preference':
            return self.episode_prefered_region
        if self.mode == 'avoid':
            return -self.episode_avoid_region 
        if self.mode == 'both':
            return -self.episode_avoid_region + self.episode_prefered_region
        
    def get_human_feedback(self, observation):
        if self.mode == 'avoid' or self.mode == 'both':
            if 0<=observation[0]<=7 or 41<=observation[0]<=48:
                return -1
        elif self.mode == 'preference' or self.mode == 'both':
            if 16<observation[0]<=31:
                return 1
        return 0
    
    def get_statistics(self, episode_step):
        if self.mode == 'preference':
            return [self.episode_prefered_region/episode_step]
        if self.mode == 'avoid':
            return [self.episode_avoid_region/episode_step]
        if self.mode == 'both':
            return [self.episode_prefered_region/episode_step, self.episode_avoid_region/episode_step]
        
    def get_statistics_description(self):
        if self.mode == 'preference':
            return ["Prefered region %"]
        if self.mode == 'avoid':
            return ["Avoid region %"]
        if self.mode == 'both':
            return ["Prefered region %", "Avoid region %"]
        
    def update_counts(self, observation):
        if 20<observation[0]<30:
            self.episode_prefered_region += 1
            self.cummulative_prefered_region += 1
        elif (0<=observation[0]<10 or 38<observation[0]<=48):
            self.episode_avoid_region += 1
            self.cummulative_avoid_region += 1
        
    def reset_episode_count(self):
        self.episode_prefered_region = 0
        self.episode_avoid_region = 0
    
    def reset_arrays(self):
        self.cummulative_avoid_region = 0
        self.cummulative_prefered_region = 0
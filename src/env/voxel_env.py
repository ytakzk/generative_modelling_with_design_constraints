from src.env.base_env import BaseEnv
from src.common import Vector
import numpy as np
import math

class VoxelEnv(BaseEnv):
    
    def __init__(self, bound, targets, window_width=3, max_failure=5, is_cnn=False, vertical_steps=[1], horizontal_steps=[1], possible_actions=6):
        
        super().__init__(bound=bound, targets=targets, window_width=window_width, max_failure=max_failure, is_cnn=is_cnn, action_space=6)
        self.xx = []
        self.yy = []
        self.zz = []
        self.num = 0

        # TODO: Voxel environment is not compatible with multiple steps now
        vertical_steps = [1]
        horizontal_steps = [1]

        self.possible_actions = possible_actions

    def _get_vector(self, action):

        px = (action[0] + 1.0) * 0.5 # 0 to 1
        py = (action[1] + 1.0) * 0.5 # 0 to 1
        pz = (action[2] + 1.0) * 0.5 # 0 to 1
        nx = (action[3] + 1.0) * 0.5 # 0 to 1
        ny = (action[4] + 1.0) * 0.5 # 0 to 1
        nz = (action[5] + 1.0) * 0.5 # 0 to 1

        xx = px - nx
        yy = py - ny
        zz = pz - nz

        if self.possible_actions == 6:

            ax = abs(xx)
            ay = abs(yy)
            az = abs(zz)

            if ax >= max([ay, az]):

                return Vector(1, 0, 0) if xx >= 0 else Vector(-1, 0, 0)

            if ay >= max([ax, az]):

                return Vector(0, 1, 0) if yy >= 0 else Vector(0, -1, 0)

            if az >= max([ay, ax]):

                return Vector(0, 0, 1) if zz >= 0 else Vector(0, 0, -1)

            print(xx, yy, zz)
        
        elif self.possible_actions == 26:

            def round_to_voxel(threshold):

                if xx < -threshold:
                    x = -1
                elif xx > threshold:
                    x = 1
                else:
                    x = 0

                if yy < -threshold:
                    y = -1
                elif yy > threshold:
                    y = 1
                else:
                    y = 0

                if zz < -threshold:
                    z = -1
                elif zz > threshold:
                    z = 1
                else:
                    z = 0
                
                return x, y, z
            
            i = 1
            while i <= 2:

                threshold = 1.0 / (3**i)
                x, y, z = round_to_voxel(threshold)
                if x != 0 or y != 0 or z != 0:
                    break
                i += 1

            while x == y == z == 0:

                x = np.random.choice([-1, 0, 1])
                y = np.random.choice([-1, 0, 1])
                z = np.random.choice([-1, 0, 1])
            
            return Vector(x, y, z)

        elif self.possible_actions == 27:

            if xx < -1.0 / 3:

                x = -1.0

            elif xx > 1.0 / 3:

                x = 1.0
            
            else: 

                x = 0

            if yy < -1.0 / 3:

                y = -1.0

            elif yy > 1.0 / 3:

                y = 1.0
            
            else: 

                y = 0

            if zz < -1.0 / 3:

                z = -1.0

            elif zz > 1.0 / 3:

                z = 1.0
            
            else: 

                z = 0
        
            return Vector(x, y, z)

        else:

            raise ValueError('possible_actions must be 6, 26 or 27.')

    def _get_reward(self, next_positions, is_outside, remained):
        
        if is_outside:
            
            return -1

        if remained:
            return -1

        worst_reward = 1

        for position in next_positions:
        
            if self._get_binary_value(self.target.grid, position) == 1 and self._get_binary_value(self.grid, position) == 0:
                
                reward = 1
            
            elif self._get_binary_value(self.target.grid, position) == 1 and self._get_binary_value(self.grid, position) == 1:
                
                reward = -0.5
            
            else:

                reward = -1

            if worst_reward > reward:
                worst_reward = reward
                
        return worst_reward
from src.env.base_env import BaseEnv
from src.common import Vector
import numpy as np
import math

class LegoEnv(BaseEnv):
    
    def __init__(self, bound, targets, window_width=3, vertical_steps=[1,2], horizontal_steps=[1, 2], is_cnn=False):
        
        super().__init__(bound=bound, targets=targets, window_width=window_width, max_failure=0, is_cnn=is_cnn, action_space=6, horizontal_steps=horizontal_steps, vertical_steps=vertical_steps)
        
        self.unit_length = 1 / math.sqrt(len(self.horizontal_steps) * 4 + 1)

    def _get_vector(self, action):

        # x = action[0]
        # y = action[1]
        # z = action[2]

        px = (action[0] + 1.0) * 0.5 # 0 to 1
        py = (action[1] + 1.0) * 0.5 # 0 to 1
        pz = (action[2] + 1.0) * 0.5 # 0 to 1
        nx = (action[3] + 1.0) * 0.5 # 0 to 1
        ny = (action[4] + 1.0) * 0.5 # 0 to 1
        nz = (action[5] + 1.0) * 0.5 # 0 to 1
        
        x = px - nx
        y = py - ny
        z = pz - nz

        if abs(x) <= self.unit_length and abs(y) <= self.unit_length:

            x = 0
            y = 0

        elif y > x and y > -x:

            x = 0
            y = self._get_horizontal_value(y)

        elif y > x and y <= -x:

            x = -self._get_horizontal_value(x)
            y = 0

        elif y <= x and y > -x:

            x = self._get_horizontal_value(x)
            y = 0

        elif y <= x and y <= -x:

            x = 0
            y = -self._get_horizontal_value(y)

        else:
            raise ValueError('NA')
        
        num   = len(self.vertical_steps)
        index = int(abs(z) * num)
        if index == num:
            index = num - 1

        if z <= 0:
            z = -self.vertical_steps[index]
        else:
            z = self.vertical_steps[index]
        
        return Vector(x, y, z)

    def _get_reward(self, next_positions, is_outside, remained):

        scale = 1
        
        if is_outside:
            
            return -1

        if remained:
            return -1

        worst_reward = 1

        for position in next_positions:
        
            if self._get_binary_value(self.target.grid, position) == 1 and self._get_binary_value(self.grid, position) == 0:
                
                reward = scale
            
            else:

                reward = -1

            if reward < worst_reward:
                worst_reward = reward
                
        return worst_reward

    def _get_horizontal_value(self, v):

        v = abs(v)

        if v <= self.unit_length:
            return self.horizontal_steps[0]

        if v >= 1:
            return self.horizontal_steps[-1]

        num   = len(self.horizontal_steps)
        step  = (1.0 - self.unit_length) / num
        index = int(float(v - self.unit_length) / step)

        return self.horizontal_steps[index]

    def get_next_positions(self, position):

        if self.prev_pos:

            positions = []

            def generate_diff(current_value, prev_value):

                d = current_value - prev_value
                if d > 0:
                    d = 1
                elif d < 0:
                    d = -1
                return d

            dx = generate_diff(position.x, self.prev_pos.x)
            dy = generate_diff(position.y, self.prev_pos.y)
            dz = generate_diff(position.z, self.prev_pos.z)

            x = self.prev_pos.x
            y = self.prev_pos.y
            z = self.prev_pos.z
            j = 0
            
            while True:
                z += dz
                p = Vector(x, y, z)
                positions.append(p)

                if z == position.z:
                    break
                
                j += 1  
                if j > 100:
                    raise ValueError('misconfiguration')

            is_horizontal = False

            if dx != 0 or dy != 0:
                dz = 0
                is_horizontal = True

            j = 0
            while is_horizontal:

                x = positions[-1].x + dx
                y = positions[-1].y + dy
                z = positions[-1].z

                p = Vector(x, y, z)
                positions.append(p)

                if p == position:
                    break

                j += 1

                if j > 100:
                    raise ValueError('misconfiguration')
        
        else:
            positions = [position]

        return positions
import gym
import numpy as np
import gym.spaces
from src.common import Vector
import math
import itertools
from enum import Enum

class SamplingDistrubution(Enum):
    RANDOM = 1
    EDGE   = 2
    BOTTOM = 3

class BaseEnv(gym.Env):
    
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, bound, targets, window_width=3, max_failure=5, is_cnn=False, action_space=6, vertical_steps=[1], horizontal_steps=[1], sampling_distrubution=SamplingDistrubution.EDGE):
        
        super().__init__()
        
        if window_width % 2 == 0:
            
            raise ValueError('window_width must be odd number')
        
        self.bound          = bound
        self.targets        = targets
        self.target_index   = 0
        self.max_failure    = max_failure
        self.grid           = np.zeros((self.bound.x_len, self.bound.y_len, self.bound.z_len))
        self.window_width   = window_width
        self.is_cnn         = is_cnn
        self.sampling_distrubution = sampling_distrubution
        
        self.vertical_steps   = vertical_steps
        self.horizontal_steps = horizontal_steps

        if isinstance(self.horizontal_steps, int):
            self.vertical_steps = [self.horizontal_steps]

        if isinstance(self.vertical_steps, int):
            self.vertical_steps = [self.vertical_steps]

        observation_space = (window_width, window_width, window_width, 3) if is_cnn else (window_width, window_width, window_width)
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_space,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0 if is_cnn else -1,
            high=1,
            shape=observation_space,
            dtype=np.float
        )
        
        self.is_train = True
        self.map = {}

        self.test_possible_steps = 999

        self.reset()

    def reset(self):
        
        if self.is_train:

            indexes = np.arange(0, len(self.targets))

            self.target_index = np.random.choice(indexes)
            self.target = self.targets[self.target_index].augmented
        
        else:

            self.target = self.targets[self.target_index]
        
        self.position    = self.sample()

        self.grid        = np.zeros_like(self.grid)
        self.done        = False
        self.steps       = 0
        self.path        = []
        self.prev_action = -1
        self.prev_pos    = None
        self.failed      = 0
        self._pass()
        
        return self.state

    def step(self, action):
        
        d = self._get_vector(action)

        s = '%d%d%d' % (d.x, d.y, d.z)
        if not s in self.map:
            self.map[s] = 1
            print(self.map)
        else:
            self.map[s] += 1

        remained = True if d.x == 0 and d.y == 0 and d.z == 0 else False
            
        next_last_position = self.position + d
        next_positions = self.get_next_positions(next_last_position)
        
        is_outside = False

        for p in next_positions:
            is_outside = self.bound.is_inside(p) == False
        
        if not is_outside:
            self.position = next_last_position

        reward = self._get_reward(next_positions, is_outside, remained)

        if not is_outside and not remained:
            self._pass(next_positions)

        done = False 

        if reward == -1:

            # oustide
            done = True

        elif reward <= 0:
            
            # self-intersection or remaining at the same position
            self.failed += 1
            if self.failed >= self.max_failure:

                done = True

        if not self.is_train and len(self.path) > self.test_possible_steps:
                done = True
        
        self.prev_action = d
        # print('v', d, 'reward', reward, 'action', action, 'done', done)

        return self.state, reward, done, {}

    def _get_vector(self, action):

        raise NotImplementedError('not implemented')

    def render(self, mode='human', close=False):
        
        return ''
    
    def close(self):
        pass

    def _seed(self, seed=None):
        pass

    def _get_reward(self, next_positions, is_outside, remained):
        
        raise NotImplementedError('not implemented')
        
    def _pass(self, positions=None):

        if not positions:
            positions = [self.position]

        self.path.append(positions)

        for p in positions:
            self.grid[p.x][p.y][p.z] = 1

        self.prev_pos = positions[-1]


    def _get_binary_value(self, grid, position):

        if not position:
            position = self.position
            
        if len(grid.shape) == 2:
            
            return grid[position.x][position.y]
        
        else:
            
            return grid[position.x][position.y][position.z]
    
    @property
    def state(self):
        
        w = int(self.window_width / 2)
        
        if self.is_cnn:
            window = np.zeros((self.window_width, self.window_width, self.window_width, 3))
        else:
            window = np.zeros((self.window_width, self.window_width, self.window_width))

        vals = np.arange(-w, w+1, 1)
        
        for i, dx in enumerate(vals):
            
            for j, dy in enumerate(vals):

                for k, dz in enumerate(vals):

                    if dx == 0 and dy == 0 and dz == 0:
                        
                        if self.is_cnn:
                            window[i, j, k, :] = 1
                        else:
                            window[i, j, k] = 0

                    else:
                        
                        x = self.position.x + dx
                        y = self.position.y + dy
                        z = self.position.z + dz

                        if 0 <= x < self.bound.x_len and 0 <= y < self.bound.y_len and 0 <= z < self.bound.z_len:

                            is_target_inside     = self.target.grid[x, y, z] == 1
                            is_self_intersection = self.grid[x, y, z] == 1

                            if is_target_inside and not is_self_intersection:

                                if self.is_cnn:
                                    window[i, j, k, 0] = 1
                                else:
                                    window[i, j, k] = 1
    
                            elif is_target_inside and is_self_intersection:

                                if self.is_cnn:
                                    window[i, j, k, 1] = 1
                                else:
                                    window[i, j, k] = -0.5
                                    
                            else:
                                
                                if self.is_cnn:
                                    window[i, j, k, 2] = 1
                                else:
                                    window[i, j, k] = -1
                        else:
 
                            if self.is_cnn:
                                window[i, j, k, 2] = 1
                            else:
                                window[i, j, k] = -1
        return window
    
    def sample(self):

        if self.sampling_distrubution == SamplingDistrubution.EDGE:

            candidates = np.argwhere(self.target.grid == 1)
            
            while True:

                i = np.random.choice(list(range(len(candidates))))

                x = candidates[i][0]
                y = candidates[i][1]
                z = candidates[i][2]
                
                has_outside = False
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            
                            xx = x + dx
                            yy = y + dy
                            zz = z + dz

                            if 0 <= xx < self.bound.x_len and 0 <= yy < self.bound.y_len and 0 <= zz < self.bound.z_len:
                                
                                if self.target.grid[xx, yy, zz] == 1:
                                    has_outside = True
                            
                            else:
                                
                                has_outside = True
                
                if has_outside:
                    break
                    
            return Vector(x, y, z)

        elif self.sampling_distrubution == SamplingDistrubution.RANDOM:

            candidates = np.argwhere(self.target.grid == 1)
            
            i = np.random.choice(list(range(len(candidates))))

            x = candidates[i][0]
            y = candidates[i][1]
            z = candidates[i][2]

            return Vector(x, y, z)
        
        else:

            candidates = np.argwhere(self.target.grid[:,:, 0] == 1)
            i = np.random.choice(list(range(len(candidates))))
            x = candidates[i][0]
            y = candidates[i][1]
            return Vector(x, y, 0)

    def get_next_positions(self, position):
        return [position]

    @property
    def flattened_path(self):
        return list(itertools.chain(*self.path))

    @property
    def path_dic(self):

        dic = {}

        is_1d = isinstance(self.path[0], Vector)

        if is_1d:
            path = [self.path]

        for i, positions in enumerate(self.path):
            
            key = str(i)

            dic[key] = []

            for p in positions:
                dic[key].append({"x": str(p.x), "y": str(p.y), "z": str(p.z)})

        return dic
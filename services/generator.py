from src.env.lego_env import LegoEnv
from src.env.voxel_env import VoxelEnv
from src.data_loader import load_dummy, Target
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from src.policy import LnCnn3dPolicy, cnn_3d
from src.common import Bound
from flask import Flask, request, jsonify
import math
from collections import OrderedDict
from src.common import Vector
from src.env.base_env import SamplingDistrubution

RESOLUTION = 16

WINDOW_WIDTH  = 5
IS_CNN        = False
MLP_LAYERS    = [256, 128]

targets, _ = load_dummy(resolution=RESOLUTION, num=1)

bound = Bound(x_len=RESOLUTION, y_len=RESOLUTION, z_len=RESOLUTION)
lego_env  = LegoEnv(bound=bound, targets=targets, window_width=WINDOW_WIDTH, horizontal_steps=[1, 2], vertical_steps=[1])
voxel_env = VoxelEnv(bound=bound, targets=targets, window_width=WINDOW_WIDTH, possible_actions=26)

class CustomMlpPolicy(MlpPolicy):
                                                                                                                                                               
    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(CustomMlpPolicy, self).__init__(sess, ob_space, ac_space, layers=MLP_LAYERS, **_kwargs)

lego_model  = TD3.load('./models/lego.pkl', policy=CustomMlpPolicy)
voxel_model = TD3.load('./models/voxel_26.pkl', policy=CustomMlpPolicy)

def generate(voxels, model_type='lego'):

    if model_type == 'lego':

        model = lego_model
        env   = lego_env
        max_failure = 0
        iteration = 1000
    
    elif model_type == 'voxel':

        model = voxel_model
        env   = voxel_env
        max_failure = 50
        iteration = 1200

    else:

        return None

    target = Target(grid=voxels)
    env.targets = np.array([target])

    env.target_index = 0
    env.is_train     = False
    env.max_failure  = max_failure
    env.sampling_distrubution = SamplingDistrubution.BOTTOM
    env.possible_steps = 2000

    obs  = env.reset()

    best_path   = None
    best_length = -1

    for i in range(iteration):
        
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if done:

            if len(env.path_dic) > best_length:
                best_length = len(env.path_dic)
                best_path = env.path_dic
            obs  = env.reset()
            continue

    return best_path
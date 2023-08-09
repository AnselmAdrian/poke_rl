import asyncio
import torch.nn as nn

from datetime import datetime
import pickle as pkl
import pandas as pd
import numpy as np
import pickle as pkl
import plotly.express as px
import optuna
import os
from poke_env.environment.battle import Battle
from poke_env.player import RandomPlayer
from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration
from gym.utils.env_checker import check_env
from stable_baselines3 import PPO, DQN, A2C, HerReplayBuffer 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces
import time

from common import SimpleRLEnv, RLEnv
from common import MaxDamagePlayer, RLPlayer, TestRLPlayer
from common import evaluate_player
from model import CustomMLP

# Global Variables
my_server_config= ServerConfiguration(
    "localhost:8000",
    "authentication-endpoint.com/action.php?"
)

opponent = RandomPlayer(battle_format="gen8randombattle")
train_env = RLEnv(battle_format="gen8randombattle", opponent = opponent, start_challenging=True)


#######################################################
# Check the environment object
#######################################################

opponent = RandomPlayer(battle_format="gen8randombattle")
#train_env = SimpleRLEnv(battle_format="gen8randombattle", opponent = opponent, start_challenging=True)
train_env = RLEnv(battle_format="gen8randombattle", opponent = opponent, start_challenging=True)
try:
    print('Checking Environment')
    check_env(train_env)
    print('PASSED!')
except Exception as e:
    print(e)

#######################################################
# Functions
#######################################################
def get_model(model_type:str):
    if model_type == 'PPO':
        return PPO
    if model_type == 'A2C':
        return A2C
    if model_type == 'DQN':
        return DQN
    return None

def generate_hyperparameters(trial, params):
    """ Read parameters from a dictionary TODO"""
    return None

def objective(trial: optuna.trial):
    n_battles = 100
    batch_size = 30
    # Get hyperparameters
    kwargs = dict(
                    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-3, log = True),
                    gamma = trial.suggest_float('gamma', 0.1, 1.0),
                    # gae_lambda = trial.suggest_float('gae_lambda', 0.1, 1.0),
                    # normalize_advantage = trial.suggest_categorical('normalize_advantage', [True, False]),
                    # ent_coef = trial.suggest_float('ent_coef', 0.0000001, 1.0, log = True),
                    # vf_coef = trial.suggest_float('vf_coef', 0.0000001, 1.0, log = True),
                    )
    
    policy_kwargs = dict(activation_fn=nn.ReLU,
                            # net_arch=dict(pi=[32, 32], vf=[32, 32]),
                            features_extractor_class=CustomMLP,
                            features_extractor_kwargs=dict(features_dim=128)
                    )

    # model = PPO('MlpPolicy', env = train_env, verbose = 0, batch_size = batch_size, 
    #             policy_kwargs = policy_kwargs, **kwargs)
    model = DQN('MlpPolicy', env = train_env, verbose = 0, batch_size = batch_size, 
                buffer_size = 100,
                policy_kwargs = policy_kwargs,
                # replay_buffer_class=DictReplayBuffer,
                # replay_buffer_kwargs=dict(
                #                             n_sampled_goal=4,
                #                             goal_selection_strategy="future",
                #                             ),
                **kwargs)
    try:
        model.learn(200)
    except Exception as e:
        raise optuna.TrialPruned()
    
    rl_player_generator = lambda :TestRLPlayer(model = model,
                                        battle_format="gen8randombattle",
                                    )
    opponent_generator = lambda : RandomPlayer(battle_format="gen8randombattle")
    score = evaluate_player(rl_player_generator, opponent_generator, n_battles = n_battles, verbose = 0)
    return score


#######################################################
# Hyperparameter Tuning
#######################################################

model_type = 'DQN'
version = 1
model_name = f'pokemon_{model_type}{version}'
date_now = datetime.now().strftime('%Y%m%d_%H%M%S')
study_filename = f'study_{date_now}.pkl'
folder = f'data/06_model/{model_name}'

if not os.path.exists(folder):
    os.makedirs(folder)

filename = os.path.join(folder, study_filename)

print(F'CREATING NEW STUDY')
study = optuna.create_study(study_name = model_name,
                            direction="maximize",
                            # pruner=optuna.pruners.MedianPruner(
                            #             n_startup_trials=5, n_warmup_steps=30, interval_steps=10
                            # ),
                            )

print(F'Optimizing')
try: 
    study.optimize(objective, n_trials=1500)
except Exception as e:
    print(e)
except KeyboardInterrupt:
    pass
with open(filename, 'wb') as file:
    print(F'SAVING {filename}')
    pkl.dump(study, file)
import yaml
import asyncio
from ipykernel.eventloops import register_integration
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
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces
import time

from common import SimpleRLEnv, RLEnv
from common import MaxDamagePlayer, RLPlayer, TestRLPlayer
from common import evaluate_player

# Global Variables
my_server_config= ServerConfiguration(
    "localhost:8000",
    "authentication-endpoint.com/action.php?"
)

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
# Baseline Model
#######################################################

# Create stable baseline model
# model = PPO('MlpPolicy', env = train_env, verbose = 1, batch_size = 30)
# try:
#     model.learn(1)
# except Exception as e:
#     print(e)
# except KeyboardInterrupt:
#     print('Stopping training early')

# eval_env = TestRLEnv(
#     battle_format="gen8randombattle", opponent=opponent, start_challenging=True
# )
# rl_player_generator = lambda :TestRLPlayer(model = model,
#                                     battle_format="gen8randombattle",
#                                 )
# opponent_generator = lambda : RandomPlayer(battle_format="gen8randombattle")

# print('Baseline Model Performance')
# score = evaluate_player(rl_player_generator, opponent_generator, n_battles = 10, verbose = 1)
# print(score)

# eval_env = TestRLEnv(battle_format="gen8randombattle", opponent = opponent, start_challenging=True)
# score2 = evaluate_policy(model, eval_env)
# print(score2, eval_env)
#######################################################
# Hyperparameter Tuning
#######################################################
class CustomMLP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        in_features = observation_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.network(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.network(observations))

opponent = RandomPlayer(battle_format="gen8randombattle")
train_env = RLEnv(battle_format="gen8randombattle", opponent = opponent, start_challenging=True)

def objective(trial: optuna.trial):
    n_battles = 100
    batch_size = 30
    # Get hyperparameters
    kwargs = dict(
                    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-3, log = True),
                    gamma = trial.suggest_float('gamma', 0.1, 1.0),
                    gae_lambda = trial.suggest_float('gae_lambda', 0.1, 1.0),
                    normalize_advantage = trial.suggest_categorical('normalize_advantage', [True, False]),
                    ent_coef = trial.suggest_float('ent_coef', 0.0000001, 1.0, log = True),
                    vf_coef = trial.suggest_float('vf_coef', 0.0000001, 1.0, log = True),
                    )
    
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                            net_arch=dict(pi=[32, 32], vf=[32, 32]),
                            features_extractor_class=CustomMLP,
                            features_extractor_kwargs=dict(features_dim=128)
                    )

    # train_env = TestRLEnv(battle_format="gen8randombattle", opponent = opponent, start_challenging=True)

    model = PPO('MlpPolicy', env = train_env, verbose = 0, batch_size = batch_size, 
                policy_kwargs = policy_kwargs, **kwargs)
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

model_name = 'pokemon_ppo1'
date_now = datetime.now().strftime('%Y%m%d_%H%M%S')
study_filename = f'study_{date_now}.pkl'
folder = f'E:/Public/Trading/data/06_model/{model_name}'

if not os.path.exists(folder):
    os.makedirs(folder)

filename = os.path.join(folder, study_filename)

print(F'CREATING NEW STUDY')
study = optuna.create_study(study_name = 'test1',
                            direction="maximize",
                            # pruner=optuna.pruners.MedianPruner(
                            #             n_startup_trials=5, n_warmup_steps=30, interval_steps=10
                            # ),
                            )

print(F'Optimizing')
try: 
    study.optimize(objective, n_trials=150)
except Exception as e:
    print(e)
except KeyboardInterrupt:
    pass
with open(filename, 'wb') as file:
    print(F'SAVING {filename}')
    pkl.dump(study, file)
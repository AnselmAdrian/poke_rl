import numpy as np
from gym.spaces import Space, Box
from poke_env.player import Gen8EnvSinglePlayer
from datetime import datetime
import torch as th

from gym.utils.env_checker import check_env
from poke_env.player import RandomPlayer, MaxBasePowerPlayer
from stable_baselines3 import PPO
import time
import asyncio
from poke_env.player import background_evaluate_player
from common import SimpleRLEnv, RLEnv, evaluate_player, TestRLPlayer
from model import CustomMLP
from random import choice

if __name__ == '__main__':

    async def main(player1, player2):
        start = time.time()

        # Now, let's evaluate our player
        await player1.battle_against(player2, n_battles=100)

        print(
            "Max damage player won %d / 100 battles [this took %f seconds]"
            % (
                player1.n_won_battles, time.time() - start
            )
        )

    opponent = RandomPlayer(battle_format="gen8randombattle")

    opponents = [
                    RandomPlayer(battle_format="gen8randombattle"),
                    # MaxBasePowerPlayer(battle_format="gen8randombattle")
                 ]
    #train_env = SimpleRLEnv(battle_format="gen8randombattle", opponent = opponent, start_challenging=True)
    train_env = RLEnv(battle_format="gen8randombattle", opponent = opponent, start_challenging=True)
    try:
        print('Checking Environment')
        check_env(train_env)
    except Exception as e:
        print(e)

    # Create stable baseline model
    # kwargs = {
    #             'learning_rate': 0.0003071888865904524, 
    #             'gamma': 0.3511201957562162, 
    #             'gae_lambda': 0.5924822869558098, 
    #             'normalize_advantage': True, 
    #             'ent_coef': 1.726829000128019e-07, 
    #             'vf_coef': 0.17962332078088222
    #             }
    kwargs = {
                'learning_rate': 0.0001,  
                'gamma': 0.95, 
                'gae_lambda': 0.6, 
                'clip_range': 0.1,
                'normalize_advantage': True, 
                'ent_coef': 1.7e-07, 
                'vf_coef': 0.18
                }
    
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                            net_arch=dict(pi=[32, 32], vf=[32, 32]),
                            features_extractor_class=CustomMLP,
                            features_extractor_kwargs=dict(features_dim=128)
                    )

    model = PPO('MlpPolicy', env = train_env, verbose = 1, batch_size = 16, 
                policy_kwargs = policy_kwargs, **kwargs, tensorboard_log="logs/ppo/")
    model.load('data/06_model/pokemon/TestRLEnv/ppo_20230630_192429.h5') # Best version so far
    try:
        for _ in range(1000):
            model.learn(1000_000)

            curr_opponent = choice(opponents)
            print(f'Changing to {curr_opponent}')
            train_env.reset_env(opponent = curr_opponent)
            model.set_env(train_env, force_reset=True)
    except Exception as e:
        print('Stopping training early due to error!')
        print(e)
    except KeyboardInterrupt:
        print('Stopping training early')

    #Save model
    date_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f'data/06_model/pokemon/TestRLEnv/ppo_{date_now}.h5'
    print('saving file at:', filepath)
    model.save(filepath)

    train_env.close()

    # train_env = SimpleRLPlayer(
    #     battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    # )
    opponent = lambda: RandomPlayer(battle_format="gen8randombattle")
    rl_player = lambda: TestRLPlayer(model = model,
        battle_format="gen8randombattle",
    )
    evaluate_player(rl_player, opponent)

    # n_challenges = 100
    # placement_battles = 40
    # eval_task = background_evaluate_player(
    #     eval_env.agent, n_challenges, placement_battles
    # )   

    # obs = eval_env.reset()
    # for _ in range(100):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = eval_env.step(action)
    #     if dones:
    #         print('Done')
    #         break
    #     eval_env.render()
    # eval_env.close()


    #print("Evaluation with included method:", eval_task.result())
    #asyncio.get_event_loop().run_until_complete(main())
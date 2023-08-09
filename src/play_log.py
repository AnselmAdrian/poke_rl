import asyncio
import time

from poke_env.player import Player, RandomPlayer
from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration
from stable_baselines3 import PPO, DQN, A2C

from common.pokemon_players import MaxDamagePlayer, RLPlayer, TestRLPlayer, HumanPlayer
from common import SimpleRLEnv, RLEnv, evaluate_player, TestRLPlayer

my_server_config= ServerConfiguration(
    "localhost:8000",
    "authentication-endpoint.com/action.php?"
)

async def human_vs_main_bot():
    n = 1
    start = time.time()

    # We create two players.
    random_player = RandomPlayer(
        battle_format="gen8randombattle",
    )
    max_damage_player = MaxDamagePlayer(
        battle_format="gen8randombattle",
    )
    
    player2 = random_player

    # model = PPO.load('data/06_model/pokemon/ppo_test2.h5')
    # rl_player = RLPlayer(model = model,
    #     battle_format="gen8randombattle"
    # )
    # model = PPO.load('data/06_model/pokemon/TestRLEnv/ppo_test1.h5')
    model = PPO.load('data/06_model/pokemon/TestRLEnv/ppo_20230630_192429.h5')
    # model = DQN.load('data/06_model/pokemon/TestRLEnv/ppo_20230630_192429.h5')
    rl_player = TestRLPlayer(model = model,
        battle_format="gen8randombattle",
    )
    player_configuration=PlayerConfiguration("testsubject", None)
    human_player = HumanPlayer(
        player_configuration=player_configuration,
        server_configuration=my_server_config,
        battle_format="gen8randombattle")

    player1 = human_player

    # Now, let's evaluate our player
    await player1.battle_against(player2, n_battles=n)

    print(
        "Player won %d / %d battles [this took %f seconds]"
        % (
            player1.n_won_battles, n, time.time() - start
        )
    )
    

if __name__ == '__main__':

    # This will work on servers that do not require authentication, which is the
    # case of the server launched in our 'Getting Started' section
    # my_player_config = PlayerConfiguration("my_username", None) 
    # my_player_config = PlayerConfiguration("my_username", "super-secret-password")

    #asyncio.get_event_loop().run_until_complete(main_bot_vs_human())
    asyncio.get_event_loop().run_until_complete(human_vs_main_bot())
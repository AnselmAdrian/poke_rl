import asyncio
import time

from poke_env.player import Player, RandomPlayer
from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration
from stable_baselines3 import PPO

from common.pokemon_players import MaxDamagePlayer, RLPlayer, TestRLPlayer

my_server_config= ServerConfiguration(
    "localhost:8000",
    "authentication-endpoint.com/action.php?"
)

async def main_bot_vs_bot():
    n = 1000
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
    model = PPO.load('data/06_model/pokemon/TestRLEnv/ppo_20230625_165229.h5')
    rl_player = TestRLPlayer(model = model,
        battle_format="gen8randombattle",
    )

    player1 = rl_player

    # Now, let's evaluate our player
    await player1.battle_against(player2, n_battles=n)

    print(
        "Player won %d / %d battles [this took %f seconds]"
        % (
            player1.n_won_battles, n, time.time() - start
        )
    )

async def main_bot_vs_human():
    start = time.time()
    player_configuration=PlayerConfiguration("deadlykitten", None)
    # We create a random player
    # player = RandomPlayer(
    #     player_configuration=PlayerConfiguration("bot_username", None),
    #     server_configuration=my_server_config,
    # )
    #model = PPO.load('data/06_model/pokemon/TestRLEnv/ppo_test1.h5')
    model = PPO.load('data/06_model/pokemon/TestRLEnv/ppo_20230619_062147.h5')

    player = TestRLPlayer(model = model,
        player_configuration=player_configuration,
        server_configuration=my_server_config,
        battle_format="gen8randombattle"
    )
    # player = RandomPlayer(
    #         battle_format="gen8randombattle",
    #         player_configuration = player_configuration,
    #     )
    # Sending challenges to 'your_username'
    await player.send_challenges("AnselmAdrian", n_challenges=1)

    # Accepting one challenge from any user
    await player.accept_challenges(None, 1)

    # Accepting three challenges from 'your_username'
    await player.accept_challenges('AnselmAdrian', 1)

    # Playing 5 games on the ladder
    await player.ladder(1)

    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
        print(battle.rating, battle.opponent_rating)

    print(
        "Max damage player won %d / 100 battles [this took %f seconds]"
        % (
                await player.accept_challenges('your_username', 1).n_won_battles, time.time() - start
        )
    )

def evaluate_player(player, opponent, n_battles = 100):

    async def main_bot_vs_bot():
        start = time.time()
        player2 = opponent

        player1 = player

        # Now, let's evaluate our player
        await player1.battle_against(player2, n_battles=n_battles)

        print(
            "Player won %d / 100 battles [this took %f seconds]"
            % (
                player1.n_won_battles, time.time() - start
            )
        )
        return player1.n_won_battles / n_battles

    return asyncio.get_event_loop().run_until_complete(main_bot_vs_bot())


if __name__ == '__main__':

    # This will work on servers that do not require authentication, which is the
    # case of the server launched in our 'Getting Started' section
    # my_player_config = PlayerConfiguration("my_username", None) 
    # my_player_config = PlayerConfiguration("my_username", "super-secret-password")

    asyncio.get_event_loop().run_until_complete(main_bot_vs_human())
    #asyncio.get_event_loop().run_until_complete(main_bot_vs_bot())
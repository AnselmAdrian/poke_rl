import time
import asyncio

def evaluate_player(player, opponent, n_battles = 100, verbose = 0):

    async def main_bot_vs_bot():
        start = time.time()

        player2 = opponent()

        player1 = player()

        # Now, let's evaluate our player
        await player1.battle_against(player2, n_battles=n_battles)

        if verbose > 0:
            print(
                "Player won %d / %d battles [this took %f seconds]"
                % (
                    player1.n_won_battles, n_battles, time.time() - start
                )
            )
        return player1.n_won_battles / n_battles

    return asyncio.get_event_loop().run_until_complete(main_bot_vs_bot())
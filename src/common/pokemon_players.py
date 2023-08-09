from poke_env.player import Player

from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
from .pokemon_env import SimpleRLEnv, RLEnv

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)
        
class HumanPlayer(Player):
    def choose_move(self, battle):
        for i, move in enumerate(battle.available_moves):
            print(i, move)
        move_idx = input()
        return battle.available_moves[move_idx]
        
class RLPlayer(Player):
    def __init__(self, model = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def embed_battle(self, battle):
        return SimpleRLEnv.embed_battle(battle)

    def action_to_move(self, action: int, battle) -> BattleOrder:  # pyre-ignore
        """Converts actions to move orders.

        The conversion is done as follows:

        action = -1:
            The battle will be forfeited.
        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        4 <= action < 8:
            The action - 4th available move in battle.available_moves is executed, with
            z-move.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        12 <= action < 16:
            The action - 12th available move in battle.available_moves is executed,
            while dynamaxing.
        16 <= action < 22
            The action - 16th available switch in battle.available_switches is executed.

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0
            <= action - 4
            < len(battle.active_pokemon.available_z_moves)  # pyre-ignore
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 8], mega=True
            )
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 12], dynamax=True
            )
        elif 0 <= action - 16 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 16])
        else:
            return self.choose_random_move(battle)

    def choose_move(self, battle):
        obs = self.embed_battle(battle)
        action, _states = self.model.predict(obs)
        best_move = self.action_to_move(action, battle)
        return best_move

class TestRLPlayer(RLPlayer):
    def embed_battle(self, battle):
        return RLEnv.embed_battle(battle)
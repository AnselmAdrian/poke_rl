from poke_env.player import Gen8EnvSinglePlayer
import numpy as np
from gym.spaces import Space, Box
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.environment.move import Move, MoveCategory
from poke_env.environment.effect import Effect
from poke_env.environment.field import Field
from poke_env.environment.pokemon_gender import PokemonGender
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.environment.weather import Weather
from poke_env.environment.battle import Battle
from collections import defaultdict
import yaml
import random

TYPE_CHART = dict(  BUG = dict(  BUG = 0.5,
                                DARK = 2,
                                DRAGON = 1,
                                ELECTRIC = 1,
                                FAIRY = 0.5,
                                FIGHTING = 0.5,
                                FIRE = 0.5,
                                FLYING = 0.5,
                                GHOST = 0.5,
                                GRASS = 2,
                                GROUND = 1,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 0.5,
                                PSYCHIC = 2,
                                ROCK = 1,
                                STEEL = 0.5,
                                WATER = 1,
                                ),
                    DARK = dict(  BUG = 1,
                                DARK = 0.5,
                                DRAGON = 1,
                                ELECTRIC = 1,
                                FAIRY = 0.5,
                                FIGHTING = 0.5,
                                FIRE = 1,
                                FLYING = 1,
                                GHOST = 2,
                                GRASS = 1,
                                GROUND = 1,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 2,
                                ROCK = 1,
                                STEEL = 1,
                                WATER = 1,
                                ),
                    DRAGON = dict(  BUG = 1,
                                DARK = 1,
                                DRAGON = 2,
                                ELECTRIC = 1,
                                FAIRY = 0,
                                FIGHTING = 1,
                                FIRE = 1,
                                FLYING = 1,
                                GHOST = 1,
                                GRASS = 1,
                                GROUND = 1,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 1,
                                ROCK = 1,
                                STEEL = 0.5,
                                WATER = 1,
                                ),
                    ELECTRIC = dict(  BUG = 1,
                                DARK = 1,
                                DRAGON = 0.5,
                                ELECTRIC = 0.5,
                                FAIRY = 1,
                                FIGHTING = 1,
                                FIRE = 1,
                                FLYING = 2,
                                GHOST = 1,
                                GRASS = 0.5,
                                GROUND = 0,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 1,
                                ROCK = 1,
                                STEEL = 1,
                                WATER = 2,
                                ),
                    FAIRY = dict(  BUG = 1,
                                DARK = 2,
                                DRAGON = 2,
                                ELECTRIC = 1,
                                FAIRY = 1,
                                FIGHTING = 2,
                                FIRE = 0.5,
                                FLYING = 1,
                                GHOST = 1,
                                GRASS = 1,
                                GROUND = 1,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 0.5,
                                PSYCHIC = 1,
                                ROCK = 1,
                                STEEL = 0.5,
                                WATER = 1,
                                ),
                    FIGHTING = dict(  BUG = 0.5,
                                DARK = 2,
                                DRAGON = 1,
                                ELECTRIC = 1,
                                FAIRY = 0.5,
                                FIGHTING = 1,
                                FIRE = 1,
                                FLYING = 0.5,
                                GHOST = 0,
                                GRASS = 1,
                                GROUND = 1,
                                ICE = 2,
                                NORMAL = 2,
                                POISON = 0.5,
                                PSYCHIC = 0.5,
                                ROCK = 2,
                                STEEL = 2,
                                WATER = 1,
                                ),
                    FIRE = dict(  BUG = 2,
                                DARK = 1,
                                DRAGON = 0.5,
                                ELECTRIC = 1,
                                FAIRY = 1,
                                FIGHTING = 1,
                                FIRE = 0.5,
                                FLYING = 1,
                                GHOST = 1,
                                GRASS = 2,
                                GROUND = 1,
                                ICE = 2,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 1,
                                ROCK = 0.5,
                                STEEL = 2,
                                WATER = 0.5,
                                ),
                    FLYING = dict(  BUG = 2,
                                DARK = 1,
                                DRAGON = 1,
                                ELECTRIC = 0.5,
                                FAIRY = 1,
                                FIGHTING = 2,
                                FIRE = 1,
                                FLYING = 1,
                                GHOST = 1,
                                GRASS = 2,
                                GROUND = 1,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 1,
                                ROCK = 0.5,
                                STEEL = 0.5,
                                WATER = 1,
                                ),
                    GHOST = dict(  BUG = 1,
                                DARK = 0.5,
                                DRAGON = 1,
                                ELECTRIC = 1,
                                FAIRY = 1,
                                FIGHTING = 1,
                                FIRE = 1,
                                FLYING = 1,
                                GHOST = 2,
                                GRASS = 1,
                                GROUND = 1,
                                ICE = 1,
                                NORMAL = 0,
                                POISON = 1,
                                PSYCHIC = 2,
                                ROCK = 1,
                                STEEL = 1,
                                WATER = 1,
                                ),
                    GRASS = dict(  BUG = 0.5,
                                DARK = 1,
                                DRAGON = 0.5,
                                ELECTRIC = 1,
                                FAIRY = 1,
                                FIGHTING = 1,
                                FIRE = 0.5,
                                FLYING = 0.5,
                                GHOST = 1,
                                GRASS = 0.5,
                                GROUND = 2,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 0.5,
                                PSYCHIC = 1,
                                ROCK = 2,
                                STEEL = 0.5,
                                WATER = 2,
                                ),
                    GROUND = dict(  BUG = 0.5,
                                DARK = 1,
                                DRAGON = 1,
                                ELECTRIC = 2,
                                FAIRY = 1,
                                FIGHTING = 1,
                                FIRE = 2,
                                FLYING = 0,
                                GHOST = 1,
                                GRASS = 0.5,
                                GROUND = 1,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 2,
                                PSYCHIC = 1,
                                ROCK = 2,
                                STEEL = 2,
                                WATER = 1,
                                ),
                    ICE = dict(  BUG = 1,
                                DARK = 1,
                                DRAGON = 2,
                                ELECTRIC = 1,
                                FAIRY = 1,
                                FIGHTING = 1,
                                FIRE = 0.5,
                                FLYING = 2,
                                GHOST = 1,
                                GRASS = 2,
                                GROUND = 2,
                                ICE = 0.5,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 1,
                                ROCK = 1,
                                STEEL = 0.5,
                                WATER = 0.5,
                                ),
                    NORMAL = dict(  BUG = 1,
                                DARK = 1,
                                DRAGON = 1,
                                ELECTRIC = 1,
                                FAIRY = 1,
                                FIGHTING = 1,
                                FIRE = 1,
                                FLYING = 1,
                                GHOST = 0,
                                GRASS = 1,
                                GROUND = 1,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 1,
                                ROCK = 0.5,
                                STEEL = 0.5,
                                WATER = 1,
                                ),
                    POISON = dict(  BUG = 1,
                                DARK = 1,
                                DRAGON = 1,
                                ELECTRIC = 1,
                                FAIRY = 1,
                                FIGHTING = 1,
                                FIRE = 1,
                                FLYING = 1,
                                GHOST = 1,
                                GRASS = 1,
                                GROUND = 1,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 1,
                                ROCK = 1,
                                STEEL = 1,
                                WATER = 1,
                                ),
                    PSYCHIC = dict(  BUG = 1,
                                DARK = 1,
                                DRAGON = 1,
                                ELECTRIC = 1,
                                FAIRY = 2,
                                FIGHTING = 1,
                                FIRE = 1,
                                FLYING = 1,
                                GHOST = 0.5,
                                GRASS = 2,
                                GROUND = 0.5,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 0.5,
                                PSYCHIC = 1,
                                ROCK = 0.5,
                                STEEL = 0,
                                WATER = 1,
                                ),
                    ROCK = dict(  BUG = 2,
                                DARK = 1,
                                DRAGON = 1,
                                ELECTRIC = 1,
                                FAIRY = 1,
                                FIGHTING = 0.5,
                                FIRE = 2,
                                FLYING = 2,
                                GHOST = 1,
                                GRASS = 1,
                                GROUND = 0.5,
                                ICE = 2,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 1,
                                ROCK = 1,
                                STEEL = 0.5,
                                WATER = 1,
                                ),
                    STEEL = dict(  BUG = 1,
                                DARK = 1,
                                DRAGON = 1,
                                ELECTRIC = 0.5,
                                FAIRY = 2,
                                FIGHTING = 1,
                                FIRE = 0.5,
                                FLYING = 1,
                                GHOST = 1,
                                GRASS = 1,
                                GROUND = 1,
                                ICE = 2,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 1,
                                ROCK = 2,
                                STEEL = 0.5,
                                WATER = 0.5,
                                ),
                    WATER = dict(  BUG = 1,
                                DARK = 1,
                                DRAGON = 0.5,
                                ELECTRIC = 1,
                                FAIRY = 1,
                                FIGHTING = 1,
                                FIRE = 2,
                                FLYING = 1,
                                GHOST = 1,
                                GRASS = 0.5,
                                GROUND = 2,
                                ICE = 1,
                                NORMAL = 1,
                                POISON = 1,
                                PSYCHIC = 1,
                                ROCK = 2,
                                STEEL = 1,
                                WATER = 0.5,
                                ),
                    )
    
class SimpleRLEnv(Gen8EnvSinglePlayer):
    def __init__(self, model = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    @staticmethod
    def embed_battle(battle):
        # TODO:
        # Include game stats
        # Opponents moves
        # Team status

        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart = TYPE_CHART,
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
    
    def choose_move(self, battle):
        obs = self.embed_battle(battle)
        action, _states = self.model.predict(obs)
        return self.action_to_move(action, battle)
        
class  RLEnv(SimpleRLEnv):
    memory = {}
    embedding_memory = {}
    team_size = 6
    pokemon_embedding_size = 1212 + 16 * 12 * 4
    move_embedding_size = 168 + 16
    max_memory_dict_size = 256
    MoveCategory_emdedding = {
                                MoveCategory.PHYSICAL: 0,
                                MoveCategory.SPECIAL: 1,
                                MoveCategory.STATUS: 2
                                }
    PokemonGender_emdedding = {
                                PokemonGender.FEMALE: 0,
                                PokemonGender.MALE: 1,
                                PokemonGender.NEUTRAL: 2
                                }
    Effect_emdedding = {
                        Effect.AFTER_YOU: 0,
                        Effect.AFTERMATH: 1,
                        Effect.AQUA_RING: 2,
                        Effect.AROMATHERAPY: 3,
                        Effect.AROMA_VEIL: 4,
                        Effect.ATTRACT: 5,
                        Effect.AUTOTOMIZE: 6,
                        Effect.BAD_DREAMS: 7,
                        Effect.BANEFUL_BUNKER: 8,
                        Effect.BATTLE_BOND: 9,
                        Effect.BIDE: 10,
                        Effect.BIND: 11,
                        Effect.BURN_UP: 12,
                        Effect.CELEBRATE: 13,
                        Effect.CHARGE: 14,
                        Effect.CLAMP: 15,
                        Effect.CONFUSION: 16,
                        Effect.COURT_CHANGE: 17,
                        Effect.CRAFTY_SHIELD: 18,
                        Effect.CURSE: 19,
                        Effect.CUSTAP_BERRY: 20,
                        Effect.DANCER: 21,
                        Effect.DESTINY_BOND: 22,
                        Effect.DISABLE: 23,
                        Effect.DISGUISE: 24,
                        Effect.DOOM_DESIRE: 25,
                        Effect.DYNAMAX: 26,
                        Effect.EERIE_SPELL: 27,
                        Effect.ELECTRIC_TERRAIN: 28,
                        Effect.EMBARGO: 29,
                        Effect.EMERGENCY_EXIT: 30,
                        Effect.ENCORE: 31,
                        Effect.ENDURE: 32,
                        Effect.FAIRY_LOCK: 33,
                        Effect.FEINT: 34,
                        Effect.FIRE_SPIN: 35,
                        Effect.FLASH_FIRE: 36,
                        Effect.FLOWER_VEIL: 37,
                        Effect.FOCUS_BAND: 38,
                        Effect.FOCUS_ENERGY: 39,
                        Effect.FORESIGHT: 40,
                        Effect.FOREWARN: 41,
                        Effect.FUTURE_SIGHT: 42,
                        Effect.G_MAX_CENTIFERNO: 43,
                        Effect.G_MAX_CHI_STRIKE: 44,
                        Effect.G_MAX_ONE_BLOW: 45,
                        Effect.G_MAX_RAPID_FLOW: 46,
                        Effect.G_MAX_SANDBLAST: 47,
                        Effect.GRAVITY: 48,
                        Effect.GRUDGE: 49,
                        Effect.GUARD_SPLIT: 50,
                        Effect.GULP_MISSILE: 51,
                        Effect.HEAL_BELL: 52,
                        Effect.HEAL_BLOCK: 53,
                        Effect.HEALER: 54,
                        Effect.HYDRATION: 55,
                        Effect.HYPERSPACE_FURY: 56,
                        Effect.HYPERSPACE_HOLE: 57,
                        Effect.ICE_FACE: 58,
                        Effect.ILLUSION: 59,
                        Effect.IMMUNITY: 60,
                        Effect.IMPRISON: 61,
                        Effect.INFESTATION: 62,
                        Effect.INGRAIN: 63,
                        Effect.INNARDS_OUT: 64,
                        Effect.INSOMNIA: 65,
                        Effect.IRON_BARBS: 66,
                        Effect.LASER_FOCUS: 67,
                        Effect.LEECH_SEED: 68,
                        Effect.LIGHTNING_ROD: 69,
                        Effect.LIMBER: 70,
                        Effect.LIQUID_OOZE: 71,
                        Effect.LOCK_ON: 72,
                        Effect.MAGMA_STORM: 73,
                        Effect.MAGNET_RISE: 74,
                        Effect.MAGNITUDE: 75,
                        Effect.MAT_BLOCK: 76,
                        Effect.MAX_GUARD: 77,
                        Effect.MIMIC: 78,
                        Effect.MIMICRY: 79,
                        Effect.MIND_READER: 80,
                        Effect.MINIMIZE: 81,
                        Effect.MIRACLE_EYE: 82,
                        Effect.MIST: 83,
                        Effect.MISTY_TERRAIN: 84,
                        Effect.MUMMY: 85,
                        Effect.NEUTRALIZING_GAS: 86,
                        Effect.NIGHTMARE: 87,
                        Effect.NO_RETREAT: 88,
                        Effect.OBLIVIOUS: 89,
                        Effect.OCTOLOCK: 90,
                        Effect.OWN_TEMPO: 91,
                        Effect.PASTEL_VEIL: 92,
                        Effect.PERISH0: 93,
                        Effect.PERISH1: 94,
                        Effect.PERISH2: 95,
                        Effect.PERISH3: 96,
                        Effect.PHANTOM_FORCE: 97,
                        Effect.POLTERGEIST: 98,
                        Effect.POWDER: 99,
                        Effect.POWER_CONSTRUCT: 100,
                        Effect.POWER_SPLIT: 101,
                        Effect.POWER_TRICK: 102,
                        Effect.PROTECT: 103,
                        Effect.PROTECTIVE_PADS: 104,
                        Effect.PSYCHIC_TERRAIN: 105,
                        Effect.PURSUIT: 106,
                        Effect.QUASH: 107,
                        Effect.QUICK_CLAW: 108,
                        Effect.QUICK_GUARD: 109,
                        Effect.REFLECT: 110,
                        Effect.RIPEN: 111,
                        Effect.ROUGH_SKIN: 112,
                        Effect.SAFEGUARD: 113,
                        Effect.SAFETY_GOGGLES: 114,
                        Effect.SAND_TOMB: 115,
                        Effect.SCREEN_CLEANER: 116,
                        Effect.SHADOW_FORCE: 117,
                        Effect.SHED_SKIN: 118,
                        Effect.SKETCH: 119,
                        Effect.SKILL_SWAP: 120,
                        Effect.SKY_DROP: 121,
                        Effect.SLOW_START: 122,
                        Effect.SMACK_DOWN: 123,
                        Effect.SNAP_TRAP: 124,
                        Effect.SNATCH: 125,
                        Effect.SPEED_SWAP: 126,
                        Effect.SPITE: 127,
                        Effect.STICKY_HOLD: 128,
                        Effect.STICKY_WEB: 129,
                        Effect.STOCKPILE: 130,
                        Effect.STOCKPILE1: 131,
                        Effect.STOCKPILE2: 132,
                        Effect.STOCKPILE3: 133,
                        Effect.STORM_DRAIN: 134,
                        Effect.STRUGGLE: 135,
                        Effect.SUBSTITUTE: 136,
                        Effect.SUCTION_CUPS: 137,
                        Effect.SWEET_VEIL: 138,
                        Effect.SYMBIOSIS: 139,
                        Effect.SYNCHRONIZE: 140,
                        Effect.TAR_SHOT: 141,
                        Effect.TAUNT: 142,
                        Effect.TELEKINESIS: 143,
                        Effect.TELEPATHY: 144,
                        Effect.THROAT_CHOP: 145,
                        Effect.THUNDER_CAGE: 146,
                        Effect.TORMENT: 147,
                        Effect.TRAPPED: 148,
                        Effect.TRICK: 149,
                        Effect.TYPEADD: 150,
                        Effect.TYPECHANGE: 151,
                        Effect.TYPE_CHANGE: 152,
                        Effect.UPROAR: 153,
                        Effect.VITAL_SPIRIT: 154,
                        Effect.WANDERING_SPIRIT: 155,
                        Effect.WATER_BUBBLE: 156,
                        Effect.WATER_VEIL: 157,
                        Effect.WHIRLPOOL: 158,
                        Effect.WIDE_GUARD: 159,
                        Effect.WIMP_OUT: 160,
                        Effect.WRAP: 161,
                        Effect.YAWN: 162,
    }
    PokemonType_emdedding = {
                                PokemonType.BUG: 1,
                                PokemonType.DARK: 2,
                                PokemonType.DRAGON  : 3,
                                PokemonType.ELECTRIC: 4,
                                PokemonType.FAIRY: 5,
                                PokemonType.FIGHTING: 6,
                                PokemonType.FIRE: 7,
                                PokemonType.FLYING: 8,
                                PokemonType.GHOST: 9,
                                PokemonType.GRASS: 10,
                                PokemonType.GROUND: 11,
                                PokemonType.ICE: 12,
                                PokemonType.POISON: 13,
                                PokemonType.NORMAL: 14,
                                PokemonType.ROCK: 15,
                                PokemonType.PSYCHIC: 16,
                                PokemonType.STEEL: 17,
                                PokemonType.WATER: 18,
                             }
    Field_emdedding = {
                            Field.ELECTRIC_TERRAIN: 1,
                            Field.GRASSY_TERRAIN: 2,
                            Field.GRAVITY: 3,
                            Field.HEAL_BLOCK: 4,
                            Field.MAGIC_ROOM: 5,
                            Field.MISTY_TERRAIN: 6,
                            Field.MUD_SPORT: 7,
                            Field.MUD_SPOT: 8,
                            Field.PSYCHIC_TERRAIN: 9,
                            Field.TRICK_ROOM: 10,
                            Field.WATER_SPORT: 11,
                            Field.WONDER_ROOM: 12,
                        }                         
    SideCondition_emdedding = {
                                SideCondition.AURORA_VEIL: 1,
                                SideCondition.FIRE_PLEDGE: 2,
                                SideCondition.G_MAX_CANNONADE: 3,
                                SideCondition.G_MAX_STEELSURGE: 4,
                                SideCondition.G_MAX_VINE_LASH: 5,
                                SideCondition.G_MAX_VOLCALITH: 6,
                                SideCondition.G_MAX_WILDFIRE: 7,
                                SideCondition.GRASS_PLEDGE: 8,
                                SideCondition.LIGHT_SCREEN: 9,
                                SideCondition.LUCKY_CHANT: 10,
                                SideCondition.MIST: 11,
                                SideCondition.REFLECT: 12,
                                SideCondition.SAFEGUARD: 13,
                                SideCondition.SPIKES: 14,
                                SideCondition.STEALTH_ROCK: 15,
                                SideCondition.STICKY_WEB: 16,
                                SideCondition.TAILWIND: 17,
                                SideCondition.TOXIC_SPIKES: 18,
                                SideCondition.WATER_PLEDGE: 19
                                }
    Status_emdedding = {    
                            Status.BRN: 1,
                            Status.FNT: 2,
                            Status.FRZ: 3,
                            Status.PAR: 4,
                            Status.PSN: 5,
                            Status.SLP: 6,
                            Status.TOX: 7,}
    Weather_emdedding = {
                            Weather.DESOLATELAND: 1,
                            Weather.DELTASTREAM: 2,
                            Weather.HAIL: 3,
                            Weather.PRIMORDIALSEA: 4,
                            Weather.RAINDANCE: 5,
                            Weather.SANDSTORM: 6,
                            Weather.SUNNYDAY: 7,
                        }

    #Effect_emdedding = {k: i+ 1 for i, k in enumerate(list(vars(Effect).keys())[13:-1])}
    #Field_emdedding = {k: i+ 1 for i, k in enumerate(list(vars(Field).keys())[11:-1])}
    #PokemonType_emdedding = {k: i+ 1 for i, k in enumerate(list(vars(PokemonType).keys())[10:-1])}
    #SideCondition_emdedding = {k: i+ 1 for i, k in enumerate(list(vars(SideCondition).keys())[10:-1])}
    #Status_emdedding = {k: i+ 1 for i, k in enumerate(list(vars(Status).keys())[8:-1])}
    #Weather_emdedding = {k: i+ 1 for i, k in enumerate(list(vars(Weather).keys())[10:-1])}

    # print(Effect_emdedding)
    # print(Field_emdedding)
    # print(PokemonType_emdedding)
    # print(SideCondition_emdedding)
    # print(Status_emdedding)
    # print(Weather_emdedding)

    Move_low = [0.0] * move_embedding_size
    Move_high = [1.0] * move_embedding_size

    def load(self, filepath):
        with open(filepath) as file:
            in_dict = yaml.load(file, Loader=yaml.FullLoader)

        self.memory = in_dict['memory']
        self.embedding_memory = in_dict['embedding_memory']
        self.pokemon_embedding_size = in_dict['pokemon_embedding_size']
        self.max_memory_dict_size = in_dict['max_memory_dict_size']
        self.MoveCategory_emdedding = in_dict['MoveCategory_emdedding']
        self.PokemonGender_emdedding = in_dict['PokemonGender_emdedding']
        self.Effect_emdedding = in_dict['Effect_emdedding']
        self.Field_emdedding = in_dict['Field_emdedding']
        self.PokemonType_emdedding = in_dict['PokemonType_emdedding']
        self.SideCondition_emdedding = in_dict['SideCondition_emdedding']
        self.Status_emdedding = in_dict['Status_emdedding']
        self.Weather_emdedding = in_dict['Weather_emdedding']

    def save(self, filepath):
        out_dict = dict(memory = self.memory,
                        embedding_memory = self.embedding_memory,
                        pokemon_embedding_size = self.pokemon_embedding_size,
                        max_memory_dict_size = self.max_memory_dict_size,
                        MoveCategory_emdedding = self.MoveCategory_emdedding,
                        PokemonGender_emdedding = self.PokemonGender_emdedding,
                        Effect_emdedding = self.Effect_emdedding,
                        Field_emdedding = self.Field_emdedding,
                        PokemonType_emdedding = self.PokemonType_emdedding,
                        SideCondition_emdedding = self.SideCondition_emdedding,
                        Status_emdedding = self.Status_emdedding,
                        Weather_emdedding = self.Weather_emdedding,
                        )
        
        with open(filepath, 'w') as file:
            yaml.dump(out_dict, file)
        

    @classmethod
    def embed_str(self, input_str, category, max_memory_dict_size = None, val = 1):
        if max_memory_dict_size is None:
            max_memory_dict_size = self.max_memory_dict_size

        embedded_str = np.zeros(max_memory_dict_size)

        if input_str is None:
            return embedded_str
        
        if category not in self.embedding_memory:
            self.embedding_memory[category] = [], set()
        
        if isinstance(input_str, list):
            # Add multiple embeddings
            for input_str_i in input_str:
                if input_str_i in self.embedding_memory[category][1]:
                    idx = self.embedding_memory[category][0].index(input_str_i)
                else:
                    idx = len(self.embedding_memory[category][0])
                    # Update memorey of embeddings
                    self.embedding_memory[category][0].append(input_str_i)
                    self.embedding_memory[category][1].add(input_str_i)
                embedded_str[idx] = val
        else:
            if input_str in self.embedding_memory[category][1]:
                idx = self.embedding_memory[category][0].index(input_str)
            else:
                idx = len(self.embedding_memory[category][0])
                # Update memorey of embeddings
                self.embedding_memory[category][0].append(input_str)
                self.embedding_memory[category][1].add(input_str)
            embedded_str[idx] = val
        return embedded_str


    @staticmethod
    def embed_from_dict(x, myDict, default = 0):
        if x in myDict:
            return myDict[x]
        return default

    @classmethod
    def embed_dict_one_hot(self, val, myDict, reserve_zero = False):
        if reserve_zero:
            x = np.zeros(len(myDict) + 1)
        else:
            x = np.zeros(len(myDict))
        if val is None:
            return x

        if isinstance(val, dict):
            for v in val.keys():
                x[myDict[v]] = 1
            return x
        if isinstance(val, list):
            for v in val:
                x[myDict[v]] = 1
            return x
        x[myDict[val]] = 1
        return x

    @classmethod
    def embed_dict(self, myDict, category, value_embedding = None, max_memory_dict_size = None):
        if max_memory_dict_size is None:
            max_memory_dict_size = self.max_memory_dict_size

        embedded_dict = np.zeros(max_memory_dict_size)

        if myDict is None:
            return embedded_dict

        if category not in self.embedding_memory:
            self.embedding_memory[category] = [], set()

        # Embed what we know
        for i, key in enumerate(self.embedding_memory[category][0]):
            if key in myDict.keys():
                if value_embedding is None:
                    value = myDict[key]
                else:
                    value = value_embedding(myDict[key])
                embedded_dict[i] = value
        
        # Add new embeddings
        keys_to_add = myDict.keys() - self.embedding_memory[category][1]
        for key in list(keys_to_add):
            if value_embedding is None:
                value = myDict[key]
            else:
                value = value_embedding(myDict[key])
            embedded_dict[len(self.embedding_memory[category][0])] = value
            
            # Update memorey of embeddings
            self.embedding_memory[category][0].append(key)
            self.embedding_memory[category][1].add(key)

        return embedded_dict

    @classmethod
    def embed_ability(self, ability, possible_abilities, category = 'pokemon_ability', max_memory_dict_size = None):
        if ability is None:
            # Possible abilitys are embedded with 1
            return self.embed_str(possible_abilities, category, max_memory_dict_size = max_memory_dict_size, val = 0.5)

        # Revealed abilities are embedded with 2
        return self.embed_str(ability, category, max_memory_dict_size = max_memory_dict_size, val = 1)

    @classmethod
    def embed_move(self, move: Move, ):
        if move is None:
            return np.zeros(self.move_embedding_size)
        
        emded_move = np.array([
                                move.accuracy, # 0
                                0 if move.base_power is None else move.base_power / 860.0,
                                float(move.breaks_protect),
                                float(move.can_z_move),
                                self.MoveCategory_emdedding[move.category] / (len(self.MoveCategory_emdedding) - 1),
                                move.crit_ratio / 6.0,
                                move.current_pp/56.0,
                                move.max_pp/56.0,
                                # move.damage,,], not needed for now need to encode
                                # move.deduced_target,,], not needed
                                self.MoveCategory_emdedding[move.defensive_category] / (len(self.MoveCategory_emdedding) - 1),
                                move.drain,
                                # move.dynamaxed,,],  not needed
                                move.expected_hits/6.0, # 10
                                move.force_switch,
                                move.heal,
                                # move.id,,],  not needed
                                move.ignore_ability,
                                move.ignore_defensive,
                                move.ignore_evasion,
                                int(move.ignore_immunity) if isinstance(move.ignore_immunity, bool) else 0,
                                move.is_empty,
                                move.is_protect_counter,
                                move.is_protect_move,
                                # move.is_side_protect_move,0.0, 1.0],
                                move.is_z, # 20
                                move.n_hit[0]/5.0,
                                move.n_hit[1]/5.0,
                                move.no_pp_boosts,
                                move.non_ghost_target,
                                # move.pseudo_weather,,], not needed
                                move.recoil,
                                # move.secondary,,],
                                # move.self_boost,,],
                                # move.self_destruct,,],
                                # move.self_switch,,],
                                # move.side_condition,,],
                                move.sleep_usable,
                                # move.slot_condition,,],
                                move.stalling_move,
                                move.steals_boosts,
                                # move.target,,], for doubles
                                move.thaws_target,
                                move.use_target_offensive, # 30
                                # move.volatile_status,,],
                                # move.z_move_boost,,],
                                # move.z_move_effect,,],
                                move.z_move_power/300.0,
                                
                                ])
        # if emded_move.max() > 1:
        #     print('emded_move', emded_move.max())
        #     print(list(map(tuple, np.where((emded_move > 1) | (emded_move < 0)))))

        # emded_move = emded_move.fillna(0)
        # print('emded_move', emded_move.max())
        # print(list(map(tuple, np.where((emded_move > 1) | (emded_move < 0)))))

        # print('check embedding')
        # print(move)
        emded_move = emded_move.astype('float32')
        additional_embeddings = []

        n = len(self.Status_emdedding) + 1
        status_embed_move = self.embed_dict_one_hot(move.status, self.Status_emdedding, reserve_zero = True)
        # if status_embed_move.max() > 1:
        #    print('status_embed_move', status_embed_move.max())
        additional_embeddings.append(status_embed_move)

        n = len(self.Field_emdedding) + 1
        field_embed_move = self.embed_dict_one_hot(move.terrain, self.Field_emdedding, reserve_zero = True)
        # if field_embed_move.max() > 1:
        #     print('field_embed_move', field_embed_move.max())
        additional_embeddings.append(field_embed_move)

        n = len(self.PokemonType_emdedding) + 1
        type_embed_move = self.embed_dict_one_hot(move.type, self.PokemonType_emdedding, reserve_zero = True)
        # if type_embed_move.max() > 1:
        #     print('type_embed_move', type_embed_move.max())
        additional_embeddings.append(type_embed_move)

        n = len(self.Weather_emdedding) + 1
        weather_embed_move = self.embed_dict_one_hot(move.weather, self.Weather_emdedding, reserve_zero = True)
        # if weather_embed_move.max() > 1:
        #     print('weather_embed_move', weather_embed_move.max())
        additional_embeddings.append(weather_embed_move)

        n = 8
        boost_embed_move = self.embed_dict(move.boosts, 'boosts', value_embedding = None, max_memory_dict_size = n)
        boost_embed_move = (boost_embed_move / 12) + 0.5
        # if boost_embed_move.max() > 1:
        #     print('boost_embed_move', boost_embed_move.max())
        additional_embeddings.append(boost_embed_move)
        # print(boost_embed_move)
        
        
        # print(move.self_boost)
        n = 8
        self_boost_embed_move = (self.embed_dict(move.self_boost, 'self_boost', value_embedding = None, max_memory_dict_size = n) / 12) + 0.5
        # if self_boost_embed_move.max() > 1:
        #     print('self_boost_embed_move', self_boost_embed_move.max())
        additional_embeddings.append(self_boost_embed_move)
        # print(self_boost_embed_move)

        # print(move.secondary)
        # n = 8
        # temp = move.secondary
        # if not temp:
        #     temp = None
        # secondary_embed_move = np.concatenate([
        #                                     self.embed_dict(, 'secondary', value_embedding = None, max_memory_dict_size = n).reshape(-1,1), 
        #                                     np.array([-6] * n).reshape(-1,1), 
        #                                     np.array([6] * n).reshape(-1,1)
        #                                     ], 
        #                                     axis = 1)
        # print(secondary_embed_move)
        
        # print(move.side_condition)
        n = 16
        side_condition_embed_move = self.embed_str(move.side_condition, 'side_condition', max_memory_dict_size = n)
        # if side_condition_embed_move.max() > 1:
        #     print('side_condition_embed_move', side_condition_embed_move.max())
        additional_embeddings.append(side_condition_embed_move)
        # print(side_condition_embed_move)

        # print(move.slot_condition)
        n = 16
        slot_condition_embed_move = self.embed_str(move.slot_condition, 'slot_condition', max_memory_dict_size = n)
        # if slot_condition_embed_move.max() > 1:
        #     print('slot_condition_embed_move', slot_condition_embed_move.max())
        additional_embeddings.append(slot_condition_embed_move)
        # print(slot_condition_embed_move)

        # print(move.volatile_status)
        n = 32
        volatile_status_embed_move = self.embed_str(move.volatile_status, 'volatile_status',  max_memory_dict_size = n)
        # if volatile_status_embed_move.max() > 1:
        #     print('volatile_status_embed_move', volatile_status_embed_move.max())
        additional_embeddings.append(volatile_status_embed_move)
        # print(volatile_status_embed_move)

        # print(move.z_move_boost)
        n = 8
        z_move_boost_embed_move = (self.embed_dict(move.z_move_boost, 'z_move_boost', value_embedding = None, max_memory_dict_size = n) / 12) + 0.5
        # if z_move_boost_embed_move.max() > 1:
        #     print('z_move_boost_embed_move', z_move_boost_embed_move.max())
        additional_embeddings.append(z_move_boost_embed_move)
        # print(z_move_boost_embed_move)

        # print(move.z_move_effect)
        n = 16
        z_move_effect_embed_move = self.embed_str(move.z_move_effect, 'z_move_effect', max_memory_dict_size = n)
        # if z_move_effect_embed_move.max() > 1:
        #     print('z_move_effect_embed_move', z_move_effect_embed_move.max())
        additional_embeddings.append(z_move_effect_embed_move)
        # print(z_move_effect_embed_move)

        emded_move = np.concatenate([emded_move] + additional_embeddings)
        #print(emded_move.shape)

        return emded_move

    @classmethod
    def embed_pokemon(self, pokemon: Pokemon):
        revealed = pokemon.revealed
        embedded_pokemon = np.zeros(self.pokemon_embedding_size)
        if revealed:
            # 0
            i = 0
            embedded_pokemon[i] = revealed
            i += 1
            
            # 1
            n = 300
            embedded_pokemon[i:i + n] = self.embed_ability(pokemon.ability, pokemon.possible_abilities, category = 'pokemon_ability', max_memory_dict_size = n)
            i += n

            # 189
            embedded_pokemon[i] = float(pokemon.active)
            i += 1

            # 190
            n = 8
            embedded_pokemon[i:i + n] = (self.embed_dict(pokemon.boosts, 'boosts', value_embedding = None, max_memory_dict_size = n) / 12.0) + 0.5
            i += n

            # 198
            embedded_pokemon[i] = 0 if pokemon.current_hp is None else pokemon.current_hp / 1200.0 # pokemon.current_hp_fraction
            i += 1

            # 199
            embedded_pokemon[i] = float(pokemon.current_hp_fraction)
            i += 1

            # 200
            n = len(self.Effect_emdedding) + 1
            embedded_pokemon[i:i + n] = (self.embed_dict_one_hot(pokemon.effects, self.Effect_emdedding, reserve_zero = True) / 8.0)
            i += n

            # 364
            embedded_pokemon[i] = float(pokemon.fainted)
            i += 1
            
            # 365
            embedded_pokemon[i] = float(pokemon.first_turn)
            i += 1
            
            # 366
            embedded_pokemon[i] = self.embed_from_dict(pokemon.gender, self.PokemonGender_emdedding) / (len(self.PokemonGender_emdedding) - 1)
            i += 1

            # 367
            embedded_pokemon[i] = float(pokemon.is_dynamaxed)
            i += 1

            # 368
            # pokemon.base_stats # encode dict
            n = 7
            try:
                embedded_pokemon[i:i + n] = self.embed_dict(pokemon.stats, 'stats', value_embedding = None, max_memory_dict_size = n) / 510.0
            except AttributeError:
                pass # no embedding

            # if embedded_pokemon[i:i + n].max() > 1:
            #     print('Poke-Here-states', (i,i + n), (embedded_pokemon[i:i + n].min(), embedded_pokemon[i:i + n].max()))
            #     print(pokemon.stats)
            i += n

            # 375
            embedded_pokemon[i] = pokemon.level / 100.0
            i += 1

            # 376
            embedded_pokemon[i] = 0 if pokemon.max_hp is None else pokemon.max_hp / 1200.0
            i += 1

            # 377
            for move in pokemon.moves.values():
                temp = self.embed_move(move) # encode moves
                n = temp.shape[0]
                embedded_pokemon[i:i + n] = temp
                # if temp.max() > 1:
                #     print('Poke-Here-move!!!', (i,i + n), (temp.min(), temp.max()))
                i += n

            # 1049
            embedded_pokemon[i] = float(pokemon.must_recharge)
            i += 1

            #pokemon.preparing
            
            # 1050
            pokemon.protect_counter
            embedded_pokemon[i] = min(1, float(pokemon.protect_counter) / 3) 
            i += 1

            # 1051
            n = len(self.Status_emdedding) + 1
            embedded_pokemon[i:i + n] = self.embed_dict_one_hot(pokemon.status, self.Status_emdedding, reserve_zero = True)
            i += n

            embedded_pokemon[i] = pokemon.status_counter / 16.0
            i += 1

            n = len(self.PokemonType_emdedding) + 1
            embedded_pokemon[i:i + n] = self.embed_dict_one_hot(pokemon.type_1, self.PokemonType_emdedding, reserve_zero = True)
            i += n
            
            n = len(self.PokemonType_emdedding) + 1
            embedded_pokemon[i:i + n] = self.embed_dict_one_hot(pokemon.type_2, self.PokemonType_emdedding, reserve_zero = True)
            i += n

            embedded_pokemon[i] = pokemon.weight / 2000.0
            i += 1

            embedded_pokemon = embedded_pokemon.astype('float32')
            embedded_pokemon[np.isnan(embedded_pokemon)] = 0

            # print(list(map(tuple, np.where(embedded_pokemon > 1))))
        return embedded_pokemon

    @classmethod
    def embed_battle(self, battle: Battle):
        
        # Embed player team
        player_pokemons = list(battle.team.values())

        player_pokemon_embeddings = np.zeros(self.team_size * self.pokemon_embedding_size)
        player_available_moves = np.zeros(4 * self.move_embedding_size)
        player_available_switches = np.zeros(self.team_size * 1)

        active_pokemon_index = player_pokemons.index(battle.active_pokemon)
        for i, mon in enumerate(player_pokemons):
            if i == active_pokemon_index:
                j = 0
            elif i < active_pokemon_index:
                j = i + 1
            else:
                j = i
            player_pokemon_embeddings[j * self.pokemon_embedding_size: (j + 1) * self.pokemon_embedding_size] = self.embed_pokemon(mon)

        # for i, move in enumerate(battle.available_moves):
        #     player_available_moves[i * self.move_embedding_size: (i + 1)* self.move_embedding_size] = self.embed_move(move)

        for i, mon in enumerate(player_pokemons):
            player_available_switches[i] = mon in battle.available_switches

        player_status = np.array([
                                    #player_pokemons.index(battle.active_pokemon) / self.team_size,
                                    float(battle.can_dynamax), 
                                    float(battle.can_mega_evolve),
                                    float(battle.can_z_move),
                                    1.0 if isinstance(battle.force_switch, list) else float(battle.force_switch),
                                    float(battle.maybe_trapped)
                                    ])
        
        # Embed opponent's team
        opponent_pokemons = list(battle.opponent_team.values())

        opponent_pokemon_embeddings = np.zeros(self.team_size * self.pokemon_embedding_size)
        opponent_available_moves = np.zeros(4 * self.move_embedding_size)
        opponent_available_switches = np.zeros(self.team_size * 1)

        active_pokemon_index = opponent_pokemons.index(battle.opponent_active_pokemon)
        for i, mon in enumerate(opponent_pokemons):
            if i == active_pokemon_index:
                j = 0
            elif i < active_pokemon_index:
                j = i + 1
            else:
                j = i
            opponent_pokemon_embeddings[j * self.pokemon_embedding_size: (j + 1)* self.pokemon_embedding_size] = self.embed_pokemon(mon)

        # for i, move in enumerate(battle.available_moves):
        #     opponent_available_moves[i * self.move_embedding_size: (i + 1)* self.move_embedding_size] = self.embed_move(move)

        for i, mon in enumerate(opponent_pokemons):
            opponent_available_switches[i] = mon in battle.available_switches

        opponent_status = np.array([
                                    #opponent_pokemons.index(battle.opponent_active_pokemon) / self.team_size,
                                    float(battle.opponent_can_dynamax), 
                                    float(battle.opponent_can_mega_evolve),
                                    float(battle.opponent_can_z_move),
                                    # battle.opponent_trapped
                                    ], dtype = 'float32')
        
        
        # player_fainted_mon = len([mon for mon in battle.team.values() if mon.fainted])
        # opponent_fainted_mon = (
        #     len([mon for mon in battle.opponent_team.values() if mon.fainted])
        # )

        # TODO: Normalize embedings
        # n = 0
        # for x in [
        #                     player_pokemon_embeddings, #[]
        #                     #player_available_moves.reshape(-1,1), #[]
        #                     player_available_switches, #[]
        #                     player_status,
        #                     #player_fainted_mon.reshape(-1,1),
        #                     opponent_pokemon_embeddings,
        #                     #opponent_available_moves.reshape(-1,1),
        #                     opponent_available_switches,
        #                     opponent_status,
        #                     #opponent_fainted_mon.reshape(-1,1),
        #                     ]:
        #     n += x.shape[0]
        #     if x.max() > 1:
        #         print(x.shape, n, (x.min(), x.max()))
        #         for x_i in list(map(tuple, np.where(x>1))):
        #             print(f'\t',x_i)


        return np.concatenate([
                            player_pokemon_embeddings, #[]
                            #player_available_moves.reshape(-1,1), #[]
                            player_available_switches, #[]
                            player_status,
                            #player_fainted_mon.reshape(-1,1),
                            opponent_pokemon_embeddings,
                            #opponent_available_moves.reshape(-1,1),
                            opponent_available_switches,
                            opponent_status,
                            #opponent_fainted_mon.reshape(-1,1),
                            ], dtype='float32'
                         ).clip(0,1)
    
    def describe_embedding(self) -> Space:
        n = 12 * self.pokemon_embedding_size # player + opponent pokemon
        n += 12 # player + opponent available switches
        n += 5 # player_status
        n += 3 # opponent_status

        low = [0] * n
        high = [1] * n
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

class  RLEnv_multi_opponent(RLEnv):
    def __init__(self, model = None, opponents = None, *args, **kwargs):
        assert opponents, 'at least 1 opponent is required'
        self.opponents = opponents
        super().__init__(*args, opponent = self.opponents[0], **kwargs)
        self.model = model

    def _select_next_opponent(self):
        return random.choice(self.opponents)

    def reset_env(self, restart: bool = True):  # pragma: no cover
        """
        Resets the environment to an inactive state: it will forfeit all unfinished
        battles, reset the internal battle tracker and optionally change the next
        opponent and restart the challenge loop.

        :param restart: If True the challenge loop will be restarted before returning,
            otherwise the challenge loop will be left inactive and can be
            started manually.
        :type restart: bool
        """
        self.close(purge=False)
        self.reset_battles()
        if opponent:
            self.set_opponent(self._select_next_opponent())
        if restart:
            self.start_challenging()
        

if __name__ == '__main__': 
    from gym.utils.env_checker import check_env
    from poke_env.player import RandomPlayer
    import traceback

    ###########################################
    ################## RLEnv ##################
    ###########################################
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = RLEnv(battle_format="gen8randombattle", opponent = opponent, start_challenging=True)

    x = train_env.reset()

    try:
        print('Checking RLEnv Environment')
        for i in range(10):
            if i % 50 == 0:
                print(f'iteration: {i}')
            train_env.reset()
            check_env(train_env)
        train_env.close()
    except Exception as e:
        print(f'stopped at iteration: {i}')
        print(e)
        traceback.print_exc()
        train_env.close()
    except KeyboardInterrupt:
        train_env.close()
        exit(1)
    train_env.close()

    ###########################################
    ########### RLEnv_multi_opponent ##########
    ###########################################
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = RLEnv_multi_opponent(battle_format="gen8randombattle", opponent = opponent, start_challenging=True)

    x = train_env.reset()

    try:
        print('Checking RLEnv_multi_opponent Environment')
        for i in range(10):
            if i % 50 == 0:
                print(f'iteration: {i}')
            train_env.reset()
            check_env(train_env)
        train_env.close()
    except Exception as e:
        print(f'stopped at iteration: {i}')
        print(e)
        traceback.print_exc()
        train_env.close()
    except KeyboardInterrupt:
        train_env.close()
        exit(1)
    train_env.close()
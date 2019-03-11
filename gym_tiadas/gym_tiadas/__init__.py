from gym.envs.registration import register

register(
    id='russell-norvig-v0',
    entry_point='gym_tiadas.envs:RussellNorvig'
)

register(
    id='buridan-ass-v0',
    entry_point='gym_tiadas.envs:BuridanAss'
)

register(
    id='deep-sea-treasure-v0',
    entry_point='gym_tiadas.envs:DeepSeaTreasure'
)

register(
    id='resource-gathering-v0',
    entry_point='gym_tiadas.envs:ResourceGathering'
)

register(
    id='pressurized-bountiful-sea-treasure-v0',
    entry_point='gym_tiadas.envs:PressurizedBountifulSeaTreasure'
)

register(
    id='deep-sea-treasure-transactions-v0',
    entry_point='gym_tiadas.envs:DeepSeaTreasureTransactions'
)

register(
    id='mo-puddle-word-v0',
    entry_point='gym_tiadas.envs:MoPuddleWorld'
)

register(
    id='deep-sea-treasure-simplified-v0',
    entry_point='gym_tiadas.envs:DeepSeaTreasureSimplified'
)

register(
    id='bonus-world-v0',
    entry_point='gym_tiadas.envs:BonusWorld'
)

register(
    id='space-exploration-v0',
    entry_point='gym_tiadas.envs:SpaceExploration'
)

register(
    id='linked-rings-v0',
    entry_point='gym_tiadas.envs:LinkedRings'
)

register(
    id='non-recurrent-rings-v0',
    entry_point='gym_tiadas.envs:NonRecurrentRings'
)

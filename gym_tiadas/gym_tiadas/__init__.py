from gym.envs.registration import register

register(
    id='bonus-world-v0',
    entry_point='gym_tiadas.envs:BonusWorld'
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
    id='deep-sea-treasure-mixed-v0',
    entry_point='gym_tiadas.envs:DeepSeaTreasureMixed'
)

register(
    id='deep-sea-treasure-simplified-v0',
    entry_point='gym_tiadas.envs:DeepSeaTreasureSimplified'
)

register(
    id='deep-sea-treasure-stochastic-v0',
    entry_point='gym_tiadas.envs:DeepSeaTreasureStochastic'
)

register(
    id='linked-rings-v0',
    entry_point='gym_tiadas.envs:LinkedRings'
)

register(
    id='mo-puddle-word-v0',
    entry_point='gym_tiadas.envs:MoPuddleWorld'
)

register(
    id='non-recurrent-rings-v0',
    entry_point='gym_tiadas.envs:NonRecurrentRings'
)

register(
    id='pressurized-bountiful-sea-treasure-v0',
    entry_point='gym_tiadas.envs:PressurizedBountifulSeaTreasure'
)

register(
    id='resource-gathering-v0',
    entry_point='gym_tiadas.envs:ResourceGathering'
)

register(
    id='resource-gathering-limit-v0',
    entry_point='gym_tiadas.envs:ResourceGatheringLimit'
)

register(
    id='russell-norvig-v0',
    entry_point='gym_tiadas.envs:RussellNorvig'
)

register(
    id='space-exploration-v0',
    entry_point='gym_tiadas.envs:SpaceExploration'
)

from gym.envs.registration import register

register(
    id='russell-norvig-v0',
    entry_point='gym_tiadas.envs:RussellNorvig'
)

register(
    id='russell-norvig-discrete-v0',
    entry_point='gym_tiadas.envs:RussellNorvigDiscrete'
)

register(
    id='buridan-ass-v0',
    entry_point='gym_tiadas.envs:BuridanAss'
)

register(
    id='deep-sea-treasure-v0',
    entry_point='gym_tiadas.envs:DeepSeaTreasure'
)

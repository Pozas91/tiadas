from environments import ResourceGatheringEpisodic

nothing = (0, 0)
gold = (1, 0)
gem = (0, 1)
both = (1, 1)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

attacked = False

# Build instance of environment
env = ResourceGatheringEpisodic()

default_policies = {state: UP for state in env.states()}

policies: list = [
    # Green (1)
    {
        **default_policies,
        **{
            ((2, 4), nothing, attacked): RIGHT,
            ((3, 4), nothing, attacked): RIGHT,
            ((4, 4), nothing, attacked): UP,
            ((4, 3), nothing, attacked): UP,
            ((4, 2), nothing, attacked): UP,
            ((4, 1), gem, attacked): DOWN,
            ((4, 2), gem, attacked): DOWN,
            ((4, 3), gem, attacked): DOWN,
            ((4, 4), gem, attacked): LEFT,
            ((3, 4), gem, attacked): LEFT
        }
    },
    # Orange (2)
    {
        **default_policies,
        **{
            ((2, 4), nothing, attacked): RIGHT,
            ((3, 4), nothing, attacked): RIGHT,
            ((4, 4), nothing, attacked): UP,
            ((4, 3), nothing, attacked): UP,
            ((4, 2), nothing, attacked): UP,
            ((4, 1), gem, attacked): LEFT,
            ((3, 1), gem, attacked): LEFT,
            ((2, 1), gem, attacked): UP,
            ((2, 0), both, attacked): DOWN,
            ((2, 1), both, attacked): DOWN,
            ((2, 2), both, attacked): DOWN,
            ((2, 3), both, attacked): DOWN,
        }
    },
    # Light-blue (3)
    {
        **default_policies,
        **{
            ((2, 4), nothing, attacked): UP,
            ((2, 3), nothing, attacked): UP,
            ((2, 2), nothing, attacked): UP,
            ((2, 1), nothing, attacked): UP,
            ((2, 0), gold, attacked): DOWN,
            ((2, 1), gold, attacked): DOWN,
            ((2, 2), gold, attacked): DOWN,
            ((2, 3), gold, attacked): DOWN,
        }
    },
    # Blue (4)
    {
        **default_policies,
        **{
            ((2, 4), nothing, attacked): RIGHT,
            ((3, 4), nothing, attacked): RIGHT,
            ((4, 4), nothing, attacked): UP,
            ((4, 3), nothing, attacked): UP,
            ((4, 2), nothing, attacked): UP,
            ((4, 1), gem, attacked): LEFT,
            ((3, 1), gem, attacked): LEFT,
            ((2, 1), gem, attacked): UP,
            ((2, 0), both, attacked): LEFT,
            ((1, 0), both, attacked): DOWN,
            ((1, 1), both, attacked): DOWN,
            ((1, 2), both, attacked): DOWN,
            ((1, 3), both, attacked): DOWN,
            ((1, 4), both, attacked): RIGHT,
        }
    },
    # Pink (5)
    {
        **default_policies,
        **{
            ((2, 4), nothing, attacked): LEFT,
            ((1, 4), nothing, attacked): UP,
            ((1, 3), nothing, attacked): UP,
            ((1, 2), nothing, attacked): UP,
            ((1, 1), nothing, attacked): UP,
            ((1, 0), nothing, attacked): RIGHT,
            ((2, 0), gold, attacked): LEFT,
            ((1, 0), gold, attacked): DOWN,
            ((1, 1), gold, attacked): DOWN,
            ((1, 2), gold, attacked): DOWN,
            ((1, 3), gold, attacked): DOWN,
            ((1, 4), gold, attacked): RIGHT,
        }
    },
    # Grey (6)
    {
        **default_policies,
        **{
            ((2, 4), nothing, attacked): LEFT,
            ((1, 4), nothing, attacked): UP,
            ((1, 3), nothing, attacked): UP,
            ((1, 2), nothing, attacked): UP,
            ((1, 1), nothing, attacked): UP,
            ((1, 0), nothing, attacked): RIGHT,
            ((2, 0), gold, attacked): DOWN,
            ((2, 1), gold, attacked): DOWN,
            ((2, 2), gold, attacked): DOWN,
            ((2, 3), gold, attacked): DOWN,
        }
    },
]

from environments import ResourceGatheringEpisodic, ResourceGathering

nothing = (0, 0)
gold = (1, 0)
gem = (0, 1)
both = (1, 1)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Build instance of environment
env = ResourceGathering()

default_policies = {state: UP for state in env.states()}

policies: list = [
    # Green (1)
    {
        **default_policies,
        **{
            ((2, 4), nothing): RIGHT,
            ((3, 4), nothing): RIGHT,
            ((4, 4), nothing): UP,
            ((4, 3), nothing): UP,
            ((4, 2), nothing): UP,
            ((4, 1), gem): DOWN,
            ((4, 2), gem): DOWN,
            ((4, 3), gem): DOWN,
            ((4, 4), gem): LEFT,
            ((3, 4), gem): LEFT
        }
    },
    # Orange (2)
    {
        **default_policies,
        **{
            ((2, 4), nothing): RIGHT,
            ((3, 4), nothing): RIGHT,
            ((4, 4), nothing): UP,
            ((4, 3), nothing): UP,
            ((4, 2), nothing): UP,
            ((4, 1), gem): LEFT,
            ((3, 1), gem): LEFT,
            ((2, 1), gem): UP,
            ((2, 0), both): DOWN,
            ((2, 1), both): DOWN,
            ((2, 2), both): DOWN,
            ((2, 3), both): DOWN,
        }
    },
    # Light-blue (3)
    {
        **default_policies,
        **{
            ((2, 4), nothing): UP,
            ((2, 3), nothing): UP,
            ((2, 2), nothing): UP,
            ((2, 1), nothing): UP,
            ((2, 0), gold): DOWN,
            ((2, 1), gold): DOWN,
            ((2, 2), gold): DOWN,
            ((2, 3), gold): DOWN,
        }
    },
    # Blue (4)
    {
        **default_policies,
        **{
            ((2, 4), nothing): RIGHT,
            ((3, 4), nothing): RIGHT,
            ((4, 4), nothing): UP,
            ((4, 3), nothing): UP,
            ((4, 2), nothing): UP,
            ((4, 1), gem): LEFT,
            ((3, 1), gem): LEFT,
            ((2, 1), gem): UP,
            ((2, 0), both): LEFT,
            ((1, 0), both): DOWN,
            ((1, 1), both): DOWN,
            ((1, 2), both): DOWN,
            ((1, 3), both): DOWN,
            ((1, 4), both): RIGHT,
        }
    },
    # Pink (5)
    {
        **default_policies,
        **{
            ((2, 4), nothing): LEFT,
            ((1, 4), nothing): UP,
            ((1, 3), nothing): UP,
            ((1, 2), nothing): UP,
            ((1, 1), nothing): UP,
            ((1, 0), nothing): RIGHT,
            ((2, 0), gold): LEFT,
            ((1, 0), gold): DOWN,
            ((1, 1), gold): DOWN,
            ((1, 2), gold): DOWN,
            ((1, 3), gold): DOWN,
            ((1, 4), gold): RIGHT,
        }
    },
    # Grey (6)
    {
        **default_policies,
        **{
            ((2, 4), nothing): LEFT,
            ((1, 4), nothing): UP,
            ((1, 3), nothing): UP,
            ((1, 2), nothing): UP,
            ((1, 1), nothing): UP,
            ((1, 0), nothing): RIGHT,
            ((2, 0), gold): DOWN,
            ((2, 1), gold): DOWN,
            ((2, 2), gold): DOWN,
            ((2, 3), gold): DOWN,
        }
    },
]

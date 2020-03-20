from agents import AgentW
from agents.agent_bn import AgentBN
from environments import DeepSeaTreasureRightDown
from models import Vector


def main():
    environment = DeepSeaTreasureRightDown(columns=1)

    agent = AgentW(environment=environment, seed=0)

    agent.train(limit=1)

    agent.save()


def load():
    # agent_10 = AgentBN.load(
    #     'bn/models/rge_1583857516_0.01.bin'
    # )

    # sorted_v_s_0_10 = sorted(agent_10.v[((2, 4), (0, 0))], key=lambda k: k[0])

    # agent_15 = AgentBN.load(
    #     'bn/models/rge_1583857532_0.01.bin'
    # )

    # sorted_v_s_0_15 = sorted(agent_15.v[((2, 4), (0, 0))], key=lambda k: k[0])

    agent_30: AgentBN = AgentBN.load(
        'bn/models/rge_1583857678_0.01.bin'
    )

    v_s_0_30 = agent_30.v[((2, 4), (0, 0))]
    v_s_0_30_nd = Vector.m3_max(set(v_s_0_30))

    # agent_10625 = AgentBN.load(
    #     filename='bn/models/rge_1583924116_0.01.bin'
    # )

    # sorted_v_s_0_10625 = sorted(agent_10625.v[((2, 4), (0, 0))], key=lambda k: k[0])

    pass


if __name__ == '__main__':
    load()

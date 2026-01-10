from dataclasses import dataclass
from typing import List, Sequence

from .agent import Agent
from .policy import Policy


@dataclass
class SharedPolicyMultiAgent:
    policy: Policy
    num_agents: int

    def act_all(self, observations: Sequence) -> List:
        if len(observations) != self.num_agents:
            raise ValueError("observations length must match num_agents")
        return [self.policy.act(obs) for obs in observations]

    def observe_all(self, transitions: Sequence[tuple]) -> None:
        if len(transitions) != self.num_agents:
            raise ValueError("transitions length must match num_agents")
        # Shared policy has no per-agent buffer by default.
        return None


@dataclass
class IndependentMultiAgent:
    agents: List[Agent]

    def act_all(self, observations: Sequence) -> List:
        if len(observations) != len(self.agents):
            raise ValueError("observations length must match agents length")
        return [agent.act(obs) for agent, obs in zip(self.agents, observations)]

    def observe_all(self, transitions: Sequence[tuple]) -> None:
        if len(transitions) != len(self.agents):
            raise ValueError("transitions length must match agents length")
        for agent, transition in zip(self.agents, transitions):
            agent.observe(transition)

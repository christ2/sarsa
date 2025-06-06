from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.model = nn.Sequential(
            nn.Linear(state_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> int:
        s = torch.FloatTensor(s)
        probabilities = self.model(s).detach().numpy()
        return np.random.choice(len(probabilities), p=probabilities)

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        s = torch.FloatTensor(s)
        log_prob = torch.log(self.model(s)[a])
        loss = -log_prob * gamma_t * delta
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.model = nn.Sequential(
            nn.Linear(state_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> float:
        s = torch.FloatTensor(s)
        return self.model(s).item()

    def update(self,s,G):
        s = torch.FloatTensor(s)
        G = torch.FloatTensor([G])
        loss = nn.MSELoss()(self.model(s), G)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    G_0_list = []
    for _ in range(num_episodes):
        trajectory = []
        s = env.reset()
        done = False

        while not done:
            a = pi(s)
            s_next, r, done, _ = env.step(a)
            trajectory.append((s, a, r))
            s = s_next

        G = 0
        for t in reversed(range(len(trajectory))):
            s, a, r = trajectory[t]
            G = r + gamma * G
            if isinstance(V, VApproximationWithNN):
                delta = G - V(s)
                V.update(s, G)
            else:
                delta = G
            pi.update(s, a, gamma ** t, delta)

        G_0_list.append(trajectory[0][2])
    return G_0_list
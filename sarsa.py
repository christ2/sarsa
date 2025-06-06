import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = np.array(state_low)
        self.state_high = np.array(state_high)
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = np.array(tile_width)
        self.num_tiles = np.ceil((self.state_high - self.state_low) / self.tile_width).astype(int) + 1
        self.offsets = [
            - (i / num_tilings) * self.tile_width for i in range(num_tilings)
        ]

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tilings * np.prod(self.num_tiles)

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros(self.feature_vector_len())

        indices = []
        for i, offset in enumerate(self.offsets):
            coords = np.floor((s - (self.state_low + offset)) / self.tile_width).astype(int)
            indices.append(np.ravel_multi_index(coords, self.num_tiles))

        feature_vector = np.zeros(self.feature_vector_len())
        for i, tile_index in enumerate(indices):
            feature_index = a * (self.num_tilings * np.prod(self.num_tiles)) + i * np.prod(self.num_tiles) + tile_index
            feature_vector[feature_index] = 1
        return feature_vector

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    for _ in range(num_episode):
        s, done = env.reset(), False
        a = epsilon_greedy_policy(s, done, w, epsilon=0.1)
        x = X(s, done, a)
        z = np.zeros_like(w)
        Q_old = 0

        while not done:
            s_next, r, done, _ = env.step(a)
            a_next = epsilon_greedy_policy(s_next, done, w, epsilon=0.1)
            x_next = X(s_next, done, a_next)
            Q = np.dot(w, x)
            Q_next = np.dot(w, x_next)

            delta = r + (gamma * Q_next * (not done)) - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_next
            x, a = x_next, a_next
    return w
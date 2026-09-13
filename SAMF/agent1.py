import numpy as np

class MultiAgentController:
    """
    Minimal multi-agent controller (retains the original random strategy),
    supports n_agents > 1, action_dim = 3 (w_recon, w_mean, w_tri).
    Provides select_actions(), observe(obs), update(rewards)
    """
    def __init__(self, n_agents=3, action_dim=3):
        self.n_agents = n_agents
        self.action_dim = action_dim
        # Each agent controls three weight parameters [w_recon, w_mean, w_tri]
        # Initialized to reasonable mid-range values (e.g., 1.0)
        self.actions = np.ones((n_agents, action_dim)) * 0.5
        self.last_rewards = np.zeros(n_agents)
        self.last_obs = np.zeros(n_agents)  # per-slice scalar obs

    def select_actions(self):
        # Temporary strategy: random perturbation in [-0.3, +0.3] range (consistent with original impl)
        delta = np.random.uniform(-0.3, 0.3, size=(self.n_agents, self.action_dim))
        self.actions += delta
        # Clamp each parameter to a reasonable range
        self.actions = np.clip(self.actions, 0.1, 2.0)
        return self.actions.copy()  # Return proposals, shape (n_agents, 3)

    def observe(self, obs):
        """
        Receive per-slice observations (array of length n_agents), e.g., per-slice recon error
        """
        try:
            obs = np.asarray(obs)
            if obs.shape[0] == self.n_agents:
                self.last_obs = obs.copy()
        except Exception:
            pass

    def update(self, rewards):
        """
        rewards: array-like shape (n_agents,) or a scalar broadcastable
        Currently simple recording; can be replaced with RL algorithms (PPO/MADDPG, etc.)
        """
        rewards = np.asarray(rewards)
        if rewards.size == 1:
            rewards = np.repeat(rewards, self.n_agents)
        self.last_rewards = rewards.copy()
        # Simple heuristic: slightly increase actions for agents with positive reward, decrease otherwise
        lr = 0.01
        for i in range(self.n_agents):
            r = rewards[i]
            # Use the sign of reward as the fine-tuning direction for actions (a very simple heuristic)
            self.actions[i] += lr * r
        # clip again
        self.actions = np.clip(self.actions, 0.1, 2.0)

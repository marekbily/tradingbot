"""
DreamerV3 Components for Trading
Based on: https://github.com/danijar/dreamerv3
Paper: https://arxiv.org/abs/2301.04104

Core Architecture:
1. RSSM (Recurrent State Space Model) - learns latent market dynamics
2. Encoder - compresses observations into embeddings
3. Dynamics - predicts next latent state
4. Decoder - reconstructs observations from latent
5. Reward Predictor - predicts rewards in latent space
6. Actor-Critic - policy and value networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np


def symlog(x):
    """Symlog transformation - squashes large values while preserving small ones"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    """Inverse of symlog"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / norm


class GRUCell(nn.Module):
    """GRU Cell with RMSNorm"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Reset, update, and new gates
        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_r = RMSNorm(hidden_size)

        self.W_iz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_z = RMSNorm(hidden_size)

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_n = RMSNorm(hidden_size)

    def forward(self, x, h):
        r = torch.sigmoid(self.norm_r(self.W_ir(x) + self.W_hr(h)))
        z = torch.sigmoid(self.norm_z(self.W_iz(x) + self.W_hz(h)))
        n = torch.tanh(self.norm_n(self.W_in(x) + self.W_hn(r * h)))
        h_new = (1 - z) * n + z * h
        return h_new


class Encoder(nn.Module):
    """Encodes observations into embeddings.

    Two modes:
    - Flat: original MLP that takes a flattened observation vector.
    - Temporal pooled: when `window` and `feats` are provided, the encoder
      applies a small MLP per time-step (shared), pools across the window,
      and projects to `embed_dim`. This drastically reduces compute for
      large flattened inputs like (window * num_features).
    """
    def __init__(self, obs_dim, embed_dim=256, window: int = None, feats: int = None):
        super().__init__()
        self.window = window
        self.feats = feats
        self.embed_dim = embed_dim

        # If window and feats provided and match obs_dim, use temporal pooled encoder
        if self.window is not None and self.feats is not None and (self.window * self.feats + 1) == obs_dim:
            # per-time-step embedding
            per_hidden = 128
            self.temporal = True
            self.per_step = nn.Sequential(
                nn.Linear(self.feats, per_hidden),
                RMSNorm(per_hidden),
                nn.SiLU(),
            )
            # aggregate pooled embedding to final embed_dim
            self.aggregate = nn.Sequential(
                nn.Linear(per_hidden, embed_dim),
                RMSNorm(embed_dim),
                nn.SiLU(),
            )
        else:
            self.temporal = False
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 512),
                RMSNorm(512),
                nn.SiLU(),
                nn.Linear(512, 512),
                RMSNorm(512),
                nn.SiLU(),
                nn.Linear(512, embed_dim),
                RMSNorm(embed_dim),
            )

    def forward(self, obs):
        # Apply symlog to observations for stability
        obs = symlog(obs)
        if self.temporal:
            # obs: (N, obs_dim) where obs_dim = window * feats + 1 (position)
            # split features and position
            feats_flat = obs[:, : self.window * self.feats]
            # reshape to (N, window, feats) using reshape (safe for torch.compile)
            feats_seq = feats_flat.reshape(-1, self.window, self.feats)
            # apply per-step MLP
            N = feats_seq.shape[0]
            per = self.per_step(feats_seq.reshape(-1, self.feats))
            per = per.view(N, self.window, -1)
            # pool across time
            pooled = per.mean(dim=1)
            out = self.aggregate(pooled)
            return out
        else:
            return self.net(obs)


class RSSM(nn.Module):
    """
    Recurrent State Space Model

    The "World Model" that learns market dynamics:
    - h_t (deterministic): GRU hidden state (memory)
    - z_t (stochastic): latent state (current market regime)
    """
    def __init__(self, embed_dim=256, hidden_dim=512, stoch_dim=32, num_categories=32, action_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.num_categories = num_categories
        self.action_dim = action_dim

        # Prior: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, stoch_dim * num_categories)
        )

        # Posterior: q(z_t | h_t, e_t) where e_t is encoded observation
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, stoch_dim * num_categories)
        )

        # Dynamics: use a cuDNN-backed GRU for vectorized recurrence
        # stoch_dim * num_categories because z is flattened one-hot
        self.gru = nn.GRU(input_size=stoch_dim * num_categories + action_dim, hidden_size=hidden_dim, batch_first=True)

    def initial_state(self, batch_size, device):
        """Initialize h_0 and z_0"""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        # z is flattened one-hot, so it's stoch_dim * num_categories
        z = torch.zeros(batch_size, self.stoch_dim * self.num_categories, device=device)
        return h, z

    def observe_sequence(self, embed_seq, action_seq, h0=None, n_iter=2):
        """
        Vectorized observation over a full sequence using iterative posterior refinement.

        Args:
            embed_seq: (B, T, embed_dim)
            action_seq: (B, T, action_dim)
            h0: (B, hidden_dim) initial hidden state, zeros if None
            n_iter: number of refinement passes (2 is a good tradeoff)

        Returns:
            h_seq: (B, T, hidden_dim)
            z_flat: (B, T, stoch_dim * num_categories) -- soft posterior probs
            prior_logits: (B, T, stoch_dim, num_categories)
            posterior_logits: (B, T, stoch_dim, num_categories)
        """
        B, T, _ = embed_seq.shape
        device = embed_seq.device

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_dim, device=device)

        # initialize soft z to zeros
        z_shape = (B, T, self.stoch_dim * self.num_categories)
        z_soft = torch.zeros(z_shape, device=device)

        for _ in range(n_iter):
            # build z_prev_shift where for time t we provide z_{t-1} (zero for t=0)
            z_prev_shift = torch.zeros_like(z_soft)
            z_prev_shift[:, 1:, :] = z_soft[:, :-1, :]

            # inputs: concat(z_prev_shift, actions) -> (B, T, input_size)
            inputs = torch.cat([z_prev_shift, action_seq], dim=-1)

            # run GRU over the whole sequence
            # GRU expects input (B, T, input_size). initial hidden must be (num_layers, B, hidden)
            h_seq, _ = self.gru(inputs, h0.unsqueeze(0))

            # compute posterior logits from h_seq and embeds
            hp = torch.cat([h_seq, embed_seq], dim=-1).reshape(B * T, -1)
            posterior_logits = self.posterior_net(hp).reshape(B, T, self.stoch_dim, self.num_categories)

            # soft posterior (probabilities)
            posterior_probs = torch.softmax(posterior_logits, dim=-1)
            z_soft = posterior_probs.reshape(B, T, -1)

        # final prior logits from h_seq
        prior_logits = self.prior_net(h_seq.reshape(B * T, -1)).reshape(B, T, self.stoch_dim, self.num_categories)

        return h_seq, z_soft, prior_logits, posterior_logits

    def observe(self, embed, action, h_prev, z_prev):
        """
        Posterior inference: q(z_t | h_t, e_t)
        Used during training with real observations
        """
        # Update deterministic state using GRU for a single step.
        # GRU returns (output_seq, h_n) when used; unpack correctly.
        inp = torch.cat([z_prev, action], dim=-1).unsqueeze(1)  # (B, 1, input_size)
        out_seq, h_n = self.gru(inp, h_prev.unsqueeze(0))
        # h_n: (num_layers=1, B, hidden_dim) -> squeeze to (B, hidden_dim)
        h = h_n.squeeze(0)

        # Compute posterior distribution
        posterior_logits = self.posterior_net(torch.cat([h, embed], dim=-1))
        posterior_logits = posterior_logits.reshape(-1, self.stoch_dim, self.num_categories)

        # Sample z_t from categorical distribution
        z = self._sample_categorical(posterior_logits)

        # Also compute prior for KL regularization
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.reshape(-1, self.stoch_dim, self.num_categories)

        return h, z, prior_logits, posterior_logits

    def imagine(self, action, h_prev, z_prev):
        """
        Prior imagination: p(z_t | h_t)
        Used for dreaming/planning without real observations
        """
        # Update deterministic state for a single step using GRU
        inp = torch.cat([z_prev, action], dim=-1).unsqueeze(1)
        out_seq, h_n = self.gru(inp, h_prev.unsqueeze(0))
        h = h_n.squeeze(0)

        # Sample from prior
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.reshape(-1, self.stoch_dim, self.num_categories)
        z = self._sample_categorical(prior_logits)

        return h, z, prior_logits

    def imagine_sequence(self, h0, z0, horizon, actor, reward_predictor, critic, deterministic=False):
        """
        Vectorized imagination loop executed inside RSSM to reduce Python overhead.

        Args:
            h0, z0: starting latent states (B, hidden), (B, z_flat)
            horizon: number of imagine steps
            actor: callable actor module (expects state -> logits)
            reward_predictor: reward module
            critic: value module
            deterministic: if True use greedy actions

        Returns: states, rewards, actor_logits, values
        """
        B = h0.shape[0]
        device = h0.device
        state_dim = h0.shape[1] + z0.shape[1]

        states = torch.empty((B, horizon, state_dim), device=device)
        rewards = torch.empty((B, horizon), device=device)
        actor_dim = getattr(actor, 'action_dim', None)
        if actor_dim is None and hasattr(actor, 'net'):
            # try to infer from last linear layer
            try:
                actor_dim = actor.net[-1].out_features
            except Exception:
                actor_dim = 3
        actor_logits = torch.empty((B, horizon, actor_dim), device=device)

        h = h0
        z = z0

        for t in range(horizon):
            state = self.get_state(h, z)
            states[:, t, :] = state

            # Detach state before passing to reward/actor so their graphs do not
            # connect back to the RSSM/world-model. This avoids cross-phase
            # gradient conflicts when we run multiple backward() passes.
            state_det = state.detach()
            reward_pred = reward_predictor(state_det)
            rewards[:, t] = symexp(reward_pred)

            logits = actor(state_det)
            actor_logits[:, t, :] = logits

            # sample or take greedy action
            if deterministic:
                action = F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).float()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                idx = dist.sample()
                action = F.one_hot(idx, logits.shape[-1]).float()

            # imagine next state using prior
            h, z, _ = self.imagine(action, h, z)

        # compute values in one batch
        values = critic(states.reshape(B * horizon, -1)).reshape(B, horizon)

        return states, rewards, actor_logits, values

    def _sample_categorical(self, logits):
        """Sample from categorical distribution with straight-through estimator"""
        # Sample during training, use mode during eval
        if self.training:
            # Gumbel-Softmax trick
            dist = torch.distributions.OneHotCategorical(logits=logits)
            z_one_hot = dist.sample()
        else:
            # Use argmax (mode) during evaluation
            z_one_hot = F.one_hot(logits.argmax(dim=-1), self.num_categories).float()

        # Flatten the one-hot vectors
        return z_one_hot.reshape(-1, self.stoch_dim * self.num_categories)

    def get_state(self, h, z):
        """Concatenate h and z for full latent state"""
        return torch.cat([h, z], dim=-1)

    def kl_loss(self, prior_logits, posterior_logits, free_nats=1.0, balance=0.8):
        """
        KL divergence with free bits and balancing
        Prevents posterior collapse
        """
        prior = torch.distributions.Categorical(logits=prior_logits)
        posterior = torch.distributions.Categorical(logits=posterior_logits)

        # KL divergence
        kl = torch.distributions.kl_divergence(posterior, prior)

        # Free nats: don't penalize KL below this threshold
        kl = torch.maximum(kl, torch.tensor(free_nats / self.stoch_dim, device=kl.device))

        # KL balancing: mix between treating prior/posterior as constant
        kl_balanced = balance * kl + (1 - balance) * kl.detach()

        return kl_balanced.sum(dim=-1).mean()


class Decoder(nn.Module):
    """Reconstructs observations from latent state"""
    def __init__(self, state_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, obs_dim),
        )

    def forward(self, state):
        """Returns mean of Gaussian distribution"""
        return symexp(self.net(state))


class RewardPredictor(nn.Module):
    """Predicts rewards from latent state"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        """Returns symlog-transformed reward prediction"""
        return self.net(state).squeeze(-1)


class Actor(nn.Module):
    """Policy network - outputs action distribution"""
    def __init__(self, state_dim, action_dim=3):
        super().__init__()
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, state):
        """Returns action logits for categorical distribution"""
        return self.net(state)

    def sample(self, state, deterministic=False):
        """Sample action from policy"""
        logits = self(state)
        if deterministic:
            action = F.one_hot(logits.argmax(dim=-1), self.action_dim).float()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()
            action = F.one_hot(action_idx, self.action_dim).float()
        return action


class Critic(nn.Module):
    """Value network - estimates state value"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        """Returns symlog-transformed value prediction"""
        return self.net(state).squeeze(-1)


if __name__ == "__main__":
    # Quick sanity check
    print("Testing DreamerV3 components...")

    batch_size = 4
    obs_dim = 64 * 11  # window * features
    device = 'cpu'

    # Test encoder
    encoder = Encoder(obs_dim, embed_dim=256)
    obs = torch.randn(batch_size, obs_dim)
    embed = encoder(obs)
    print(f"✅ Encoder: {obs.shape} -> {embed.shape}")

    # Test RSSM
    rssm = RSSM(embed_dim=256, hidden_dim=512, stoch_dim=32, num_categories=32)
    h, z = rssm.initial_state(batch_size, device)
    action = F.one_hot(torch.randint(0, 3, (batch_size,)), 3).float()
    h_new, z_new, prior, posterior = rssm.observe(embed, action, h, z)
    print(f"✅ RSSM: h={h_new.shape}, z={z_new.shape}")

    # Test decoder
    state = rssm.get_state(h_new, z_new)
    decoder = Decoder(state.shape[-1], obs_dim)
    obs_recon = decoder(state)
    print(f"✅ Decoder: {state.shape} -> {obs_recon.shape}")

    # Test reward predictor
    reward_pred = RewardPredictor(state.shape[-1])
    reward = reward_pred(state)
    print(f"✅ Reward Predictor: {state.shape} -> {reward.shape}")

    # Test actor-critic
    actor = Actor(state.shape[-1], action_dim=3)
    critic = Critic(state.shape[-1])
    action_logits = actor(state)
    value = critic(state)
    print(f"✅ Actor: {state.shape} -> {action_logits.shape}")
    print(f"✅ Critic: {state.shape} -> {value.shape}")

    print("\n🎉 All components working!")

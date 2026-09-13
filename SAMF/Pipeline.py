import os
import random

import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from tqdm import tqdm
from torch.backends import cudnn
import pandas as pd

from .Models import SAMF_model
from .GLNS import GLNSampler, GLNSampler_BC

from .agent1 import MultiAgentController


class SC_pipeline:
    def __init__(self, adata, edge_index, num_clusters, device, config, roundseed=0, imputation=False):
        # --- Retain all existing initialization settings ---
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.edge_index = edge_index
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters
        self.imputation = imputation

        if self.imputation:
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        self.edge_index = self.edge_index.to(self.device)

        self.input_dim = self.X.shape[-1]
        self.model = SAMF_model(self.input_dim, self.model_config, imputation=self.imputation).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=0.001,
            weight_decay=3e-4,
        )

        self.sampler = GLNSampler(self.num_clusters, self.device)
        self.anchor_pair = None

        # === Initialize multi-agent (replaced with per-slice agent) ===
        self.use_agent = config.get('use_agent', True)
        if self.use_agent:
            # Obtain slice category from adata.obs['slice_name'] (label used by graph_construction3D)
            if 'slice_name' in self.adata.obs:
                # Ensure it is categorical
                if not pd.api.types.is_categorical_dtype(self.adata.obs['slice_name']):
                    self.adata.obs['slice_name'] = self.adata.obs['slice_name'].astype('category')
                slice_cats = list(self.adata.obs['slice_name'].cat.categories)
                self.n_slices = len(slice_cats)
                # spot -> slice idx mapping
                cat_to_idx = {c: i for i, c in enumerate(slice_cats)}
                self.spot2slice = np.array([cat_to_idx[s] for s in self.adata.obs['slice_name'].values])
            else:
                # fallback: treat everything as one slice
                self.n_slices = 1
                self.spot2slice = np.zeros(self.adata.n_obs, dtype=int)

            # Use multi-agent controller (one agent per slice)
            self.agent = MultiAgentController(n_agents=self.n_slices, action_dim=3)
            print(f"✅ Multi-Agent Controller initialized with {self.n_slices} agents.")

    def trian(self):
        neighbors = self.train_config['topk_neighs']
        pbar = tqdm(range(self.train_config['epochs']))

        # config tunables (you can set these in config['train'])
        eval_every = self.train_config.get('agent_eval_every', 5)  # how often to compute ARI / per-slice obs
        label_key = self.train_config.get('label_key', 'Ground Truth')  # ground truth column in adata.obs
        pred_key = self.train_config.get('pred_key', 'spacross_pred')  # where to store kmeans preds
        lambda_loss = self.train_config.get('agent_lambda_loss', 0.1)  # tradeoff between ARI gain and loss
        w_min, w_max = self.train_config.get('agent_w_bounds', (0.5, 2.0))  # bounds for w

        prev_ari = None

        for epoch in pbar:

            # === agent propose & aggregate ===
            if self.use_agent:
                proposals = self.agent.select_actions()  # shape (n_agents, action_dim) or (1,3) for single-agent
                # if single-agent controller returned shape (1,3), handle both cases
                proposals = np.asarray(proposals)
                if proposals.ndim == 1:
                    proposals = proposals.reshape(1, -1)

                # aggregation: simple mean across agents -> global weights
                agg = np.mean(proposals, axis=0)
                # clip into bounds to avoid degenerate values
                agg = np.clip(agg, w_min, w_max)
                self.train_config['w_recon'] = float(agg[0])
                self.train_config['w_mean'] = float(agg[1])
                self.train_config['w_tri'] = float(agg[2])

            # === anchor update (same as before) ===
            if epoch % self.train_config['t_step'] == 0 and epoch > 1:
                self.model.eval()
                s_rep, t_rep = self.model.std_tgt_embedding(self.X, self.edge_index)
                self.anchor_pair = self.sampler(self.edge_index,
                                                F.normalize(s_rep, dim=-1, p=2),
                                                F.normalize(t_rep, dim=-1, p=2),
                                                neighbors, cluster_method="kmeans")

            # === training step (global on full 3D graph) ===
            self.model.train()
            self.optimizer.zero_grad()
            mean_loss, rec_loss, tri_loss, causal_loss = self.model(self.X, self.edge_index, self.anchor_pair)
            loss = (self.train_config['w_recon'] * rec_loss +
                    self.train_config['w_mean'] * mean_loss +
                    self.train_config['w_tri'] * tri_loss +
                    0.1 * causal_loss)  # Causal loss weight set to 0.1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            with torch.no_grad():
                self.model.momentum_update()

            # === periodically compute per-slice obs and global ARI (for agent input & reward) ===
            ari_now = None
            per_slice_obs = None
            if self.use_agent and (epoch % eval_every == 0):
                self.model.eval()
                with torch.no_grad():
                    enc_rep, recon = self.model.evaluate(self.X, self.edge_index)
                    enc_np = enc_rep.to('cpu').numpy()
                    recon_np = recon.to('cpu').numpy()
                    X_cpu = self.X.to('cpu').numpy()

                    # compute per-spot reconstruction error (fallback to latent norm if shapes mismatch)
                    if recon_np.ndim == 2 and X_cpu.ndim == 2 and recon_np.shape == X_cpu.shape:
                        per_spot_err = np.mean((recon_np - X_cpu) ** 2, axis=1)
                    else:
                        per_spot_err = np.linalg.norm(enc_np, axis=1)

                    # aggregate to per-slice mean error if slice_name exists
                    if hasattr(self, 'spot2slice'):
                        per_slice_mean_err = []
                        for s in range(self.n_slices):
                            idxs = np.where(self.spot2slice == s)[0]
                            if len(idxs) == 0:
                                per_slice_mean_err.append(0.0)
                            else:
                                per_slice_mean_err.append(float(np.mean(per_spot_err[idxs])))
                        per_slice_obs = np.array(per_slice_mean_err)
                    else:
                        per_slice_obs = np.array([float(np.mean(per_spot_err))])

                    # supply observation to agent if supported
                    if hasattr(self.agent, 'observe'):
                        try:
                            self.agent.observe(per_slice_obs)
                        except Exception:
                            pass

                    # build predictions via KMeans on enc_rep to compute ARI
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(enc_np)
                    preds = kmeans.labels_
                    # put preds into adata.obs for get_metrics
                    self.adata.obs[pred_key] = pd.Categorical(preds.astype(str))

                    # compute ARI using your helper get_metrics (returns ARI first)
                    try:
                        ari_now, _, _ = get_metrics(self.adata, label_key, pred_key)
                    except Exception:
                        # fallback: try compute_ARI directly if available
                        try:
                            ari_now = compute_ARI(self.adata, label_key, pred_key)
                        except Exception:
                            ari_now = None

            # === compute reward and update agent(s) ===
            if self.use_agent:
                # default: if ARI available, use delta ARI; else fallback to -loss (but prefer ARI)
                if ari_now is not None and prev_ari is not None:
                    delta_ari = ari_now - prev_ari
                    # combine ARI gain with a small penalty on raw loss (normalized)
                    # normalize loss by (1 + abs(loss)) to keep scale bounded
                    loss_norm = float(loss.item()) / (1.0 + abs(float(loss.item())))
                    reward_scalar = float(delta_ari) - lambda_loss * loss_norm
                elif ari_now is not None and prev_ari is None:
                    # first ARI measurement: give small reward equal to current ARI
                    loss_norm = float(loss.item()) / (1.0 + abs(float(loss.item())))
                    reward_scalar = float(ari_now) - lambda_loss * loss_norm
                else:
                    # fallback: if ARI can't be computed, use -loss (legacy)
                    reward_scalar = -float(loss.item())

                # For cooperative setting: broadcast same reward to all agents
                if getattr(self.agent, 'n_agents', 1) > 1:
                    rewards = np.repeat(reward_scalar, getattr(self.agent, 'n_agents'))
                else:
                    rewards = np.array([reward_scalar])

                # update agent
                try:
                    self.agent.update(rewards)
                except Exception:
                    # backward compatibility: if agent.update expects scalar
                    try:
                        self.agent.update(reward_scalar)
                    except Exception:
                        pass

                # update prev_ari if available
                if ari_now is not None:
                    prev_ari = ari_now

                # logging
                print(f"[Agent] Epoch {epoch}: w_recon={self.train_config['w_recon']:.4f}, "
                      f"w_mean={self.train_config['w_mean']:.4f}, w_tri={self.train_config['w_tri']:.4f}, "
                      f"reward={reward_scalar:.4f}, ari={ari_now}")

            pbar.set_description(
                "Epoch {0} total loss={1:.3f} recon loss={2:.3f} mean loss={3:.3f} tri loss={4:.3f}".format(
                    epoch, loss, rec_loss, mean_loss, tri_loss),
                refresh=True)

    def process(self):
        self.model.eval()
        enc_rep, recon = self.model.evaluate(self.X, self.edge_index)
        enc_rep = enc_rep.to('cpu').detach().numpy()
        recon = recon.to('cpu').detach().numpy()
        recon[recon < 0] = 0

        self.adata.obsm['latent'] = enc_rep
        self.adata.obsm['ReX'] = recon
        return enc_rep, recon



class SC_BC_pipeline:
    def __init__(self, adata, edge_index, num_clusters, device, config, roundseed=0, imputation=False):
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.edge_index = edge_index
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters
        self.imputation = imputation
        self.batch_id = torch.tensor(adata.obs['slice_id'].to_numpy(), dtype=torch.float32)

        self.agent = MultiAgentController(n_agents=len(set(adata.obs['slice_id'])), action_dim=3)
        # ↑ Create corresponding agents based on slice count, each controlling three loss weights

        if self.imputation:
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        self.edge_index = self.edge_index.to(self.device)

        self.input_dim = self.X.shape[-1]
        self.model = SAMF_model(self.input_dim, self.model_config, imputation=self.imputation).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=0.001,
            weight_decay=3e-4,
        )

        self.sampler = GLNSampler_BC(self.num_clusters, self.device)
        self.anchor_pair = None

    def trian(self):
        neighbors = self.train_config['topk_neighs']
        neighbors_inter = self.train_config['topk_neighs_inter']
        pbar = tqdm(range(self.train_config['epochs']))

        for epoch in pbar:

            # ======== 🧠 New: agent decision step ========
            actions = self.agent.select_actions()  # Each agent generates actions
            # Map each agent's actions to loss weights
            avg_action = actions.mean(axis=0)
            self.train_config['w_recon'] = float(avg_action[0])
            self.train_config['w_mean'] = float(avg_action[1])
            self.train_config['w_tri'] = float(avg_action[2])
            # ========================================



            if epoch % self.train_config['t_step'] == 0 and epoch > 1:
                self.model.eval()
                s_rep, t_rep = self.model.std_tgt_embedding(self.X, self.edge_index)
                # (self, adj, enc_rep, batch_id, top_k, top_k_inter, cluster_method="kmeans")
                self.anchor_pair = self.sampler(self.edge_index, F.normalize(s_rep, dim=-1, p=2), self.batch_id, neighbors, neighbors_inter, cluster_method="kmeans")

            self.model.train()
            self.optimizer.zero_grad()
            mean_loss, rec_loss, tri_loss = self.model(self.X, self.edge_index, self.anchor_pair)
            loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_mean'] * mean_loss + \
                   self.train_config['w_tri'] * tri_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            with torch.no_grad():
                self.model.momentum_update()

            # ======== 🧠 New: feedback to agent ========
            reward = -loss.item()  # Use negative loss as reward
            rewards = np.array([reward] * self.agent.n_agents)
            self.agent.update(rewards)
            # ====================================





            pbar.set_description(
                "Epoch {0} total loss={1:.3f} recon loss={2:.3f} mean loss={3:.3f} tri loss={4:.3f}".format(
                    epoch, loss, rec_loss, mean_loss, tri_loss),
                refresh=True)

    def process(self):
        self.model.eval()
        enc_rep, recon = self.model.evaluate(self.X, self.edge_index)
        enc_rep = enc_rep.to('cpu').detach().numpy()
        recon = recon.to('cpu').detach().numpy()
        recon[recon < 0] = 0

        self.adata.obsm['latent'] = enc_rep
        self.adata.obsm['ReX'] = recon
        return enc_rep, recon

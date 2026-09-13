import copy
import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset, uniform
from torch_scatter import scatter_add
import numpy as np
import warnings
from torch_geometric.utils import degree


# ==================== New: Wavelet Transform Module ====================
class SpaWaveletTransform(nn.Module):
    """
    Spatial transcriptomics wavelet transform module.

    Architecture (corresponding to paper Section 2.2):
      Decomposition: X → decomposed1(d/2) → decomposed2(d/4)     [progressive abstraction]
      Reconstruction: decomposed2 → reconstructed1(d/2) → reconstructed(d) [symmetric recovery]
      Gated fusion: α ⊙ reconstructed + (1-α) ⊙ residual_proj(X)    [adaptive balancing]

    decomposed2 = coarsest scale (global domain pattern)
    residual = X - reconstructed = high-frequency details
    α = per-dimension coarse/fine preference (learnable)
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.wavelet_levels = 2

        # Wavelet decomposition layers
        d_h1 = max(1, in_features // 2)
        d_h2 = max(1, in_features // 4)
        self.decompose1 = nn.Linear(in_features, d_h1)
        self.decompose2 = nn.Linear(d_h1, d_h2)

        # Wavelet reconstruction layers
        self.reconstruct1 = nn.Linear(d_h2, d_h1)
        self.reconstruct2 = nn.Linear(d_h1, out_features)

        # Residual projection
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = nn.Identity()

        # Gated fusion vector α (paper Eq.3)
        # Stores logit(α), converted to (0,1) via sigmoid in forward
        self.alpha_logit = nn.Parameter(torch.ones(out_features) * 0.8473)  # sigmoid(0.8473) ≈ 0.7

    def forward(self, x, return_all_scales=False):
        """
        Args:
            x: [N, in_features]
            return_all_scales: if True, return all intermediate scales (for visualization)
        Returns:
            default: x_wavelet [N, out_features]
            return_all_scales=True: dict {
                'decomposed1': (N, d_h1),
                'decomposed2': (N, d_h2),    ← coarsest scale
                'reconstructed1': (N, d_h1),
                'reconstructed': (N, out_features), ← reconstruction (smoothed)
                'residual': (N, out_features),    ← high-frequency residual
                'x_wavelet': (N, out_features),   ← fused output
                'alpha': (out_features,),         ← gate weights
            }
        """
        original_features = x

        # Wavelet decomposition (Eq.1)
        decomposed1 = F.relu(self.decompose1(x))      # (N, d_h1)    Formula 3
        decomposed2 = F.relu(self.decompose2(decomposed1))  # (N, d_h2) ← coarsest

        # Wavelet reconstruction (Eq.2)
        reconstructed1 = F.relu(self.reconstruct1(decomposed2))  # (N, d_h1)  Formula 4
        reconstructed = self.reconstruct2(reconstructed1)         # (N, out_features)

        # Residual projection
        residual = self.residual_proj(original_features)  # (N, out_features)

        # Gated fusion (Eq.3): α ⊙ reconstructed + (1-α) ⊙ residual
        alpha = torch.sigmoid(self.alpha_logit)  # (out_features,)
        x_wavelet = alpha * reconstructed + (1. - alpha) * residual  # Formula 5

        if return_all_scales:
            return {
                'decomposed1': decomposed1,
                'decomposed2': decomposed2,          # coarsest scale
                'reconstructed1': reconstructed1,
                'reconstructed': reconstructed,       # reconstruction (smoothed)
                'residual': residual - reconstructed, # high-frequency residual
                'x_wavelet': x_wavelet,               # fused output
                'alpha': alpha,                       # gate weights
            }
        return x_wavelet


# ==================== New: Causal Inference Module ====================
class SpaCausalInference(nn.Module):
    """Spatial causal inference module"""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Spatial causal attention
        self.spatial_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Gene expression causal attention
        self.gene_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )  # Q=YW_Q, K=YW_K, V=YW_V

        # Feed-forward network
        self.ffn = nn.Sequential(      # FFN(Y₂) = W₂·GELU(W₁Y₂+b₁)+b₂
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spatial_attn_mask=None, gene_attn_mask=None):
        # M_spatial[i,j] = 0 if A_ij=1, -inf otherwise
        # Z_spatial = softmax(QKᵀ/√dk + M_spatial)V    (7)
        # M_gene[i,j] = 0 if A_gene[ψ(i),ψ(j)]=1, -inf otherwise
        # Z_gene = softmax(QKᵀ/√dk + M_gene)V          (8)
        """
        Args:
            x: [num_nodes, hidden_dim]  multi-scale feature matrix Y
            spatial_attn_mask: [num_nodes, num_nodes]  spatial causal mask
                              0.0 = allow attention (causal edge exists)
                              -inf = block attention (no causal edge)
            gene_attn_mask: [num_nodes, num_nodes]  gene causal mask
                            0.0 = allow attention (co-regulated)
                            -inf = block attention (not co-regulated)
        Returns:
            Z: [num_nodes, hidden_dim]  causality-enhanced feature matrix
        """

        spatial_output, _ = self.spatial_attention(
            x, x, x, attn_mask=spatial_attn_mask
        )
        x = self.norm1(x + self.dropout(spatial_output))   # Y₁ = LayerNorm(Y + Dropout(Z_spatial))

        # Channel 2 — gene expression causal attention (simulating transcriptional co-regulation)
        # M^G derived from gene co-expression network G^G: info flows only between co-regulated genes
        gene_output, _ = self.gene_attention(
            x, x, x, attn_mask=gene_attn_mask
        )
        x = self.norm2(x + self.dropout(gene_output))   # Y₂ = LayerNorm(Y₁ + Dropout(Z_gene))

        # Feed-forward network — nonlinear causal feature enhancement
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))  # Z' = LayerNorm(Y₂ + Dropout(FFN(Y₂)))

        return x


# ==================== Original functions kept unchanged ====================
def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def full_block(in_features, out_features, p_drop, act=nn.ELU()):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        act,
        nn.Dropout(p=p_drop),
    )


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True):
        super(GraphConv, self).__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn(out_features)
        self.act = act
        self.dropout = dropout
        self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features, cached=True)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x


# ==================== Modified Encoder class ====================
class Encoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = config['feat_hidden1']
        self.feat_hidden2 = config['feat_hidden2']
        self.gcn_hidden = config['gcn_hidden']
        self.latent_dim = config['latent_dim']
        self.p_drop = config['p_drop']

        # Default enable wavelet transform and causal inference module
        self.use_wavelet = config.get('use_wavelet', True)
        self.use_causal = config.get('use_causal', True)

        print(f"Enhanced encoder initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Use wavelet transform: {self.use_wavelet}")
        print(f"  Use causal inference: {self.use_causal}")

        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        # Wavelet transform module (enabled by default)
        self.wavelet_module = SpaWaveletTransform(
            self.feat_hidden2,
            self.feat_hidden2
        )

        # Causal inference module (enabled by default)
        self.causal_module = SpaCausalInference(
            self.feat_hidden2,
            num_heads=4,
            dropout=self.p_drop
        )

        # GCN layers
        self.gc1 = GraphConv(self.feat_hidden2, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.gc2 = GraphConv(self.gcn_hidden, self.latent_dim, dropout=self.p_drop, act=lambda x: x)

    def _build_spatial_causal_mask(self, edge_index, num_nodes):
        """
        Build spatial causal attention mask M^S from the 3D spatial adjacency graph.

        Paper formula:
            M^S_ij = 1  if e_ij ∈ G  (spatial causal edge exists)
            M^S_ij = 0   otherwise

        Implementation: additive mask before softmax
            mask_ij = 0.0   if e_ij ∈ G   → softmax allows attention
            mask_ij = -inf  otherwise      → softmax blocks attention

        Returns:
            spatial_mask: [num_nodes, num_nodes]
        """
        mask = torch.full((num_nodes, num_nodes), float('-inf'),
                          device=edge_index.device if isinstance(edge_index, torch.Tensor) else 'cpu')

        # Parse edge_index to obtain source-target pairs
        if isinstance(edge_index, torch.Tensor):
            if edge_index.is_sparse:
                edge_index = edge_index.coalesce()
                src, dst = edge_index.indices()
            elif edge_index.dim() == 2:
                src, dst = edge_index[0], edge_index[1]
            else:
                return torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        else:
            return torch.zeros(num_nodes, num_nodes)

        # Set spatial causal edges: 0.0 means allow attention
        mask[src, dst] = 0.0
        mask[dst, src] = 0.0  # Undirected graph, bidirectional causality
        # Self-loop: each spot has causal influence on itself
        mask[torch.arange(num_nodes), torch.arange(num_nodes)] = 0.0

        return mask

    def _build_gene_causal_mask(self, features, topk_ratio=0.1):
        """
        Build gene expression causal attention mask M^G from gene co-expression network.

        Paper:
            Compute pairwise Pearson correlation from co-expression matrix, threshold to retain significant edges → G^G
            M^G_ij = 1  if (g_i, g_j) ∈ G^G  (gene co-regulation)
            M^G_ij = 0   otherwise

        Code implementation (approximating expression correlation with PCA feature cosine similarity):
            1. Normalize feature vectors
            2. Compute cosine similarity matrix S [N,N]
            3. Keep topk most similar neighbors per spot (simulating "co-regulation" thresholding)
            4. mask_ij = 0.0 (high similarity, allow causal attention) / -inf (dissimilar, block)

        Args:
            features: [num_nodes, hidden_dim]
            topk_ratio: fraction of topk similar neighbors to keep (default 10%)
        Returns:
            gene_mask: [num_nodes, num_nodes]
        """
        num_nodes = features.shape[0]
        topk = max(1, int(num_nodes * topk_ratio))

        # Cosine similarity matrix [N,N]
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.matmul(features_norm, features_norm.T)

        # Keep topk most similar neighbors per spot (simulating Pearson correlation thresholding)
        _, topk_indices = torch.topk(similarity, k=min(topk, num_nodes), dim=1)

        # Build mask: default block (-inf), topk similar neighbors allow (0.0)
        gene_mask = torch.full((num_nodes, num_nodes), float('-inf'),
                               device=features.device)
        row_idx = torch.arange(num_nodes, device=features.device).unsqueeze(1).expand(-1, topk)
        gene_mask[row_idx, topk_indices] = 0.0
        # Self-loop always allowed
        gene_mask[torch.arange(num_nodes), torch.arange(num_nodes)] = 0.0

        return gene_mask

    def forward(self, x, edge_index, return_wavelet_features=False, return_causal_features=False):
        # Base feature extraction
        base_features = self.encoder(x)

        # Apply wavelet transform
        if self.use_wavelet and return_wavelet_features:
            wf = self.wavelet_module(base_features, return_all_scales=True)
            wavelet_features = wf['x_wavelet']
        elif self.use_wavelet:
            wavelet_features = self.wavelet_module(base_features, return_all_scales=False)
        else:
            wavelet_features = base_features

        # Apply causal inference (with causal structure masks)
        if self.use_causal:
            # Build spatial causal mask M^S: based on 3D adjacency graph G
            spatial_attn_mask = self._build_spatial_causal_mask(
                edge_index, wavelet_features.shape[0]
            )
            # Build gene causal mask M^G: based on feature co-expression similarity
            gene_attn_mask = self._build_gene_causal_mask(wavelet_features)

            # Causal dual-channel attention (constrained by M^S + M^G)
            causal_features = self.causal_module(
                wavelet_features,
                spatial_attn_mask=spatial_attn_mask,
                gene_attn_mask=gene_attn_mask
            )
        else:
            causal_features = wavelet_features

        # Graph convolution (further propagate over causal features)
        x = self.gc1(causal_features, edge_index)
        x = self.gc2(x, edge_index)

        # Build wavelet feature dictionary
        if self.use_wavelet and return_wavelet_features:
            wf['causal_features'] = causal_features
            wf['gcn1_features'] = x  # gc2 input
            if return_causal_features:
                return x, wf, causal_features
            return x, wf

        # Return causal module output for regularization
        if return_causal_features:
            return x, causal_features

        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, config, imputation=True):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = config['latent_dim']
        self.p_drop = config['p_drop']
        self.imputation = imputation
        if self.imputation:
            self.layer1 = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.layer1 = GraphConv(self.input_dim, self.output_dim, dropout=self.p_drop, act=nn.Identity())

    def forward(self, x, edge_index):
        if self.imputation:
            return self.layer1(x)
        return self.layer1(x, edge_index)


# ==================== Modified SAMF_model class ====================
class SAMF_model(nn.Module):
    def __init__(self, input_dim, config, imputation=True):
        super().__init__()
        self.imputation = imputation
        self.dec_in_dim = config['latent_dim']

        # Default enable wavelet transform and causal inference
        self.use_wavelet = True  # Enable wavelet transform by default
        self.use_causal = True  # Enable causal inference by default
        self.causal_reg_weight = config.get('causal_reg_weight', 0.01)
        self.gamma1 = config.get('gamma1', 0.5)  # γ₁: spatial smoothness
        self.gamma2 = config.get('gamma2', 0.2)  # γ₂: feature independence

        print(f"\nEnhanced SAMF model initialized (enhancements enabled by default):")
        print(f"  Input dim: {input_dim}")
        print(f"  Latent dim: {self.dec_in_dim}")
        print(f"  Use wavelet transform: {self.use_wavelet}")
        print(f"  Use causal inference: {self.use_causal}")
        print(f"  Causal regularization weight: {self.causal_reg_weight}")
        print(f"  γ₁ (spatial): {self.gamma1:.2f}, γ₂ (independence): {self.gamma2:.2f}")

        self.online_encoder = Encoder(input_dim, config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self._init_target()

        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, config['project_dim'], bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.projector = GraphConv(config['project_dim'], self.dec_in_dim, dropout=config['p_drop'], act=lambda x: x)
        self.decoder = Decoder(input_dim, config, self.imputation)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.rep_mask = nn.Parameter(torch.zeros(1, self.dec_in_dim))
        self.mask_rate = config['mask_rate']
        self.t = config['t']
        self.momentum_rate = config['momentum_rate']
        self.replace_rate = 0.05
        self.mask_token_rate = 1 - self.replace_rate
        self.anchor_pair = None

        self.weight = nn.Parameter(torch.empty(self.dec_in_dim, self.dec_in_dim))
        uniform(self.dec_in_dim, self.weight)

    def _init_target(self):
        for param_teacher in self.target_encoder.parameters():
            param_teacher.detach()
            param_teacher.requires_grad = False

    def momentum_update(self):
        base_momentum = self.momentum_rate
        for param_encoder, param_teacher in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + param_encoder.data * (1. - base_momentum)

    def encoding_mask_noise(self, x, edge_index, mask_rate=0.3):
        num_nodes = x.shape[0]
        self.num_nodes = num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]

        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_edge_index = edge_index.clone()

        return out_x, use_edge_index, (mask_nodes, keep_nodes)

    def generate_neg_nodes(self, mask_nodes):
        num_mask_nodes = mask_nodes.size(0)
        neg_nodes_x = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
        neg_nodes_y = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
        return neg_nodes_x, neg_nodes_y

    def compute_causal_regularization(self, causal_features, edge_index):
        """
        Causal regularization loss L_causal = γ₁ L_spatial + γ₂ L_indep

        L_spatial (Causal Markov condition):
            Spatially adjacent spots in the causal graph should have similar representations
            L_spatial = E_{(i,j)~E} ||Z_i - Z_j||²₂

        L_indep (Independent Causal Mechanisms):
            Decorrelation: different causal factors should operate independently
            L_indep = ||Cov(Z) - I||_F

        Args:
            causal_features: causal module output Z [N, hidden_dim]
            edge_index: edge set of the 3D spatial adjacency graph
        Returns:
            Scalar causal regularization loss
        """
        if not self.use_causal or self.causal_reg_weight <= 0:
            return torch.tensor(0.0, device=causal_features.device)

        # Handle different edge_index formats
        if isinstance(edge_index, torch.sparse.Tensor):
            edge_index = edge_index.coalesce()
            src, dst = edge_index.indices()
        elif isinstance(edge_index, tuple) and len(edge_index) == 2:
            src, dst = edge_index
        elif isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            src, dst = edge_index[0], edge_index[1]
        elif isinstance(edge_index, list):
            if len(edge_index) == 0:
                return torch.tensor(0.0, device=causal_features.device)
            edge_tensor = torch.tensor(edge_index[0], device=causal_features.device)
            src, dst = edge_tensor[:, 0], edge_tensor[:, 1]
        else:
            return torch.tensor(0.0, device=causal_features.device)

        # L_spatial: Causal Markov condition — representations of causal graph neighbors should be similar
        if len(src) > 0 and len(dst) > 0 and src.numel() == dst.numel():
            src_features = causal_features[src]
            dst_features = causal_features[dst]
            spatial_loss = F.mse_loss(src_features, dst_features)
        else:
            spatial_loss = torch.tensor(0.0, device=causal_features.device)

        # L_indep: Independent Causal Mechanisms — feature dimension decorrelation
        if causal_features.size(0) > 1:
            features_normalized = F.normalize(causal_features, dim=0)
            correlation = torch.matmul(features_normalized.T, features_normalized)
            identity = torch.eye(correlation.size(0), device=causal_features.device)
            causal_indep_loss = torch.mean((correlation - identity).abs())
        else:
            causal_indep_loss = torch.tensor(0.0, device=causal_features.device)

        # L_causal = γ₁ L_spatial + γ₂ L_indep  (Formula 10)
        total_loss = self.gamma1 * spatial_loss + self.gamma2 * causal_indep_loss

        return total_loss * self.causal_reg_weight

    def mask_attr_prediction(self, x, edge_index, anchor_pair):
        use_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, edge_index, self.mask_rate)

        # Encoder forward pass — simultaneously obtain causal module output for regularization
        enc_rep, causal_features = self.online_encoder(
            use_x, use_adj, return_causal_features=True
        )

        with torch.no_grad():
            x_t = x.clone()
            x_t[keep_nodes] = 0.0
            x_t[keep_nodes] += self.enc_mask_token
            rep_t = self.target_encoder(x_t, use_adj)

        # Fix: ensure cl_loss is always a torch tensor
        if anchor_pair is not None:
            anchor, positive, negative = anchor_pair
            summary = self.avg_readout(enc_rep, [anchor, positive])
            num_mask_nodes = mask_nodes.size(0)
            neg_nodes = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
            cl_loss = self.dgi_loss(enc_rep[mask_nodes], enc_rep[neg_nodes], summary[mask_nodes])
        else:
            cl_loss = torch.tensor(0.0, device=x.device)  # Fix: use torch tensor

        rep = enc_rep
        rep = self.encoder_to_decoder(rep)
        rep[mask_nodes] = 0.0
        rep = self.projector(rep, use_adj)

        match_loss = self.match_loss(rep, rep_t, mask_nodes)
        recon = self.decoder(rep, use_adj)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        rec_loss = self.sce_loss(x_rec, x_init, t=self.t)

        # Compute causal regularization loss — applied on causal module output Z
        causal_loss = self.compute_causal_regularization(causal_features, use_adj)

        return match_loss, rec_loss, cl_loss, causal_loss

    def match_loss(self, rep, rep_t, mask_nodes, t=2):
        pox_x_index, pox_y_index = mask_nodes, mask_nodes
        neg_x_index, neg_y_index = self.generate_neg_nodes(mask_nodes)
        std_emb = F.normalize(rep.clone(), p=2, dim=-1)
        tgt_emb = F.normalize(rep_t.clone(), p=2, dim=-1)

        pox_x = std_emb[pox_x_index]
        pox_y = tgt_emb[pox_y_index]
        neg_x = std_emb[neg_x_index]
        neg_y = tgt_emb[neg_y_index]

        pos_cos = (0.5 * (1 + (pox_x * pox_y).sum(dim=-1))).pow(t)
        pos_loss = -torch.log(pos_cos)
        neg_cos = (0.5 * (1 + (neg_x * neg_y).sum(dim=-1))).pow(t)
        neg_loss = -torch.log(1 - neg_cos)
        loss = 0.5 * (pos_loss.mean() + neg_loss.mean())
        return loss

    def sce_loss(self, x, y, t=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
        loss = -torch.log(cos_m.pow_(t))
        return loss.mean()

    def triplet_loss(self, emb, anchor, positive, negative, margin=1.0):
        anchor_arr = emb[anchor]
        positive_arr = emb[positive]
        negative_arr = emb[negative]
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        return tri_output

    def forward(self, x, edge_index, anchor_pair):
        # Always return 4 values; if a feature is disabled, the corresponding loss is 0
        return self.mask_attr_prediction(x, edge_index, anchor_pair)

    @torch.no_grad()
    def evaluate(self, x, edge_index):
        # Encoder evaluation
        enc_rep = self.online_encoder(x, edge_index)
        rep = self.encoder_to_decoder(enc_rep)
        rep = self.projector(rep, edge_index)
        recon = self.decoder(rep, edge_index)
        return enc_rep, recon

    @torch.no_grad()
    def get_wavelet_features(self, x, edge_index):
        """
        [Visualization only] Extract multi-scale features from the wavelet module.

        Two conditions must be met for valid wavelet features:
          1. config.use_wavelet = True
          2. Model initialized with Encoder(use_wavelet=True)

        Returns:
            (latent, wavelet_dict) where wavelet_dict contains:
                - decomposed1:  intermediate scale (N, d_h1)
                - decomposed2:  coarsest scale (N, d_h2) ← global domain pattern
                - reconstructed: reconstructed features (N, out_features)
                - residual:      high-frequency residual (N, out_features)
                - x_wavelet:     gated fusion output (N, out_features)
                - alpha:         gate weights (out_features,)
                - causal_features: causal inference output (N, out_features)
        """
        self.eval()
        result = self.online_encoder(x, edge_index, return_wavelet_features=True)
        if isinstance(result, tuple):
            latent, wf = result
            return latent, wf
        else:
            # Wavelet not enabled, return None
            return result, None

    @torch.no_grad()
    def std_tgt_embedding(self, x, edge_index):
        s_rep = self.online_encoder(x, edge_index)
        t_rep = self.target_encoder(x, edge_index)
        return s_rep, t_rep

    def avg_readout(self, rep_pos_x, edge_index):
        # Handle edge_index type
        if isinstance(edge_index, torch.sparse.Tensor):
            # For sparse tensor, convert to dense to get edges
            adj = edge_index.to_dense()
            src, dst = torch.nonzero(adj, as_tuple=True)
        elif isinstance(edge_index, list):
            # If edge_index is a list, assume it's [anchor, positive]
            # Directly compute the feature mean of these points
            anchor, positive = edge_index
            src = anchor
            dst = positive
        else:
            src, dst = edge_index

        neighbor_sum = scatter_add(rep_pos_x[src], dst, dim=0, dim_size=rep_pos_x.size(0))
        neighbor_count = scatter_add(torch.ones_like(src, dtype=torch.float), dst, dim=0, dim_size=rep_pos_x.size(0))
        neighbor_count = neighbor_count.clamp(min=1)
        summary = neighbor_sum / neighbor_count.unsqueeze(-1)
        return torch.sigmoid(summary)

    def discriminate(self, z, summary, sigmoid=True):
        assert isinstance(summary, torch.Tensor), "Summary should be a torch.Tensor"
        value = torch.matmul(z, torch.matmul(self.weight, summary.t()))
        return torch.sigmoid(value) if sigmoid else value

    def dgi_loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss

    def CL_Loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        Cos_loss = -torch.log(1 - F.cosine_similarity(pos_z, neg_z) + 1e-15).mean()
        loss = Cos_loss + pos_loss + neg_loss  # 50
        return loss

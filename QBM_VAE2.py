import anndata
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Callable, Literal
from .utils import LossFunction
import logging
import kaiwu as kw # Added import

class scDataset(Dataset):
    def __init__(self, anndata_info, batch_indices=None):
        """
        Dataset class for single-cell RNA data with batch indices.

        Args:
            anndata_info (np.ndarray): Dense RNA data from AnnData.X
            batch_indices (np.ndarray or torch.Tensor, optional): Batch indices for each sample
        """
        # Expect anndata_info to be a dense NumPy array
        self.rna_tensor = torch.tensor(anndata_info, dtype=torch.float32)

        # Handle batch_indices
        if batch_indices is not None:
            self.batch_indices = batch_indices
            if len(self.batch_indices) != self.rna_tensor.shape[0]:
                raise ValueError("Length of batch_indices must match number of samples in anndata_info")
        else:
            # If no batch_indices provided, assign zeros (single batch)
            self.batch_indices = torch.zeros(self.rna_tensor.shape[0], dtype=torch.long)

    def __len__(self):
        """
        Return the number of samples.
        """
        return self.rna_tensor.shape[0]

    def __getitem__(self, idx):
        """
        Get a sample and its batch index.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (rna_data, batch_index)
        """
        return self.rna_tensor[idx, :], self.batch_indices[idx]


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 latent_dim,
                 normalization='batchnorm'):
        super(Encoder, self).__init__()

        # Create list to hold layers
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)

        # Input dimension for first layer
        current_dim = input_dim

        # Create hidden layers based on hidden_dims list
        for hidden_dim in hidden_dims:
            # Add linear layer
            self.layers.append(nn.Linear(current_dim, hidden_dim))

            # Add normalization layer
            if normalization == 'batchnorm':
                self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == 'layernorm':
                self.norm_layers.append(nn.LayerNorm(hidden_dim))
            else:
                self.norm_layers.append(nn.Identity())  # No normalization

            current_dim = hidden_dim

        # Final layer to latent dimension
        self.fc_final = nn.Linear(current_dim, latent_dim)

    def forward(self, x):
        h = x
        # Process through hidden layers
        for layer, norm in zip(self.layers, self.norm_layers):
            h = layer(h)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)

        # Final layer
        q_logits = self.fc_final(h)
        return q_logits


class Decoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 hidden_dims,
                 output_dim,
                 normalization='batchnorm'):
        super(Decoder, self).__init__()

        # Reverse hidden_dims to ensure low-to-high dimension order
        hidden_dims = hidden_dims[::-1]

        # Create lists to hold layers
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)

        # Input dimension for first layer
        current_dim = latent_dim

        # Create hidden layers based on reversed hidden_dims list
        for hidden_dim in hidden_dims:
            # Add linear layer
            self.layers.append(nn.Linear(current_dim, hidden_dim))

            # Add normalization layer
            if normalization == 'batchnorm':
                self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == 'layernorm':
                self.norm_layers.append(nn.LayerNorm(hidden_dim))
            else:
                self.norm_layers.append(nn.Identity())  # No normalization

            current_dim = hidden_dim

        # Final layer to output dimension
        self.fc_final = nn.Linear(current_dim, output_dim)

    def forward(self, zeta):
        h = zeta
        # Process through hidden layers
        for layer, norm in zip(self.layers, self.norm_layers):
            h = layer(h)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)

        # Final layer
        x_recon = self.fc_final(h)
        return x_recon


class ZINBDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, normalization='batchnorm'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if normalization == 'batchnorm':
                self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == 'layernorm':
                self.norm_layers.append(nn.LayerNorm(hidden_dim))
            else:
                self.norm_layers.append(nn.Identity())
            current_dim = hidden_dim

        self.fc_mu = nn.Linear(current_dim, output_dim)
        self.fc_theta = nn.Linear(current_dim, output_dim)
        self.fc_pi = nn.Linear(current_dim, output_dim)

    def forward(self, h):
        for layer, norm in zip(self.layers, self.norm_layers):
            h = F.leaky_relu(norm(layer(h)))
            h = self.dropout(h)

        mu = torch.exp(self.fc_mu(h))
        theta = torch.exp(self.fc_theta(h))
        pi = torch.sigmoid(self.fc_pi(h))
        return mu, theta, pi


class LinearLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.2,
                 batchnorm: bool = False,
                 activation=None,
                 ):
        super(LinearLayer, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.batchnorm = nn.LayerNorm(output_dim) if batchnorm else None

        self.activation = None
        if activation is not None:
            if activation == 'relu':
                self.activation = F.relu
            elif activation == 'sigmoid':
                self.activation = torch.sigmoid
            elif activation == 'tanh':
                self.activation = torch.tanh
            elif activation == 'leakyrelu':
                self.activation = torch.nn.LeakyReLU()
            elif activation == 'selu':
                self.activation = torch.nn.SELU()

    def forward(self, input_x):
        x = self.linear_layer(input_x)

        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 activation=None,
                 dropout=0.2,
                 batchnorm=True,
                 input_dropout=0.4
                 ):
        super(FeatureEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [LinearLayer(input_dim, hidden_dims[0], batchnorm=batchnorm, activation=activation, dropout=input_dropout)])

        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                LinearLayer(hidden_dims[i], hidden_dims[i + 1], batchnorm=batchnorm, activation=activation, dropout=dropout))

    def forward(self, input_x):
        for layer in self.layers:
            input_x = layer(input_x)
        return input_x


class RNAVAEDecoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 hidden_dims,
                 output_dim,
                 dropout=0.1
                 ):
        super(RNAVAEDecoder, self).__init__()
        hidden_dims = hidden_dims[::-1]
        self.px_decoder = FeatureEncoder(latent_dim, hidden_dims, activation='leakyrelu', dropout=dropout)

        self.rna_scale_decoder = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softmax(dim=-1)
        )

        self.rna_rate_decoder = nn.Linear(hidden_dims[-1], output_dim)
        self.rna_dropout_decoder = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self,
                latent_z: torch.tensor):
        px = self.px_decoder(latent_z)

        px_rna_scale = self.rna_scale_decoder(px)
        px_rna_dropout = self.rna_dropout_decoder(px)
        px_rna_rate = self.rna_rate_decoder(px)

        output = {
            'px': px,
            'px_rna_scale': px_rna_scale,
            'px_rna_dropout': px_rna_dropout,
            'px_rna_rate': px_rna_rate
        }
        return output


def zinb_nll(x, mu, theta, pi, eps=1e-8):
    softplus_pi = -F.softplus(-pi)
    nb_case = (
        torch.lgamma(theta + eps + x) - torch.lgamma(x + 1.0) - torch.lgamma(theta + eps)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    )
    nb_case = nb_case + softplus_pi

    zero_case = -F.softplus(-pi + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps)))
    res = torch.where(torch.lt(x, 1e-8), zero_case, nb_case)
    return -res.sum(dim=-1).mean()


class RBM(nn.Module):
    def __init__(self, latent_dim):
        super(RBM, self).__init__()
        self.h = nn.Parameter(torch.zeros(latent_dim))
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.001)  # 对称权重
        self.latent_dim = latent_dim
        
        # Added from DVAE_RBM
        kw.utils.set_log_level("INFO")
        kw.utils.CheckpointManager.save_dir = f"./cim_test2/"
        self.worker = kw.cim.CIMOptimizer()
        self.ising_matrix = None # To store ising matrix if needed

    def energy(self, z):
        z = z.float()
        h_term = torch.sum(z * self.h, dim=-1)
        w_term = torch.sum((z @ self.W) * z, dim=-1)  # 注意对称性
        return h_term + w_term
    
    # Added from DVAE_RBM
    def _create_ising_matrix(self, number_of_hidden_units):
        total_units = number_of_hidden_units + number_of_hidden_units
        adjacency_matrix = torch.zeros((total_units, total_units), device=self.h.device)
        for i in range(number_of_hidden_units):
            for j in range(number_of_hidden_units):
                value = 0.25 * self.W[i, j]
                adjacency_matrix[i, number_of_hidden_units + j] = value
                adjacency_matrix[number_of_hidden_units + j, i] = value

        bias_terms = torch.zeros(total_units, device=self.h.device)
        for i in range(number_of_hidden_units):
            bias_terms[i] = 0.5 * self.h[i] + 0.25 * torch.sum(self.W[i, :])
        for j in range(number_of_hidden_units):
            bias_terms[number_of_hidden_units + j] = 0.5 * self.h[j] + 0.25 * torch.sum(self.W[:, j])

        left_part = torch.cat([
            adjacency_matrix,
            bias_terms.unsqueeze(0)
        ], dim=0)

        right_part = torch.cat([
            bias_terms.view(-1, 1),
            torch.zeros(1, 1, device=self.h.device)
        ], dim=0)

        ising_matrix = -torch.cat([left_part, right_part], dim=1)  # 最终形状 (d+1, d+1)
        return ising_matrix.cpu().detach().numpy()

    # Added from DVAE_RBM and modified
    def adjust_precision(self, ising_matrix):
        return kw.cim.adjust_ising_matrix_precision(ising_matrix, bit_width=14)
    
    # Added from DVAE_RBM and renamed
    def ising_sample(self, number_of_samples, number_of_hidden_units, fold_id, step, behavior):
        self.worker.size_limit = number_of_samples
        ising_matrix = self._create_ising_matrix(number_of_hidden_units)
        # 调整精度
        ising_matrix = self.adjust_precision(ising_matrix)
        self.ising_matrix = ising_matrix
        self.worker.task_name = f"fold-{fold_id}_step-{step}_{behavior}"
        
        output = self.worker.solve(ising_matrix)

        result = [sample[:-1] * sample[-1] for sample in output]
        result = kw.sampler.spin_to_binary(np.array(result))
        return torch.tensor(result[:, :number_of_hidden_units], device=self.h.device, dtype=torch.float32)

    def gibbs_sampling(self, num_samples, steps=1):
        z = torch.randint(0, 2, (num_samples, self.latent_dim), dtype=torch.float).to(self.h.device)
        for _ in range(steps):
            probs = torch.sigmoid(self.h + z @ self.W)
            z = (torch.rand_like(z) < probs).float()
        return z

    def compute_gradients(self, z_positive, num_negative_samples=64, gibbs_steps=50):
        """计算正相和负相的梯度"""
        # 正相：E_q[∂E/∂θ]
        z_positive = z_positive.float()
        positive_h_grad = z_positive.mean(dim=0)  # ∂E/∂h = z_l
        positive_w_grad = torch.einsum('bi,bj->ij', z_positive, z_positive) / z_positive.size(0)  # ∂E/∂W = z_l z_m

        # 负相：E_p[∂E/∂θ]
        z_negative = self.gibbs_sampling(num_negative_samples, steps=gibbs_steps)
        negative_h_grad = z_negative.mean(dim=0)
        negative_w_grad = torch.einsum('bi,bj->ij', z_negative, z_negative) / z_negative.size(0)

        # 总梯度
        h_grad = positive_h_grad - negative_h_grad
        w_grad = positive_w_grad - negative_w_grad
        # 对称化W的梯度（因为RBM假设W对称）
        w_grad = (w_grad + w_grad.T) / 2
        return {'h': h_grad, 'W': w_grad}


class QBM_VAE(nn.Module):
    def __init__(self,
                 hidden_dims=None,
                 latent_dim=256,
                 batch_dim=16,
                 beta=0.5,
                 beta_kl=0.0001,
                 use_norm='batchnorm',
                 device=torch.device('cpu'),
                 batch_representation: Literal["one-hot", "embedding"] = "one-hot",
                 use_zinb_decoder: bool = False):
        super(QBM_VAE, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512]
        self.decoder_input_dim = None
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.batch_dim = batch_dim
        self.beta = beta
        self.beta_kl = beta_kl
        self.device = device
        self.use_norm = use_norm
        self.use_zinb_decoder = use_zinb_decoder

        # Initialize attributes to None; will be set in set_adata
        self.input_dim = None
        self.n_batches = None
        self.encoder = None
        self.decoder = None
        self.rbm = None
        self.batch_encoder=None

        # Store AnnData and batch information
        self.adata = None
        self.batch_key = None
        self.batch_indices = None

        self.to(device)
        self.batch_representation = batch_representation

    def set_adata(self,
                  adata: anndata.AnnData,
                  batch_key='batch'):
        self.adata = adata.copy()
        self.batch_key = batch_key
        self.input_dim = adata.X.shape[1]

        # 检查是否使用 batch 信息
        use_batch = False
        if batch_key in adata.obs:
            batch_series = adata.obs[batch_key].astype(str)
            unique_batches = batch_series[batch_series != ""].unique()
            if len(unique_batches) > 0:
                use_batch = True

        self.use_batch = use_batch  # 新增标记

        if use_batch:
            batch_categories = batch_series.astype('category')
            self.batch_indices = torch.tensor(batch_categories.cat.codes.values, dtype=torch.long)
            self.n_batches = len(batch_categories.cat.categories)
        else:
            self.batch_indices = torch.zeros(adata.shape[0], dtype=torch.long)  # dummy batch
            self.n_batches = 0

        # 初始化 encoder, decoder, rbm
        self.encoder = Encoder(
            self.input_dim,
            self.hidden_dims,
            self.latent_dim,
            normalization=self.use_norm
        ).to(self.device)

        if self.batch_representation == 'one-hot' and use_batch:
            self.decoder_input_dim = self.latent_dim + self.n_batches
        elif self.batch_representation == 'embedding' and use_batch:
            self.decoder_input_dim = self.latent_dim + self.batch_dim
        else:
            self.decoder_input_dim = self.latent_dim

        self.decoder = Decoder(
            self.decoder_input_dim,
            self.hidden_dims,
            self.input_dim,
            normalization=self.use_norm
        ).to(self.device)

        self.rbm = RBM(self.latent_dim).to(self.device)

        if use_batch and self.batch_representation == 'embedding':
            self.batch_encoder = nn.Embedding(self.n_batches, self.batch_dim).to(self.device)
        else:
            self.batch_encoder = None

        if self.use_zinb_decoder:
            self.zinb_decoder = RNAVAEDecoder(
                self.decoder_input_dim,
                self.hidden_dims,
                self.input_dim,
            ).to(self.device)

        print(f"Set AnnData with input_dim={self.input_dim}, batches_used={self.use_batch}")

    def reparameterize(self, q_logits, rho):
        q = torch.sigmoid(q_logits)
        zeta = torch.zeros_like(rho)
        mask = rho > (1 - q)
        beta_tensor = torch.tensor(self.beta, dtype=torch.float32, device=rho.device)
        exp_beta_minus_1 = torch.exp(beta_tensor) - 1
        zeta[mask] = (1 / beta_tensor) * torch.log(
            (torch.clamp(rho[mask] - (1 - q[mask]), min=0) / q[mask]) * exp_beta_minus_1 + 1
        )
        z = (zeta > 0).float()
        return zeta, z, q

    def _get_batch_one_hot(self, indices):
        """
        Convert batch indices to one-hot encodings.

        Args:
            indices (torch.Tensor): Batch indices for the current batch

        Returns:
            torch.Tensor: One-hot encoded batch tensor
        """
        return F.one_hot(indices, num_classes=self.n_batches).long().to(self.device)

    def kl_divergence(self, z, q):
        q = torch.clamp(q, min=1e-7, max=1 - 1e-7)
        log_q = q * torch.log(q) + (1 - q) * torch.log(1 - q)
        entropy = -log_q.sum(dim=-1)
        energy_pos = self.rbm.energy(z)
        z_negative = self.rbm.gibbs_sampling(z.size(0))
        energy_neg = self.rbm.energy(z_negative)
        # 用负相能量的均值作为 logZ 的近似
        logZ = energy_neg.mean()
        kl = (energy_pos - entropy + logZ).mean()
        return kl

    def forward(self, x, batch_indices):
        q_logits = self.encoder(x)
        rho = Uniform(0, 1).sample(q_logits.shape).to(x.device)
        zeta, z, q = self.reparameterize(q_logits, rho)

        if self.use_batch:
            if self.batch_representation == 'embedding':
                batch_emb = self.batch_encoder(batch_indices)
                decoder_input = torch.cat([zeta, batch_emb], dim=-1)
            elif self.batch_representation == 'one-hot':
                batch_one_hot = self._get_batch_one_hot(batch_indices)
                decoder_input = torch.cat([zeta, batch_one_hot], dim=-1)
        else:
            decoder_input = zeta

        if self.use_zinb_decoder:
            lib_size = x.sum(1).to(self.device).unsqueeze(1)
            decoder_output = self.zinb_decoder(decoder_input)
            px_rna_scale_final = decoder_output['px_rna_scale'] * lib_size
            recon_loss = LossFunction.zinb_reconstruction_loss(
                x,
                mu=px_rna_scale_final,
                theta=decoder_output['px_rna_rate'].exp(),
                gate_logits=decoder_output['px_rna_dropout'],
            ).sum() / x.size(0)
        else:
            x_recon = self.decoder(decoder_input)
            recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)

        kl_loss = self.kl_divergence(z, q)
        elbo = -recon_loss - self.beta_kl * kl_loss
        return elbo, recon_loss, kl_loss, z, zeta

    def get_representation(self,
                           adata=None,
                           batch_size=128,
                           layer_key=None):
        """
        Extract batch-corrected latent representations.
        """

        if adata is None and self.adata is None:
            raise ValueError("No AnnData object provided or set")
        adata = adata if adata is not None else self.adata

        if layer_key is not None:
            adata_array = adata.layers[layer_key].toarray() if hasattr(adata.layers[layer_key], 'toarray') else adata.layers[layer_key]
        else:
            adata_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        if self.use_batch:
            batch_indices = torch.tensor(adata.obs[self.batch_key].astype('category').cat.codes.values,
                                         dtype=torch.long)
        else:
            batch_indices = torch.zeros(adata.shape[0], dtype=torch.long)

        dataset = scDataset(adata_array, batch_indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        self.eval()
        latent_reps = []
        with torch.no_grad():
            for x, batch_idx in dataloader:
                x = x.to(self.device)
                batch_idx = batch_idx.to(self.device)
                _, _, _, z, zeta = self(x, batch_idx)
                latent_reps.append(zeta.cpu().numpy())

        reps = np.concatenate(latent_reps, axis=0)
        print(f"Latent representation shape: {reps.shape}")
        return reps

    def fit(self,
            adata,
            val_percentage=0.1,
            batch_size=128,
            epochs=100,
            beta=0.5,
            lr=1e-4,
            weight_decay=1e-5,
            rbm_lr=1e-2,
            early_stopping=True,
            early_stopping_patience=30,
            n_epochs_kl_warmup=None,
            layer_key=None):

        if adata is None and self.adata is None:
            raise ValueError("No AnnData object provided or set")
        adata = adata if adata is not None else self.adata
        if layer_key is not None:
            adata_array = adata.layers[layer_key].toarray() if hasattr(adata.layers[layer_key], 'toarray') else adata.layers[layer_key]
        else:
            adata_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        if self.use_batch:
            batch_indices = torch.tensor(adata.obs[self.batch_key].astype('category').cat.codes.values,
                                         dtype=torch.long)
        else:
            batch_indices = torch.zeros(adata.shape[0], dtype=torch.long)

        if early_stopping:
            train_indices, val_indices = train_test_split(
                np.arange(adata.shape[0]), test_size=val_percentage, random_state=0
            )

            adata_train_array = adata_array[train_indices]
            adata_val_array = adata_array[val_indices]
            train_batch_indices = batch_indices[train_indices]
            val_batch_indices = batch_indices[val_indices]

            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(adata_val_array, dtype=torch.float32),
                val_batch_indices
            )
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            adata_train_array = adata_array
            train_batch_indices = batch_indices

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(adata_train_array, dtype=torch.float32),
            train_batch_indices
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr,
            # weight_decay=weight_decay
        )
        rbm_optimizer = torch.optim.Adam(self.rbm.parameters(), lr=rbm_lr)

        # Early stopping variables
        best_val_elbo = float('-inf')
        patience_counter = 0
        epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Progress", total=epochs)
        best_state_dict = None

        kl_warmup_epochs = n_epochs_kl_warmup or epochs

        for epoch in epoch_pbar:
            # if n_epochs_kl_warmup and epoch <= kl_warmup_epochs:
            #     current_kl_weight = min(0.0001, 0.0001* epoch / kl_warmup_epochs)
            #     self.beta_kl = current_kl_weight

            self.train()
            total_elbo, total_recon, total_kl = 0, 0, 0
            for x, batch_idx in train_dataloader:
                x = x.to(self.device)
                batch_idx = batch_idx.to(self.device)
                optimizer.zero_grad()
                rbm_optimizer.zero_grad()

                elbo, recon_loss, kl_loss, z, zeta = self(x, batch_idx)
                loss = -elbo
                loss.backward()

                # 手动计算RBM的梯度
                # rbm_grads = self.rbm.compute_gradients(z.detach())  # z.detach()避免重复求导
                # with torch.no_grad():
                #     self.rbm.h.grad = rbm_grads['h']
                #     self.rbm.W.grad = rbm_grads['W']

                optimizer.step()
                rbm_optimizer.step()

                total_elbo += elbo.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            avg_elbo = total_elbo / len(train_dataloader)
            avg_recon = total_recon / len(train_dataloader)
            avg_kl = total_kl / len(train_dataloader)
            # print(f"Epoch [{epoch}/{epochs}], ELBO: {avg_elbo:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
            epoch_pbar.set_postfix({
                'KL_weight': f'{self.beta_kl}',
                'ELBO': f'{avg_elbo:.4f}',
                'Recon': f'{avg_recon:.4f}',
                'KL': f'{avg_kl:.4f}'
            })

            if early_stopping:
                self.eval()
                val_total_elbo, val_total_recon, val_total_kl = 0, 0, 0
                for x, batch_idx in val_dataloader:
                    x = x.to(self.device)
                    batch_idx = batch_idx.to(self.device)
                    with torch.no_grad():
                        elbo, recon_loss, kl_loss, z, zeta = self(x, batch_idx)

                    val_total_elbo += elbo.item()
                    val_total_recon += recon_loss.item()
                    val_total_kl += kl_loss.item()

                avg_val_elbo = val_total_elbo / len(val_dataloader)
                avg_recon = val_total_recon / len(val_dataloader)
                avg_kl = val_total_kl / len(val_dataloader)

                # Early stopping logic
                if avg_val_elbo > best_val_elbo:
                    best_val_elbo = avg_val_elbo
                    patience_counter = 0
                    best_state_dict = self.state_dict()
                    # tqdm.write("Best model updated")  # Print to console without disrupting the progress bar
                else:
                    patience_counter += 1
                    # tqdm.write(f"Patience counter: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    tqdm.write(f"Early stopping triggered after {epoch} epochs")
                    if best_state_dict is not None:
                        self.load_state_dict(best_state_dict)
                    epoch_pbar.close()  # Close the progress bar early
                    break

        epoch_pbar.close()
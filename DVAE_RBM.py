import anndata
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import kaiwu as kw
import time
import os
import pickle


class scDataset(Dataset):
    def __init__(self, anndata_info, batch_indices=None):
        """
        Dataset class for single-cell RNA data with batch indices.

        Args:
            anndata_info (np.ndarray): Dense RNA data from AnnData.X
            batch_indices (np.ndarray or torch.Tensor, optional): Batch indices for each sample
        """
        self.rna_tensor = torch.tensor(anndata_info, dtype=torch.float32)

        # Handle batch_indices
        if batch_indices is not None:
            self.batch_indices = batch_indices
            if len(self.batch_indices) != self.rna_tensor.shape[0]:
                raise ValueError("Length of batch_indices must match number of samples in anndata_info")
        else:
            self.batch_indices = torch.zeros(self.rna_tensor.shape[0], dtype=torch.long)

    def __len__(self):
        return self.rna_tensor.shape[0]

    def __getitem__(self, idx):
        return self.rna_tensor[idx, :], self.batch_indices[idx]


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, normalization_method="batch"):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if normalization_method == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif normalization_method == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError("normalization_method must be 'batch' or 'layer'")
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.norm(self.fc1(x)))
        q_logits = self.fc2(h)
        return q_logits


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, normalization_method="batch"):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        if normalization_method == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif normalization_method == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError("normalization_method must be 'batch' or 'layer'")
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, zeta):
        h = F.relu(self.norm(self.fc1(zeta)))
        x_recon = self.fc2(h)
        return x_recon


class RBM(nn.Module):
    def __init__(self, latent_dim, sample_method="gibbs"):
        super(RBM, self).__init__()
        self.h = nn.Parameter(torch.zeros(latent_dim))
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.001)  # 对称权重
        self.latent_dim = latent_dim
        self.sample_method = sample_method  # 修改位置：使用传入的 sample_method 参数
        # self.ising_matrix = self._create_ising_matrix(self.latent_dim)
        self.worker = kw.classical.SimulatedAnnealingOptimizer(initial_temperature=1000,
                                                               alpha=0.5,
                                                               cutoff_temperature=0.001,
                                                               iterations_per_t=10,
                                                               size_limit=10,
                                                               rand_seed=512,
                                                               )


        # kw.utils.set_log_level("INFO")
        # kw.utils.CheckpointManager.save_dir = f"./cim_test/"
        # self.worker = kw.cim.CIMOptimizer(user_name='chenshaobo',
        #                                   password='aivuq2411oau12!',
        #                                   cim_task_manager_domain='test',
        #                                   task_name="666",
        #                                   wait=True,
        #                                   project_no='25045619')
        # self.worker = kw.cim.CIMOptimizer(user_name='chenshaobo',
        #                                   password='bafv73q932bfa',
        #                                   cim_task_manager_domain='prod',
        #                                   task_name="666",
        #                                   wait=True,
        #                                   project_no='25045619')

    def get_para(self):
        return self.W, self.h

    def energy(self, z):
        z = z.float()
        h_term = torch.sum(z * self.h, dim=-1)
        w_term = torch.sum((z @ self.W) * z, dim=-1)  # 注意对称性
        return h_term + w_term

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
        # 水平拼接左右部分并取负
        ising_matrix = -torch.cat([left_part, right_part], dim=1)  # 最终形状 (d+1, d+1)
        return ising_matrix.cpu().detach().numpy()

    def adjust_precision(self, ising_matrix, method='adjust'):
        if method == 'scale':
            return np.round(ising_matrix * 100, 2)
        elif method == 'adjust':
            return kw.cim.adjust_ising_matrix_precision(ising_matrix, bit_width=14)
        elif method == 'truncate':
            # 2可以控制在30精度
            return np.round(ising_matrix, 2)
        else:
            print("no adjust!")
            return ising_matrix

    def gibbs_sampling(self, num_samples, steps=1):
        start_time = time.time()
        z = torch.randint(0, 2, (num_samples, self.latent_dim), dtype=torch.float).to(self.h.device)
        for _ in range(steps):
            probs = torch.sigmoid(self.h + z @ self.W)
            z = (torch.rand_like(z) < probs).float()
        return z, (time.time()-start_time)

    def ising_sampling_noise(self, number_of_samples, number_of_hidden_units):
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

        self_strength = 0.9
        neighbor_strength = 0.1
        noise_strength = 0.12

        chain_state = torch.zeros(total_units, device=self.h.device)
        samples = []
        for _ in range(number_of_samples):
            chain_state = self.torch_map_clip(chain_state, adjacency_matrix, bias_terms, self_strength,
                                              neighbor_strength, noise_strength)
            samples.append(chain_state.clone())
        result = torch.stack(samples, dim=0)
        latent_variable = 0.5 * (torch.sign(result) + 1)
        return latent_variable[:, :number_of_hidden_units]

    def torch_map_clip(self, chain_state, adjacency_matrix, bias_terms, self_strength, neighbor_strength,
                       noise_strength):
        noise = torch.randn(chain_state.shape, device=chain_state.device) * noise_strength
        out = self_strength * chain_state + neighbor_strength * torch.matmul(adjacency_matrix,
                                                                             chain_state) + 0.40 * neighbor_strength * bias_terms + noise
        return torch.clamp(out, min=-0.4, max=0.4)

    def ising_sampling_sa(self, number_of_samples, number_of_hidden_units, fold_id, step, behavior):
        # FsatSA
        self.worker.size_limit = number_of_samples
        ising_matrix = self._create_ising_matrix(number_of_hidden_units)
        # 调整精度
        ising_matrix = self.adjust_precision(ising_matrix, method="adjust")
        self.ising_matrix = ising_matrix
        self.worker.task_name = f"fold-{fold_id}_step-{step}_{behavior}"
        output = self.worker.solve(ising_matrix)
        result = [sample[:-1] * sample[-1] for sample in output]
        result = kw.sampler.spin_to_binary(np.array(result))
        return torch.tensor(result[:, :number_of_hidden_units], device=self.h.device, dtype=torch.float32)

    def compute_gradients(self, positive_latent_variable, fold_id, step, number_of_negative_samples=64):
        positive_latent_variable = positive_latent_variable.float()
        positive_hidden_gradient = positive_latent_variable.mean(dim=0)
        positive_weight_gradient = torch.einsum('bi,bj->ij', positive_latent_variable,
                                                positive_latent_variable) / positive_latent_variable.size(0)
        if self.sample_method == "ising_noise":
            negative_latent_variable = self.ising_sampling_noise(number_of_negative_samples, self.latent_dim)
        elif self.sample_method == "ising_sa":
            negative_latent_variable = self.ising_sampling_sa(number_of_negative_samples, self.latent_dim,
                                                              fold_id=fold_id, step=step, behavior="_")
            sampling_time = 0
        elif self.sample_method == "gibbs":
            negative_latent_variable, sampling_time = self.gibbs_sampling(number_of_negative_samples)
        else:
            raise ValueError("Invalid sample method")
        negative_hidden_gradient = negative_latent_variable.mean(dim=0)
        negative_weight_gradient = torch.einsum('bi,bj->ij', negative_latent_variable,
                                                negative_latent_variable) / negative_latent_variable.size(0)
        hidden_gradient = positive_hidden_gradient - negative_hidden_gradient
        weight_gradient = positive_weight_gradient - negative_weight_gradient
        weight_gradient = (weight_gradient + weight_gradient.T) / 2
        return {'hidden_biases': hidden_gradient, 'weights': weight_gradient}, sampling_time


class DVAE_RBM(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 latent_dim=256,
                 beta=0.5,
                 beta_kl=0.0001,
                 normalization_method="batch",
                 sample_method='gibbs',
                 device=torch.device('cpu')):
        super(DVAE_RBM, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.beta_kl = beta_kl
        self.device = device
        self.normalization_method = normalization_method
        self.sample_method = sample_method

        self.input_dim = None
        self.n_batches = None
        self.encoder = None
        self.decoder = None
        self.rbm = None

        self.adata = None
        self.batch_key = None
        self.batch_indices = None

        self.to(device)

    def set_adata(self,
                  adata: anndata.AnnData,
                  batch_key='batch'):
        """
        Store AnnData object, set input_dim, and initialize model components.
        """
        self.adata = adata.copy()
        self.batch_key = batch_key

        self.input_dim = adata.X.shape[1]

        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

        batch_categories = adata.obs[batch_key].astype('category')
        self.batch_indices = torch.tensor(batch_categories.cat.codes.values, dtype=torch.long)
        self.n_batches = len(batch_categories.cat.categories)

        self.encoder = Encoder(
            self.input_dim,
            self.hidden_dim,
            self.latent_dim,
            normalization_method=self.normalization_method
        ).to(self.device)

        self.decoder = Decoder(
            self.latent_dim + self.n_batches,
            self.hidden_dim,
            self.input_dim,
            normalization_method=self.normalization_method
        ).to(self.device)

        self.rbm = RBM(self.latent_dim, sample_method=self.sample_method).to(self.device)

        print(f"Set AnnData with input_dim={self.input_dim}, {self.n_batches} batches")

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
        return F.one_hot(indices, num_classes=self.n_batches).float().to(self.device)

    def kl_divergence(self, z, q, behavior, fold_id, step):
        q = torch.clamp(q, min=1e-7, max=1 - 1e-7)
        log_q = z * torch.log(q) + (1 - z) * torch.log(1 - q)
        entropy = -log_q.sum(dim=-1)
        energy_pos = self.rbm.energy(z)
        if self.sample_method == "gibbs":
            z_negative, _ = self.rbm.gibbs_sampling(z.size(0))
        elif self.sample_method == "ising_noise":
            z_negative = self.rbm.ising_sampling_noise(z.size(0), self.latent_dim)
        elif self.sample_method == "ising_sa":
            z_negative = self.rbm.ising_sampling_sa(z.size(0), self.latent_dim, fold_id=fold_id, step=step,
                                                    behavior=behavior)
        energy_neg = self.rbm.energy(z_negative)
        logZ = energy_neg.mean()
        kl = (energy_pos - entropy + logZ).mean()
        return energy_pos,energy_neg, entropy, kl

    def forward(self, x, batch_indices, step, fold_id):
        batch_one_hot = self._get_batch_one_hot(batch_indices)

        q_logits = self.encoder(x)
        rho = Uniform(0, 1).sample(q_logits.shape).to(x.device)
        zeta, z, q = self.reparameterize(q_logits, rho)

        # decoder_input = torch.cat([zeta, batch_one_hot], dim=-1)
        # x_recon = self.decoder(decoder_input)
        decoder_input = torch.cat([zeta, batch_one_hot], dim=-1)
        x_recon = self.decoder(decoder_input)

        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        if self.training:
            energy_pos,energy_neg, entropy,kl_loss = self.kl_divergence(z, q, behavior='_', step=step, fold_id=fold_id)
        else:
            energy_pos,energy_neg, entropy,kl_loss = self.kl_divergence(z, q, behavior='_', step=step, fold_id=fold_id)

        elbo = -recon_loss - self.beta_kl * kl_loss
        return elbo, recon_loss, kl_loss, z, zeta, energy_pos,energy_neg, entropy

    def get_representation(self,
                           step, fold_id,
                           adata=None,
                           batch_size=5000):
        if adata is None and self.adata is None:
            raise ValueError("No AnnData object provided or set")
        adata = adata if adata is not None else self.adata

        adata_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        batch_indices = torch.tensor(adata.obs[self.batch_key].astype('category').cat.codes.values, dtype=torch.long)

        dataset = scDataset(adata_array, batch_indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.eval()
        latent_reps = []
        with torch.no_grad():
            for x, batch_idx in dataloader:
                x = x.to(self.device)
                batch_idx = batch_idx.to(self.device)
                _, _, _, z, zeta, _ ,_,_= self(x, batch_idx, step=step, fold_id=fold_id)
                latent_reps.append(zeta.cpu().numpy())

        reps = np.concatenate(latent_reps, axis=0)
        print(f"Latent representation shape: {reps.shape}")
        return reps

    def fit(self,
            adata,
            val_percentage=0.1,
            batch_size=128,
            epochs=100,
            lr=1e-4,
            rbm_lr=1e-3,
            early_stopping=True,
            early_stopping_patience=10,
            n_epochs_kl_warmup=None,
            verbose=0,
            fold_id=None,
            output_dir = None):

        if adata is None and self.adata is None:
            raise ValueError("No AnnData object provided or set")
        adata = adata if adata is not None else self.adata

        adata_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        batch_indices = torch.tensor(adata.obs[self.batch_key].astype('category').cat.codes.values, dtype=torch.long)


        if early_stopping:
            train_indices, val_indices = train_test_split(
                np.arange(adata.shape[0]), test_size=val_percentage, random_state=0
            )
            split = {'train_indices':train_indices,'val_indices':val_indices}
            with open(output_dir+'/train_data_split.pkl', 'wb') as f:
                pickle.dump(split, f)  # 序列化数据到文件
            print(f"成功保存数据到 {output_dir+'/train_data_split.pkl'}")

            adata_train_array = adata_array[train_indices]
            adata_val_array = adata_array[val_indices]
            train_batch_indices = batch_indices[train_indices]
            val_batch_indices = batch_indices[val_indices]

            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(adata_val_array, dtype=torch.float32),
                val_batch_indices
            )
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            adata_train_array = adata_array
            train_batch_indices = batch_indices

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(adata_train_array, dtype=torch.float32),
            train_batch_indices
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        rbm_optimizer = torch.optim.Adam(self.rbm.parameters(), lr=rbm_lr)

        best_val_elbo = float('-inf')
        patience_counter = 0
        epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Progress", total=epochs)
        best_state_dict = None

        kl_warmup_epochs = n_epochs_kl_warmup or epochs

        # 初始化存储中间结果的变量
        intermediate_results = {
            # 'rbm_params': [],
            'all_train_elbo': [],
            'all_train_recon_loss': [],
            'all_train_kl': [],
            'all_val_elbo': [],
            'all_val_recon_loss' : [],
            'all_val_kl' : [],
            # 'ising_matrix': [],
            # 'precision': [],
            'time': [],
            'sampling_time' : [],
            'train_pos_energy' : [],
            'train_neg_energy': [],
            'train_entropy': [],
            'val_pos_energy' : [],
            'val_neg_energy': [],
            'val_entropy': [],

        } if verbose == 1 else None

        # train_step = 1
        # val_step = 1
        step = 0

        for epoch in epoch_pbar:
            self.train()
            total_elbo, total_recon, total_kl = 0, 0, 0
            for x, batch_idx in train_dataloader:
                strat_time = time.time()
                x = x.to(self.device)
                optimizer.zero_grad()
                rbm_optimizer.zero_grad()

                elbo, recon_loss, kl_loss, z, zeta, energy_pos,energy_neg, entropy = self(x, batch_idx, step=step, fold_id=fold_id)
                loss = -elbo
                loss.backward()
                # if verbose == 1:
                # with torch.no_grad():
                # current_W = self.rbm.W.detach().clone().cpu().numpy()
                # current_h = self.rbm.h.detach().clone().cpu().numpy()
                # intermediate_results['rbm_params'].append({'W': current_W, 'h': current_h})
                # ising_matrix = self.rbm.ising_matrix.clone().cpu().numpy()
                # ising_matrix = self.rbm.ising_matrix
                # intermediate_results['ising_matrix'].append(ising_matrix)
                # print(intermediate_results['ising_matrix'])
                # precision = kw.cim.calculate_ising_matrix_bit_width(ising_matrix, bit_width=30)
                # intermediate_results['precision'].append(precision)
                # print(precision)
                # rbm_grads, sampling_time = self.rbm.compute_gradients(z.detach(), fold_id=fold_id, step=step)
                # intermediate_results['sampling_time'].append(sampling_time)

                # with torch.no_grad():
                #     # 注意：这里直接赋值梯度
                #     self.rbm.h.grad = rbm_grads['hidden_biases']
                #     self.rbm.W.grad = rbm_grads['weights']

                optimizer.step()
                rbm_optimizer.step()

                # 这里rbm的参数产生了变化
                step += 1

                total_elbo += elbo.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

                if verbose == 1:
                    iter_duration = time.time() - strat_time
                    intermediate_results['time'].append(iter_duration)

            avg_elbo = total_elbo / len(train_dataloader)
            avg_recon = total_recon / len(train_dataloader)
            avg_kl = total_kl / len(train_dataloader)
            epoch_pbar.set_postfix({
                'KL_weight': f'{self.beta_kl}',
                'ELBO': f'{avg_elbo:.4f}',
                'Recon': f'{avg_recon:.4f}',
                'KL': f'{avg_kl:.4f}'
            })
            if verbose == 1:
                intermediate_results['all_train_elbo'].append(avg_elbo)
                intermediate_results['all_train_recon_loss'].append(avg_recon)
                intermediate_results['all_train_kl'].append(avg_kl)
                intermediate_results['train_pos_energy'].append(energy_pos)
                intermediate_results['train_neg_energy'].append(energy_neg)
                intermediate_results['train_entropy'].append(entropy)
                # os.makedirs(f"{output_dir}/models/", exist_ok=True)
                # model_save_path = f'{output_dir}/models/model_fold{fold_id}_epoch{epoch}.pth'
                # torch.save(self.state_dict(), model_save_path)

                # with open(os.path.join("./log", f"epoch_log_fold{fold_id}.txt"), "a") as log_file:
                #     log_file.write(
                #         f"Epoch {epoch}: ELBO={avg_elbo:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Time={iter_duration:.2f}s\n")

            if early_stopping:
                self.eval()
                val_total_elbo, val_total_recon, val_total_kl = 0, 0, 0
                for x, batch_idx in val_dataloader:
                    x = x.to(self.device)
                    with torch.no_grad():
                        elbo, recon_loss, kl_loss, z, zeta, energy_pos,energy_neg, entropy = self(x, batch_idx, step=step, fold_id=fold_id)
                    val_total_elbo += elbo.item()
                    val_total_recon += recon_loss.item()
                    val_total_kl += kl_loss.item()

                avg_val_elbo = val_total_elbo / len(val_dataloader)
                avg_val_recon = val_total_recon / len(val_dataloader)
                avg_val_kl = val_total_kl / len(val_dataloader)
                if verbose == 1:
                    intermediate_results['all_val_elbo'].append(avg_val_elbo)
                    intermediate_results['all_val_recon_loss'].append(avg_val_recon)
                    intermediate_results['all_val_kl'].append(avg_val_kl)
                    intermediate_results['val_pos_energy'].append(energy_pos)
                    intermediate_results['val_neg_energy'].append(energy_neg)
                    intermediate_results['val_entropy'].append(entropy)

                if avg_val_elbo > best_val_elbo:
                    best_val_elbo = avg_val_elbo
                    patience_counter = 0
                    best_state_dict = self.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    tqdm.write(f"Early stopping triggered after {epoch} epochs")
                    if best_state_dict is not None:
                        self.load_state_dict(best_state_dict)
                    epoch_pbar.close()
                    break

        epoch_pbar.close()

        if verbose == 1:
            return intermediate_results
        else:
            return None


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, normalization_method="batch"):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if normalization_method == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif normalization_method == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError("normalization_method must be 'batch' or 'layer'")
        self.dropout = nn.Dropout(0.1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.norm(self.fc1(x)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAE(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 latent_dim=256,
                 beta_kl=0.0001,
                 normalization_method="batch",  # 修改位置
                 device=torch.device('cpu')):
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.normalization_method = normalization_method  # 修改位置
        self.device = device

        self.input_dim = None
        self.n_batches = None
        self.encoder = None
        self.decoder = None

        self.adata = None
        self.batch_key = None
        self.batch_indices = None

        self.to(device)

    def set_adata(self,
                  adata: anndata.AnnData,
                  batch_key='batch'):
        """
        Store AnnData object, set input_dim, and initialize model components.
        """
        self.adata = adata.copy()
        self.batch_key = batch_key

        self.input_dim = adata.X.shape[1]

        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

        batch_categories = adata.obs[batch_key].astype('category')
        self.batch_indices = torch.tensor(batch_categories.cat.codes.values, dtype=torch.long)
        self.n_batches = len(batch_categories.cat.categories)

        # 修改位置：传入 normalization_method 给 VAEEncoder 和 Decoder
        self.encoder = VAEEncoder(
            self.input_dim,
            self.hidden_dim,
            self.latent_dim,
            normalization_method=self.normalization_method  # 修改位置
        ).to(self.device)

        self.decoder = Decoder(
            self.latent_dim + self.n_batches,
            self.hidden_dim,
            self.input_dim,
            normalization_method=self.normalization_method  # 修改位置
        ).to(self.device)

        print(f"Set AnnData with input_dim={self.input_dim}, {self.n_batches} batches")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def kl_divergence(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def _get_batch_one_hot(self, indices):
        return F.one_hot(indices, num_classes=self.n_batches).float().to(self.device)

    def forward(self, x, batch_indices):
        batch_one_hot = self._get_batch_one_hot(batch_indices)

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        decoder_input = torch.cat([z, batch_one_hot], dim=-1)
        x_recon = self.decoder(decoder_input)

        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl_loss = self.kl_divergence(mu, logvar)
        elbo = -recon_loss - self.beta_kl * kl_loss

        return elbo, recon_loss, kl_loss, z

    def fit(self,
            adata=None,
            val_percentage=0.1,
            batch_size=128,
            epochs=100,
            lr=1e-3,
            early_stopping=True,
            early_stopping_patience=10,
            n_epochs_kl_warmup=None,
            verbose=0
            ):
        if adata is None and self.adata is None:
            raise ValueError("No AnnData object provided or set")
        adata = adata if adata is not None else self.adata

        adata_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        batch_indices = torch.tensor(adata.obs[self.batch_key].astype('category').cat.codes.values, dtype=torch.long)

        # 初始化存储中间结果的变量
        intermediate_results = {
            'rbm_params': [],
            'all_train_elbo': [],
            'all_val_elbo': []
        } if verbose == 1 else None

        if early_stopping:
            train_indices, val_indices = train_test_split(
                np.arange(adata.shape[0]), test_size=val_percentage, random_state=0
            )

            adata_train_array = adata_array[train_indices]
            adata_val_array = adata_array[val_indices]
            train_batch_indices = batch_indices[train_indices]
            val_batch_indices = batch_indices[val_indices]

            val_dataset = scDataset(adata_val_array, val_batch_indices)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            adata_train_array = adata_array
            train_batch_indices = batch_indices

        train_dataset = scDataset(adata_train_array, train_batch_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_val_elbo = float('-inf')
        patience_counter = 0
        epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Progress", total=epochs)
        best_state_dict = None

        kl_warmup_epochs = n_epochs_kl_warmup or epochs

        train_step = 1
        val_step = 1

        for epoch in epoch_pbar:
            self.train()
            total_elbo, total_recon, total_kl = 0, 0, 0
            for x, batch_idx in train_dataloader:
                x = x.to(self.device)
                batch_idx = batch_idx.to(self.device)
                optimizer.zero_grad()

                elbo, recon_loss, kl_loss, z = self(x, batch_idx)
                loss = -elbo
                loss.backward()
                optimizer.step()

                total_elbo += elbo.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            avg_elbo = total_elbo / len(train_dataloader)
            avg_recon = total_recon / len(train_dataloader)
            avg_kl = total_kl / len(train_dataloader)
            if verbose == 1:
                intermediate_results['all_train_elbo'].append(avg_elbo)

                # with open(os.path.join(output_dir, f"epoch_log_fold{fold_id}.txt"), "a") as log_file:
                #     log_file.write(
                #         f"Epoch {epoch}: ELBO={avg_elbo:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Time={epoch_duration:.2f}s\n")

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
                        elbo, recon_loss, kl_loss, z = self(x, batch_idx)

                    val_total_elbo += elbo.item()
                    val_total_recon += recon_loss.item()
                    val_total_kl += kl_loss.item()

                avg_val_elbo = val_total_elbo / len(val_dataloader)
                if verbose == 1:
                    intermediate_results['all_val_elbo'].append(avg_val_elbo)

                if avg_val_elbo > best_val_elbo:
                    best_val_elbo = val_total_elbo / len(val_dataloader)
                    patience_counter = 0
                    best_state_dict = self.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    tqdm.write(f"Early stopping triggered after {epoch} epochs")
                    if best_state_dict is not None:
                        self.load_state_dict(best_state_dict)
                    epoch_pbar.close()
                    break
        epoch_pbar.close()

        if verbose == 1:
            return intermediate_results
        else:
            return None

    def get_representation(self,
                           adata=None,
                           batch_size=128):
        if adata is None and self.adata is None:
            raise ValueError("No AnnData object provided or set")
        adata = adata if adata is not None else self.adata

        adata_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        batch_indices = torch.tensor(adata.obs[self.batch_key].astype('category').cat.codes.values, dtype=torch.long)

        dataset = scDataset(adata_array, batch_indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        self.eval()
        latent_reps = []
        with torch.no_grad():
            for x, batch_idx in dataloader:
                x = x.to(self.device)
                batch_idx = batch_idx.to(self.device)
                _, _, _, z = self(x, batch_idx)
                latent_reps.append(z.cpu().numpy())

        reps = np.concatenate(latent_reps, axis=0)
        print(f"Latent representation shape: {reps.shape}")
        return reps

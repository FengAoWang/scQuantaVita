#!/usr/bin/env python

# @Time    : 2025/7/1 19:20
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : AE.py


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



class AEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, normalization_method="batch"):
        super(AEEncoder, self).__init__()
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
        z = self.fc2(h)
        return z


class AEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, normalization_method="batch"):
        super(AEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        if normalization_method == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif normalization_method == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError("normalization_method must be 'batch' or 'layer'")
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.norm(self.fc1(z)))
        x_recon = self.fc2(h)
        return x_recon


class AE(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 latent_dim=256,
                 normalization_method="batch",
                 device=torch.device('cpu')):
        super(AE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.normalization_method = normalization_method
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

        self.encoder = AEEncoder(
            self.input_dim,
            self.hidden_dim,
            self.latent_dim,
            normalization_method=self.normalization_method
        ).to(self.device)

        self.decoder = AEDecoder(
            self.latent_dim + self.n_batches,
            self.hidden_dim,
            self.input_dim,
            normalization_method=self.normalization_method
        ).to(self.device)

        print(f"Set AnnData with input_dim={self.input_dim}, {self.n_batches} batches")

    def _get_batch_one_hot(self, indices):
        return F.one_hot(indices, num_classes=self.n_batches).float().to(self.device)

    def forward(self, x, batch_indices):
        batch_one_hot = self._get_batch_one_hot(batch_indices)

        z = self.encoder(x)
        decoder_input = torch.cat([z, batch_one_hot], dim=-1)
        x_recon = self.decoder(decoder_input)

        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        return -recon_loss, recon_loss, z

    def fit(self,
            adata=None,
            val_percentage=0.1,
            batch_size=128,
            epochs=100,
            lr=1e-3,
            early_stopping=True,
            early_stopping_patience=10,
            n_epochs_kl_warmup=None,
            verbose=0,
            fold_id = None,
            output_dir = None
            ):
        if adata is None and self.adata is None:
            raise ValueError("No AnnData object provided or set")
        adata = adata if adata is not None else self.adata

        adata_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        batch_indices = torch.tensor(adata.obs[self.batch_key].astype('category').cat.codes.values, dtype=torch.long)

        # 初始化存储中间结果的变量
        intermediate_results = {
            'all_train_recon': [],
            'all_val_recon_loss': []
        } if verbose == 1 else None

        if early_stopping:
            train_indices, val_indices = train_test_split(
                np.arange(adata.shape[0]), test_size=val_percentage, random_state=0
            )
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
            total_elbo, total_recon = 0, 0
            for x, batch_idx in train_dataloader:
                x = x.to(self.device)
                batch_idx = batch_idx.to(self.device)
                optimizer.zero_grad()

                elbo, recon_loss, z = self(x, batch_idx)
                loss = -elbo
                loss.backward()
                optimizer.step()

                total_elbo += elbo.item()
                total_recon += recon_loss.item()

            avg_elbo = total_elbo / len(train_dataloader)
            avg_recon = total_recon / len(train_dataloader)

            if verbose == 1:
                intermediate_results['all_train_recon'].append(avg_elbo)

            epoch_pbar.set_postfix({
                'ELBO': f'{avg_elbo:.4f}',
                'Recon': f'{avg_recon:.4f}'
            })

            if early_stopping:
                self.eval()
                val_total_elbo, val_total_recon = 0, 0
                for x, batch_idx in val_dataloader:
                    x = x.to(self.device)
                    batch_idx = batch_idx.to(self.device)
                    with torch.no_grad():
                        elbo, recon_loss, z = self(x, batch_idx)

                    val_total_elbo += elbo.item()
                    val_total_recon += recon_loss.item()

                avg_val_elbo = val_total_elbo / len(val_dataloader)
                avg_val_recon = val_total_recon / len(val_dataloader)
                if verbose == 1:
                    intermediate_results['all_val_recon_loss'].append(avg_val_recon)
                    os.makedirs(f"{output_dir}/models/", exist_ok=True)
                    model_save_path = f'{output_dir}/models/model_fold{fold_id}_epoch{epoch}.pth'
                    torch.save(self.state_dict(), model_save_path)

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
                _, _, z = self(x, batch_idx)
                latent_reps.append(z.cpu().numpy())

        reps = np.concatenate(latent_reps, axis=0)
        print(f"Latent representation shape: {reps.shape}")
        return reps
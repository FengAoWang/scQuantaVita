from sklearn.model_selection import KFold
import os
import pickle
from sklearn import metrics
import torch.multiprocessing as mp
import logging
import matplotlib.pyplot as plt
import torch
import scanpy as sc
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection


class GridSearchConfig:
    def __init__(self,
                 hidden_dim=512,
                 latent_dim=256,
                 beta=0.5,
                 beta_kl=0.0001,
                 normalization_method="batch",  # "batch" or "layer"
                 sample_method="gibbs",  # "gibbs" or "ising_noise" or "ising_fsa"
                 lr=1e-2,
                 rbm_lr=1e-2,
                 epochs=500,
                 batch_size=2048,
                 early_stopping_patience=10,
                 seed=42):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.beta_kl = beta_kl
        self.normalization_method = normalization_method
        self.sample_method = sample_method
        self.lr = lr
        self.rbm_lr = rbm_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        print(f"batch size:{self.batch_size}")

    def __str__(self):
        return (f"norm-{self.normalization_method}_"
                f"sample-{self.sample_method}_"
                f"latent-{self.latent_dim}_"
                f"beta_kl-{self.beta_kl}")


def split_data(dataset, adata):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    output_dir = f'split_indices/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)

    n_samples = adata.X.shape[0]

    all_folds = {
        'train': [],
        'test': []
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(range(n_samples))):
        all_folds['train'].append(train_idx)
        all_folds['test'].append(test_idx)

        print(f"Fold {fold + 1}:")
        print(f"Train indices: {len(train_idx)}")
        print(f"Test indices: {len(test_idx)}")

    output_file = f'{output_dir}/five_fold_indices.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_folds, f)


def load_fold_indices(dataset, fold_num=None):
    """
    Load fold indices from pickle file.

    Parameters:
    - dataset (str): Name of the dataset
    - fold_num (int, optional): Specific fold number to load (0-4). If None, returns all folds

    Returns:
    - If fold_num is specified: tuple of (train_indices, test_indices)
    - If fold_num is None: dictionary containing all folds
    """
    # 设置文件路径
    input_file = f'split_indices/{dataset}/five_fold_indices.pkl'

    # 读取pickle文件
    with open(input_file, 'rb') as f:
        all_folds = pickle.load(f)

    print(f"Loaded indices for dataset: {dataset}")
    print(f"Number of folds: {len(all_folds['train'])}")

    if fold_num is not None:
        if not 0 <= fold_num < len(all_folds['train']):
            raise ValueError(f"fold_num must be between 0 and {len(all_folds['train']) - 1}")

        train_indices = all_folds['train'][fold_num]
        test_indices = all_folds['test'][fold_num]
        print(f"Fold {fold_num}:")
        print(f"Train indices count: {len(train_indices)}")
        print(f"Test indices count: {len(test_indices)}")
        return train_indices, test_indices

    return all_folds


def compute_clusters_performance(adata,
                                 cell_key,
                                 cluster_key='leiden'):
    ARI = metrics.adjusted_rand_score(adata.obs[cell_key], adata.obs[cluster_key])
    AMI = metrics.adjusted_mutual_info_score(adata.obs[cell_key], adata.obs[cluster_key])
    NMI = metrics.normalized_mutual_info_score(adata.obs[cell_key], adata.obs[cluster_key])
    HOM = metrics.homogeneity_score(adata.obs[cell_key], adata.obs[cluster_key])
    FMI = metrics.fowlkes_mallows_score(adata.obs[cell_key], adata.obs[cluster_key])
    return ARI, AMI, NMI, HOM, FMI


def compute_batchEffect(adata,
                        batch_key='batch',
                        label_key='cell_type',
                        x_emb='reps'):
    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=[x_emb],
        batch_correction_metrics=BatchCorrection(),
        bio_conservation_metrics=BioConservation(),
        n_jobs=6,
    )

    bm.benchmark()
    df = bm.get_results(min_max_scale=False)
    print(df.columns)

    # 筛选批次矫正相关的指标

    # ['Isolated labels', 'KMeans NMI', 'KMeans ARI', 'Silhouette label',
    #        'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity',
    #        'PCR comparison', 'Batch correction', 'Bio conservation', 'Total']
    # 只取出 'reps' 对应的行
    reps_metrics = df.loc[x_emb]
    print(reps_metrics.values)
    return reps_metrics.values


def multiprocessing_train_fold(folds, worker_function, func_args_list, train_function):
    processes = []
    return_queue = mp.Queue()
    for i in range(folds):
        p = mp.Process(target=worker_function, args=(func_args_list[i], return_queue, train_function))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results_dict = {}
    for _ in range(folds):
        fold_id, result = return_queue.get()
        if result is None:
            logging.error(f"Fold {fold_id} failed.")
        results_dict[fold_id] = result
    print(results_dict)

    results = [results_dict[i] for i in range(folds)]
    return results


def single_train_fold(func_args_list, train_function):
    results_dict = {}
    for fold_id, train_args in enumerate(func_args_list):
        result = train_function(*train_args)
        if result is None:
            logging.error(f"Fold {fold_id} failed.")
        results_dict[fold_id] = result
    print(results_dict)
    results = [results_dict[i] for i in range(len(func_args_list))]
    return results


def worker_function(func_args, return_queue, train_function):
    fold_id = func_args[3]
    try:
        result = train_function(*func_args)
        return_queue.put((fold_id, result))
    except Exception as e:
        logging.error(f"Error in fold {fold_id}: {str(e)}")
        return_queue.put((fold_id, None))


def print_celltype_counts(adata, key):
    """
    打印 AnnData 对象中 celltype 的类别及数量

    参数:
        adata (anndata.AnnData): 输入的 AnnData 对象（需包含 'celltype' 列在 obs 中）
    """
    # 获取 celltype 列并统计数量
    celltype_series = adata.obs[key]
    celltype_counts = celltype_series.value_counts()  # 返回按数量降序排列的 Series

    # 打印结果
    print(f"=====细胞类别及数量 =====")
    for celltype, count in celltype_counts.items():
        print(f"{celltype}: {count} 个细胞")


def train_fold(Model, adata, dataset_name, fold_id, params, config, output_name, device_id=0, verbose=0):
    config_folder = str(config)
    output_dir = os.path.join(output_name, dataset_name, config_folder)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # train_idx, test_idx = load_fold_indices(dataset_name, fold_num=fold_id)
    # gex_adata_train = adata[train_idx, :].copy()
    # gex_adata_test = adata[test_idx, :].copy()

    gex_adata_train = adata.copy()
    gex_adata_test = adata.copy()

    # print_celltype_counts(gex_adata_train, 'train')
    # print_celltype_counts(gex_adata_test, 'test')

    intermediate_results = None

    if Model.__name__ == "DVAE_RBM":
        model = Model(
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            beta=config.beta,
            beta_kl=config.beta_kl,
            normalization_method=config.normalization_method,
            sample_method=config.sample_method,
            device=device
        )
        model.set_adata(gex_adata_train, batch_key=params['batch_key'])

        intermediate_results = model.fit(
            gex_adata_train,
            epochs=config.epochs,
            lr=config.lr,
            rbm_lr=config.rbm_lr,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            verbose=verbose,
            fold_id=fold_id,
            output_dir=output_dir
        )



    elif Model.__name__ == "VAE":
        model = Model(
            latent_dim=config.latent_dim,
            device=device
        )
        model.set_adata(gex_adata_train, batch_key=params['batch_key'])

        intermediate_results = model.fit(
            gex_adata_train,
            epochs=config.epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            verbose=verbose
        )

    if verbose == 1 and intermediate_results is not None:
        with open(f'{output_dir}/{dataset_name}_{Model.__name__}_fold{fold_id}_intermediate_results.pkl', 'wb') as f:
            pickle.dump(intermediate_results, f)

    gex_adata_test.obsm['qvae_reps'] = model.get_representation(adata=gex_adata_test, batch_size=config.batch_size,
                                                                step=999999,
                                                                fold_id=fold_id)
    latent_test = gex_adata_test.obsm['qvae_reps']

    sc.pp.neighbors(gex_adata_test, use_rep='qvae_reps')
    sc.tl.leiden(gex_adata_test, random_state=42)
    sc.tl.umap(gex_adata_test)
    sc.pl.umap(
        gex_adata_test,
        color=[params['labels_key'], params['batch_key'], 'leiden'],
        show=False,
        title=f"Config: {config_folder} (Fold {fold_id})",
        frameon=False,
        ncols=1
    )

    plt.savefig(f'{output_dir}/{dataset_name}_{Model.__name__}_cell_{fold_id}.png', dpi=1000, bbox_inches='tight')

    if params['labels_key'] in gex_adata_test.obs:
        leiden_ARI, leiden_AMI, leiden_NMI, leiden_HOM, leiden_FMI = compute_clusters_performance(gex_adata_test,
                                                                                                  params[
                                                                                                      'labels_key'])
        print(
            f'Fold {fold_id} Metrics: ARI={leiden_ARI:.4f}, AMI={leiden_AMI:.4f}, NMI={leiden_NMI:.4f}, HOM={leiden_HOM:.4f},FMI={leiden_FMI:.4f}')
        # louvain_ARI, louvain_AMI, louvain_NMI, louvain_HOM, louvain_FMI = compute_clusters_performance(gex_adata_test, DataParams['labels_key'], cluster_key='louvain')

        # scib_values = compute_batchEffect(gex_adata_test, params['batch_key'], params['labels_key'],
        #                                   x_emb='qvae_reps')
        clustering_value = [leiden_ARI, leiden_AMI, leiden_NMI, leiden_HOM, leiden_FMI]
        # clustering_value.extend(scib_values)
        return clustering_value
    else:
        print("Warning: 'cell_type' not found in gex_data.obs. Skipping clustering metrics.")

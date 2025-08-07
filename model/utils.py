from sklearn.model_selection import KFold
import os
import pickle
from sklearn import metrics
import torch.multiprocessing as mp
import logging
import matplotlib.pyplot as plt
import torch
import scanpy as sc
from scipy.stats import entropy
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
from torch.distributions import constraints, NegativeBinomial, Poisson, Beta, Normal
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import softplus
from torch.distributions.distribution import Distribution
from scipy.stats import f
from scipy.stats import kendalltau


plt.rcParams.update({
    'figure.titlesize': 7,  # 控制 suptitle 的字体大小
    'axes.titlesize': 7,  # 坐标轴标题字体大小
    'axes.labelsize': 7,  # 坐标轴标签字体大小
    'xtick.labelsize': 6,  # x轴刻度字体大小
    'ytick.labelsize': 6,  # y轴刻度字体大小
    'legend.fontsize': 6,  # 图例字体大小
    'lines.markersize': 6,  # 标记点大小
    'axes.grid': False,  # 默认显示网格
    'axes.linewidth': 0.5,  # 统一设置x轴和y轴宽度（脊线厚度）
    'ytick.major.width': 0.5,  # y轴主刻度线宽度
    'xtick.major.width': 0.5,  # x轴主刻度线宽度
    'ytick.major.size': 2,  # y轴主刻度线长度
    'xtick.major.size': 2,  # x轴主刻度线长度
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def split_data(dataset, adata):
    # 设置5折交叉验证
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=66)
    # 创建保存目录
    output_dir = f'split_indices/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)
    # 获取样本数量
    n_samples = adata.X.shape[0]
    # 初始化字典来存储所有折的索引
    all_folds = {
        'train': [],
        'test': []
    }

    # 进行5折划分
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(n_samples))):
        # 存储索引
        all_folds['train'].append(train_idx)
        all_folds['test'].append(test_idx)

        # 打印每折的索引数量
        print(f"Fold {fold + 1}:")
        print(f"Train indices: {len(train_idx)}")
        print(f"Test indices: {len(test_idx)}")

    # 保存所有折的索引到一个pickle文件
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

    # 验证加载的数据
    print(f"Loaded indices for dataset: {dataset}")
    print(f"Number of folds: {len(all_folds['train'])}")

    if fold_num is not None:
        # 验证fold_num有效性
        if not 0 <= fold_num < len(all_folds['train']):
            raise ValueError(f"fold_num must be between 0 and {len(all_folds['train']) - 1}")

        # 返回特定fold的训练和测试索引
        train_indices = all_folds['train'][fold_num]
        test_indices = all_folds['test'][fold_num]
        print(f"Fold {fold_num}:")
        print(f"Train indices count: {len(train_indices)}")
        print(f"Test indices count: {len(test_indices)}")
        return train_indices, test_indices

    # 如果未指定fold_num，返回所有folds
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
                        x_emb=None):
    if x_emb is None:
        x_emb = ['reps']
    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=x_emb,
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
    # print(results_dict)

    results = [results_dict[i] for i in range(folds)]
    return results


def worker_function(func_args, return_queue, train_function):
    fold_id = func_args[3]
    try:
        result = train_function(*func_args)
        return_queue.put((fold_id, result))
    except Exception as e:
        logging.error(f"Error in fold {fold_id}: {e}", exc_info=True)
        return_queue.put((fold_id, None))


def train_fold(Model:nn.Module, adata, dataset_name, fold_id, DataParams, TrainingParams, device_id=6):
    from scgraph import scGraph

    output_dir = f'result/{dataset_name}/integration/{Model.__name__}_latentDim{TrainingParams['latent_dim']}_batchSize{TrainingParams['batch_size']}/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'models/{dataset_name}/', exist_ok=True)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Data shape: {adata.X.shape}")
    print(f'use model: {Model.__name__}')
    if Model.__name__ == 'DVAE_RBM':
        model = Model(latent_dim=TrainingParams['latent_dim'],
                      device=device,
                      use_norm=TrainingParams['normaliztion'],
                      beta_kl=TrainingParams['beta_kl'],
                      # use_zinb_decoder=True
                      )
        model.set_adata(adata, batch_key=DataParams['batch_key'])
    
        model.fit(adata,
                  epochs=TrainingParams['epochs'],
                  lr=TrainingParams['lr'],
                  early_stopping_patience=10,
                  n_epochs_kl_warmup=50,
                  batch_size=TrainingParams['batch_size'],
                  # layer_key='counts'
                  )
    elif Model.__name__ == 'VAE':
        model = Model(latent_dim=TrainingParams['latent_dim'],
                      device=device,
                      use_norm=TrainingParams['normaliztion'],
                      beta_kl=TrainingParams['beta_kl'],
                      # use_zinb_decoder=True
                      )
        print('model constructed')
        model.set_adata(adata, batch_key=DataParams['batch_key'])
        print('set adata')

        model.fit(adata,
                  epochs=TrainingParams['epochs'],
                  lr=TrainingParams['lr'],
                  early_stopping_patience=10,
                  n_epochs_kl_warmup=50,
                  batch_size=TrainingParams['batch_size'],
                  # layer_key='counts'
                  )

    torch.save(model.state_dict(), f'models/{dataset_name}/RBM_VAE_model{fold_id}_latentDim{TrainingParams['latent_dim']}_batchsize{TrainingParams['batch_size']}.pt')

    # Add latent representations to AnnData
    adata.obsm['reps'] = model.get_representation(
        adata,
        # layer_key='counts'
    )
    latent_test = adata.obsm['reps']

    # UMAP and clustering
    sc.pp.neighbors(adata, use_rep='reps')
    sc.tl.leiden(adata, random_state=42)

    sc.tl.umap(adata, min_dist=0.2)
    colors_use = [DataParams['labels_key'], DataParams['batch_key'], 'leiden'] if DataParams['batch_key'] != "" else [DataParams['labels_key'], 'leiden']
    sc.pl.umap(adata,
               color=colors_use,
               show=False,
               frameon=False,
               ncols=1, )
    plt.savefig(
        f'{output_dir}/{dataset_name}_{Model.__name__}_cell_latentDim{TrainingParams["latent_dim"]}_fold{fold_id}.pdf',
        dpi=1000, bbox_inches='tight')

    # sc.pl.spatial(gex_adata_test,
    #               color=colors_use,
    #               spot_size=1.0,
    #               show=False,
    #               frameon=False,
    #               ncols=1)

    # plt.savefig(
    #     f'{output_dir}{dataset_name}_{Model.__name__}_cell_Spatial_latentDim{TrainingParams["latent_dim"]}_fold{fold_id}.pdf',
    #     dpi=1000, bbox_inches='tight')
    # sc.tl.louvain(gex_adata_test, random_state=42)
    # Initialize the graph analyzer
    adata.write(f'{dataset_name}_RBM_fold{fold_id}.h5ad')
    scgraph = scGraph(
        adata_path=f'{dataset_name}_RBM_fold{fold_id}.h5ad',  # Path to AnnData object
        batch_key=DataParams['batch_key'],  # Column name for batch information
        label_key=DataParams['labels_key'],  # Column name for cell type labels
        trim_rate=0.05,  # Trim rate for robust mean calculation
        thres_batch=100,  # Minimum number of cells per batch
        thres_celltype=10,  # Minimum number of cells per cell type
        only_umap=False,  # Only evaluate 2D embeddings (mostly umaps)
    )

    # Run the analysis, return a pandas dataframe
    results = scgraph.main()
    results.to_csv(f'{output_dir}{dataset_name}_RBMVAE_scgraph_fold{fold_id}.csv')

    print('scgraph', results)

    # Clustering metrics
    clustering_value = []
    if DataParams['labels_key'] in adata.obs:
        leiden_ARI, leiden_AMI, leiden_NMI, leiden_HOM, leiden_FMI = compute_clusters_performance(adata,
                                                                                                  DataParams[
                                                                                                      'labels_key'])
        clustering_value.extend([leiden_ARI, leiden_AMI, leiden_NMI, leiden_HOM, leiden_FMI])


        # louvain_ARI, louvain_AMI, louvain_NMI, louvain_HOM, louvain_FMI = compute_clusters_performance(gex_adata_test, DataParams['labels_key'], cluster_key='louvain')

    # if DataParams['batch_key'] in gex_adata_test.obs:
    #
    #     scib_values = compute_batchEffect(gex_adata_test, DataParams['batch_key'], DataParams['labels_key'],
    #                                       x_emb='reps')
    #     clustering_value.extend(scib_values)

    return clustering_value


from sklearn.linear_model import LogisticRegression


def train_logistic_regression_classifier(Model: nn.Module, adata, dataset_name, fold_id, DataParams, TrainingParams, device_id=6):
    """
    Loads a pre-trained model, freezes its parameters, extracts embeddings,
    and then trains a Logistic Regression classifier to validate on the test set.

    Args:
        Model: The neural network model class.
        adata: AnnData object containing gene expression data.
        dataset_name: Name of the dataset.
        fold_id: The cross-validation fold ID.
        DataParams: Dictionary of data-related parameters.
        TrainingParams: Dictionary of training-related parameters.
        device_id: GPU device ID, defaults to 6.

    Returns:
        classification_metrics: A list of classification performance metrics.
    """
    import torch
    import numpy as np
    import os
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import LabelEncoder
    # Assuming these functions are defined elsewhere in your project
    # from model.visualization_func import plot_confusion_matrix
    # from your_utils import load_fold_indices

    # Setup output directory
    output_dir = f'result/{dataset_name}/classification_metrics'
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device for embedding extraction: {device}")

    # Load cross-validation fold indices
    # This function needs to be available in your environment
    train_idx, test_idx = load_fold_indices(dataset_name, fold_num=fold_id)




    # Split data into training and testing sets
    gex_adata_train = adata[train_idx, :].copy()
    gex_adata_test = adata[test_idx, :].copy()

    print(f"Training data shape: {gex_adata_train.X.shape}")
    print(f"Testing data shape: {gex_adata_test.X.shape}")

    # Initialize and load the pre-trained model
    model = Model(latent_dim=TrainingParams['latent_dim'],
                  device=device,
                  # Add other necessary parameters for your model
                 )
    model.set_adata(gex_adata_train, batch_key=DataParams['batch_key'])

    # Load saved model weights
    model_path = f'models/{dataset_name}/{dataset_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Extract embeddings for the training and test sets
    with torch.no_grad():
        gex_adata_train.obsm['reps'] = model.get_representation(gex_adata_train)
        gex_adata_test.obsm['reps'] = model.get_representation(gex_adata_test)

    # Get embeddings and labels
    X_train = gex_adata_train.obsm['reps']
    X_test = gex_adata_test.obsm['reps']

    # Ensure labels key exists
    if DataParams['labels_key'] not in gex_adata_train.obs:
        raise ValueError(f"Label key '{DataParams['labels_key']}' not found in adata.obs")

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(gex_adata_train.obs[DataParams['labels_key']])
    y_test = le.transform(gex_adata_test.obs[DataParams['labels_key']])

    # --- Start of Logistic Regression Specific Code ---
    # Initialize Logistic Regression classifier
    # Note: Logistic Regression runs on the CPU, so GPU parameters are removed.
    lr_classifier = LogisticRegression(
        random_state=0,
        max_iter=100,  # Increased max_iter for convergence on complex datasets
        n_jobs=-1       # Use all available CPU cores
    )

    # Train the Logistic Regression classifier
    print("Training Logistic Regression classifier...")
    lr_classifier.fit(X_train, y_train)

    # Predict on the test set
    print("Evaluating on the test set...")
    y_pred = lr_classifier.predict(X_test)
    # --- End of Logistic Regression Specific Code ---

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    classification_metrics = [accuracy, precision, recall, f1]

    # Save classification results (optional, uncomment to use)
    # results_file = f'{output_dir}/{dataset_name}_LogisticRegression_classification_fold{fold_id}.txt'
    # with open(results_file, 'w') as f:
    #     f.write(f"Accuracy: {accuracy:.4f}\n")
    #     f.write(f"Precision: {precision:.4f}\n")
    #     f.write(f"Recall: {recall:.4f}\n")
    #     f.write(f"F1 Score: {f1:.4f}\n")
    # print(f"Classification metrics saved to {results_file}")

    # Get original label names
    labels = le.classes_

    # Plot confusion matrix (assuming plot_confusion_matrix is defined)
    # plot_confusion_matrix(y_test, y_pred, labels, dataset_name, fold_id, output_dir, Model.__name__)

    return classification_metrics


def train_xgboost_classifier(Model: nn.Module, adata, dataset_name, fold_id, DataParams, TrainingParams, device_id=6):
    import torch
    import xgboost as xgb
    import numpy as np
    import scanpy as sc
    import seaborn as sns
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import os
    from sklearn.preprocessing import LabelEncoder
    from model.visualization_func import plot_confusion_matrix


    """
    加载训练好的模型，冻结参数，提取嵌入，训练XGBoost分类器并在测试集上验证

    参数:
        Model: 神经网络模型类
        adata: AnnData对象，包含基因表达数据
        dataset_name: 数据集名称
        fold_id: 交叉验证折编号
        DataParams: 数据相关参数字典
        TrainingParams: 训练相关参数字典
        device_id: GPU设备ID，默认为6

    返回:
        classification_metrics: 分类性能指标列表
    """
    # 设置输出目录
    output_dir = f'result/{dataset_name}/classification_metrics'
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载交叉验证折索引
    train_idx, test_idx = load_fold_indices(dataset_name, fold_num=fold_id)

    # 划分训练和测试数据
    gex_adata_train = adata[train_idx, :].copy()
    gex_adata_test = adata[test_idx, :].copy()

    print(f"训练数据形状: {gex_adata_train.X.shape}")
    print(f"测试数据形状: {gex_adata_test.X.shape}")

    # 初始化并加载训练好的模型
    model = Model(latent_dim=TrainingParams['latent_dim'],
                  device=device,
                  # normalization_method=TrainingParams['normaliztion']
                  # use_norm=TrainingParams['normaliztion'],
                  # use_zinb_decoder=True
                  )
    model.set_adata(gex_adata_train, batch_key=DataParams['batch_key'])

    # 加载保存的模型权重
    # model_path = f'models/{dataset_name}/RBM_VAE_model{fold_id}_latentDim{TrainingParams['latent_dim']}_batchsize{TrainingParams['batch_size']}.pt'
    model_path = f'models/{dataset_name}/{dataset_name}.pth'

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False

    # 提取训练集和测试集的嵌入
    gex_adata_train.obsm['reps'] = model.get_representation(gex_adata_train)
    gex_adata_test.obsm['reps'] = model.get_representation(gex_adata_test)

    # 获取嵌入和标签
    X_train = gex_adata_train.obsm['reps']
    X_test = gex_adata_test.obsm['reps']

    # 确保标签存在
    if DataParams['labels_key'] not in gex_adata_train.obs:
        raise ValueError(f"标签键 {DataParams['labels_key']} 未在adata.obs中找到")

    # 编码标签
    le = LabelEncoder()
    y_train = le.fit_transform(gex_adata_train.obs[DataParams['labels_key']])
    y_test = le.transform(gex_adata_test.obs[DataParams['labels_key']])

    # 初始化XGBoost分类器
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=0,
        tree_method='hist',  # 使用基于直方图的算法，支持GPU
        device='cuda',  # 明确指定使用GPU
        predictor='cuda',  # 使用GPU预测器
        # gpu_id=device_id,  # 指定GPU设备ID，与神经网络一致
        n_jobs=-1
    )

    # 训练XGBoost分类器
    xgb_classifier.fit(X_train, y_train)

    # 在测试集上预测
    y_pred = xgb_classifier.predict(X_test)

    # 计算分类指标
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    classification_metrics = [accuracy, precision, recall, f1]
    #
    # # 保存分类结果
    # results_file = f'{output_dir}{dataset_name}_XGBoost_classification_fold{fold_id}.txt'
    # with open(results_file, 'w') as f:
    #     f.write(f"Accuracy: {accuracy:.4f}\n")
    #     f.write(f"Precision: {precision:.4f}\n")
    #     f.write(f"Recall: {recall:.4f}\n")
    #     f.write(f"F1 Score: {f1:.4f}\n")

    # 获取原始标签名称
    labels = le.classes_

    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred, labels, dataset_name, fold_id, output_dir, Model.__name__)

    return classification_metrics


def run_pseudotime_on_rep(adata, rep_key='X', root_cell_type='HSPCs', cluster_key='cell_type'):
    """
    在已存在的 adata.obsm[rep_key] 上执行伪时序轨迹分析

    参数:
        adata: AnnData对象，包含 .obsm[rep_key] 作为低维表示
        rep_key: 表示嵌入的key，默认是'reps'
        root_cell_type: 伪时序的根细胞类型，用于设置初始根节点
        cluster_key: 指示细胞类型标签的obs列名

    返回:
        更新后的adata，包含:
            - adata.obs['dpt_pseudotime']: 伪时间值
            - adata.uns['iroot']: 根细胞索引
    """
    # 构图和最近邻构建（基于自定义嵌入）

    sc.pp.neighbors(adata, use_rep=rep_key)

    # 聚类（可选，若已有分群标签可跳过）
    if cluster_key not in adata.obs:
        sc.tl.leiden(adata)

    # 选择根节点
    if root_cell_type in adata.obs[cluster_key].unique():
        root_cells = adata.obs_names[adata.obs[cluster_key] == root_cell_type]
        if len(root_cells) > 0:
            # 使用最近邻图找到一个root cell的索引
            adata.uns['iroot'] = np.flatnonzero(adata.obs_names == root_cells[0])[0]
        else:
            raise ValueError(f"无法找到 {root_cell_type} 作为伪时序起始细胞")
    else:
        raise ValueError(f"{root_cell_type} 不在 {cluster_key} 中")

    # Diffusion Map
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)  # 使用 DPT 拟时序算法
    return adata


def compute_batch_pseudotime(unintegrated_adata, batch_key, root_cell_type='HSPCs', cluster_key='cell_type',
                             rep_key='X'):
    """
    按批次计算未整合数据的伪时序轨迹

    参数:
        unintegrated_adata: 未整合的AnnData对象
        batch_key: 批次标签的obs列名
        root_cell_type: 伪时序的根细胞类型
        cluster_key: 细胞类型标签的obs列名
        rep_key: 用于伪时序的低维表示key

    返回:
        unintegrated_adata: 更新后的AnnData对象，包含每个批次的伪时序值（列名为'dpt_pseudotime_{batch}'）
    """
    # 初始化伪时序列
    unintegrated_adata.obs['dpt_pseudotime'] = np.nan

    # 获取所有批次
    batches = unintegrated_adata.obs[batch_key].unique()
    print(f"发现 {len(batches)} 个批次: {batches}")

    for batch in batches:
        print(f"处理批次: {batch}")
        # 提取当前批次数据
        batch_adata = unintegrated_adata[unintegrated_adata.obs[batch_key] == batch].copy()

        # 确保批次中有足够的细胞和根细胞类型
        if len(batch_adata) < 10:  # 假设最少需要10个细胞
            print(f"警告: 批次 {batch} 细胞数过少 ({len(batch_adata)})，跳过")
            continue

        if root_cell_type not in batch_adata.obs[cluster_key].unique():
            print(f"警告: 批次 {batch} 缺少根细胞类型 {root_cell_type}，跳过")
            continue

        try:
            # 计算PCA（如果需要）
            sc.pp.pca(batch_adata, n_comps=50)

            # 运行伪时序分析
            batch_adata = run_pseudotime_on_rep(
                batch_adata,
                rep_key=rep_key,
                root_cell_type=root_cell_type,
                cluster_key=cluster_key
            )

            # 将伪时序结果保存回原始adata
            batch_cell_ids = batch_adata.obs_names
            unintegrated_adata.obs.loc[batch_cell_ids, f'dpt_pseudotime_{batch}'] = batch_adata.obs['dpt_pseudotime']
            # 合并到主伪时序列（可选，根据需求）
            unintegrated_adata.obs.loc[batch_cell_ids, 'dpt_pseudotime'] = batch_adata.obs['dpt_pseudotime']

        except Exception as e:
            print(f"批次 {batch} 伪时序计算失败: {str(e)}")
            continue

    return unintegrated_adata


def compute_kendall_trajectory_conservation(unintegrated_adata, integrated_adata, pseudotime_key_unint='dpt_pseudotime', pseudotime_key_int='dpt_pseudotime'):
    """
    计算整合前后伪时序的Kendall’s Tau轨迹保守性得分（全局，不按批次分割）。

    参数:
        unintegrated_adata: 未整合的AnnData对象，包含伪时序值
        integrated_adata: 整合后的AnnData对象，包含伪时序值
        pseudotime_key_unint: 未整合数据伪时序的obs列名，默认为'dpt_pseudotime'
        pseudotime_key_int: 整合数据伪时序的obs列名，默认为'dpt_pseudotime'

    返回:
        float: 轨迹保守性得分，范围[0, 1]，1表示完全一致，0表示完全相反或无法计算
    """
    # 确保细胞顺序一致
    common_cells = unintegrated_adata.obs_names.intersection(integrated_adata.obs_names)
    if len(common_cells) < 2:
        return 0.0

    # 提取伪时序值
    unintegrated_pseudotime = unintegrated_adata.obs.loc[common_cells, pseudotime_key_unint]
    integrated_pseudotime = integrated_adata.obs.loc[common_cells, pseudotime_key_int]

    # 过滤NaN值
    valid_cells = unintegrated_pseudotime.notna() & integrated_pseudotime.notna()
    if valid_cells.sum() < 2:
        return 0.0

    # 计算Kendall’s Tau
    tau, _ = kendalltau(unintegrated_pseudotime[valid_cells], integrated_pseudotime[valid_cells])

    # 缩放到[0, 1]
    conservation_score = (tau + 1) / 2
    return conservation_score

def analyze_dpt(Model: nn.Module, adata, dataset_name, fold_id, DataParams, TrainingParams, device_id=6):
    import torch
    import os
    """
    加载训练好的模型，冻结参数，提取嵌入，训练XGBoost分类器并在测试集上验证

    参数:
        Model: 神经网络模型类
        adata: AnnData对象，包含基因表达数据
        dataset_name: 数据集名称
        fold_id: 交叉验证折编号
        DataParams: 数据相关参数字典
        TrainingParams: 训练相关参数字典
        device_id: GPU设备ID，默认为6

    返回:
        classification_metrics: 分类性能指标列表
    """
    sc.pp.filter_cells(adata, min_counts=300)
    # 设置输出目录
    output_dir = f'result/{dataset_name}/dpt/'
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 初始化并加载训练好的模型
    model = Model(latent_dim=TrainingParams['latent_dim'],
                  device=device,
                  use_norm=TrainingParams['normaliztion'],
                  # use_zinb_decoder=True
                  )
    model.set_adata(adata, batch_key=DataParams['batch_key'])
    # 加载保存的模型权重
    model_path = f'models/{dataset_name}/RBM_VAE_model{fold_id}_latentDim{TrainingParams['latent_dim']}_batchsize{TrainingParams['batch_size']}.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # 构造人工伪时序分数
    pseudo_order_dict = {
        'HSPCs': 0,
        'Erythroid progenitors': 1.5,
        'Megakaryocyte progenitors': 1,
        'Erythrocytes': 2
    }
    adata.obs['pseudo_order'] = adata.obs[DataParams['labels_key']].map(pseudo_order_dict).astype(float)

    # 加载交叉验证折索引
    use_cells = ['Erythrocytes', 'Erythroid progenitors', 'HSPCs',
                 'Megakaryocyte progenitors'
                 ]
    use_batch = ['Oetjen_A', 'Oetjen_P', 'Oetjen_U']


    adata = adata[adata.obs[DataParams['labels_key']].isin(use_cells)].copy()
    adata = adata[adata.obs[DataParams['batch_key']].isin(use_batch)].copy()

    unintegrated_adata = adata.copy()

    sc.pp.pca(unintegrated_adata)
    # unintegrated_adata = compute_batch_pseudotime(
    #     unintegrated_adata,
    #     batch_key=DataParams['batch_key'],
    #     rep_key='X',  # 或 'reps'，取决于预处理方式
    #     root_cell_type='HSPCs',
    #     cluster_key=DataParams['labels_key']
    # )

    unintegrated_adata = run_pseudotime_on_rep(
        unintegrated_adata,
        rep_key='X_pca',  # 或 'reps'，取决于预处理方式
        root_cell_type='HSPCs',
        cluster_key=DataParams['labels_key']
    )

    sc.pp.neighbors(unintegrated_adata, use_rep='X')
    sc.tl.umap(unintegrated_adata, min_dist=0.1)
    sc.pl.umap(
        unintegrated_adata,
        color=['dpt_pseudotime', DataParams['labels_key'], DataParams['batch_key']],
        cmap='viridis',
        show=False,
        frameon=False,
        ncols=1, )
    plt.savefig(
        f'{output_dir}{dataset_name}_latent_dpt.pdf',
        dpi=1000, bbox_inches='tight')

    # 提取训练集和测试集的嵌入
    adata = adata[adata.obs[DataParams['labels_key']].isin(use_cells)].copy()

    adata.obsm['reps'] = model.get_representation(adata)
    # adata = adata[adata.obs[DataParams['labels_key']].isin(use_cells)].copy()
    adata = run_pseudotime_on_rep(adata,
                                  rep_key='reps',
                                  root_cell_type='HSPCs',
                                  cluster_key=DataParams['labels_key'])
    # UMAP 用于可视化（不是必须）
    sc.pp.neighbors(adata, use_rep='reps')
    sc.tl.umap(adata, min_dist=0.1)
    sc.pl.umap(
        adata,
        color=['dpt_pseudotime', DataParams['labels_key'], DataParams['batch_key']],
        cmap='viridis',
        show=False,
        frameon=False,
        ncols=1, )
    plt.savefig(
        f'{output_dir}{dataset_name}_{Model.__name__}_latent_dpt_latentDim{TrainingParams["latent_dim"]}_fold{fold_id}.pdf',
        dpi=1000, bbox_inches='tight')

    # 匹配细胞名
    # shared_cells = adata.obs_names.intersection(unintegrated_adata.obs_names)

    # 取伪时序
    pt_integrated = adata.obs['dpt_pseudotime']
    pt_unintegrated = unintegrated_adata.obs['dpt_pseudotime']

    # 计算 Spearman 相关系数
    s = pt_integrated.corr(pt_unintegrated, method='spearman')

    # 转换成 Trajectory Conservation 分数
    trajectory_conservation_score = (s + 1) / 2

    kendall_score= compute_kendall_trajectory_conservation(unintegrated_adata, adata)
    print(f"Trajectory conservation score: {kendall_score:.3f}")

    return trajectory_conservation_score


def broadcast_shape(*shapes, **kwargs):
    """
    Similar to ``np.broadcast()`` but for shapes.
    Equivalent to ``np.broadcast(*map(np.empty, shapes)).shape``.
    :param tuple shapes: shapes of tensors.
    :param bool strict: whether to use extend-but-not-resize broadcasting.
    :returns: broadcasted shape
    :rtype: tuple
    :raises: ValueError
    """
    strict = kwargs.pop("strict", False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError(
                    "shape mismatch: objects cannot be broadcast to a single shape: {}".format(
                        " vs ".join(map(str, shapes))
                    )
                )
    return tuple(reversed(reversed_shape))


class ZeroInflatedDistribution(Distribution):
    """
    Generic Zero Inflated distribution.
    This can be used directly or can be used as a base class as e.g. for
    :class:`ZeroInflatedPoisson` and :class:`ZeroInflatedNegativeBinomial`.
    :param TorchDistribution base_dist: the base distribution.
    :param torch.Tensor gate: probability of extra zeros given via a Bernoulli distribution.
    :param torch.Tensor gate_logits: logits of extra zeros given via a Bernoulli distribution.
    """

    arg_constraints = {
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }

    def __init__(self, base_dist, *, gate=None, gate_logits=None, validate_args=None):
        if (gate is None) == (gate_logits is None):
            raise ValueError(
                "Either `gate` or `gate_logits` must be specified, but not both."
            )
        if gate is not None:
            batch_shape = broadcast_shape(gate.shape, base_dist.batch_shape)
            self.gate = gate.expand(batch_shape)
        else:
            batch_shape = broadcast_shape(gate_logits.shape, base_dist.batch_shape)
            self.gate_logits = gate_logits.expand(batch_shape)
        if base_dist.event_shape:
            raise ValueError(
                "ZeroInflatedDistribution expected empty "
                "base_dist.event_shape but got {}".format(base_dist.event_shape)
            )

        self.base_dist = base_dist.expand(batch_shape)
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    @lazy_property
    def gate(self):
        return logits_to_probs(self.gate_logits)

    @lazy_property
    def gate_logits(self):
        return probs_to_logits(self.gate)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        if "gate" in self.__dict__:
            gate, value = broadcast_all(self.gate, value)
            log_prob = (-gate).log1p() + self.base_dist.log_prob(value)
            log_prob = torch.where(value == 0, (gate + log_prob.exp()).log(), log_prob)
        else:
            gate_logits, value = broadcast_all(self.gate_logits, value)
            base_log_prob = self.base_dist.log_prob(value)
            log_prob_minus_log_gate = -gate_logits + base_log_prob
            log_gate = -softplus(-gate_logits)
            log_prob = log_prob_minus_log_gate + log_gate

            zero_log_prob = softplus(log_prob_minus_log_gate) + log_gate
            log_prob = torch.where(value == 0, zero_log_prob, log_prob)
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.gate.expand(shape)).bool()
            samples = self.base_dist.expand(shape).sample()
            samples = torch.where(mask, samples.new_zeros(()), samples)
        return samples

    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.gate) * (
                self.base_dist.mean ** 2 + self.base_dist.variance
        ) - (self.mean) ** 2

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        gate = self.gate.expand(batch_shape) if "gate" in self.__dict__ else None
        gate_logits = (
            self.gate_logits.expand(batch_shape)
            if "gate_logits" in self.__dict__
            else None
        )
        base_dist = self.base_dist.expand(batch_shape)
        ZeroInflatedDistribution.__init__(
            new, base_dist, gate=gate, gate_logits=gate_logits, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


# Pyro Distributions.zero_inflated
class ZeroInflatedNegativeBinomial(ZeroInflatedDistribution):
    """
    A Zero Inflated Negative Binomial distribution.
    :param total_count: non-negative number of negative Bernoulli trials.
    :type total_count: float or torch.Tensor
    :param torch.Tensor probs: Event probabilities of success in the closed interval [0, 1].
    :param torch.Tensor logits: Event log-odds for probabilities of success.
    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor gate_logits: logits of extra zeros.
    """

    arg_constraints = {
        "total_count": constraints.greater_than_eq(0), # Should the prob be half-open?
        "probs": constraints.interval(0.0, 1.0),
        "logits": constraints.real,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count,
        *,
        probs=None,
        logits=None,
        gate=None,
        gate_logits=None,
        validate_args=None
    ):

        base_dist = NegativeBinomial(
            total_count=total_count,
            probs=probs,
            logits=logits,
            validate_args=False,
        )
        base_dist._validate_args = validate_args

        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )

    @property
    def total_count(self):
        return self.base_dist.total_count

    @property
    def probs(self):
        return self.base_dist.probs

    @property
    def logits(self):
        return self.base_dist.logits


class LossFunction:

    @staticmethod
    def kl_loss(mu: torch.tensor,
                log_var: torch.tensor
                ):
        # KL divergence loss between normal Gaussian distribution
        KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

        return KLD.cuda()

    @staticmethod
    def mse_loss(recon_x: torch.tensor,
                 original_x: torch.tensor,
                 reduction: str = 'sum'):
        return F.mse_loss(recon_x, original_x, reduction=reduction)

    @staticmethod
    def bce_loss(recon_x: torch.tensor,
                 original_x: torch.tensor,
                 reduction: str = 'sum'):
        return F.binary_cross_entropy_with_logits(recon_x, original_x, reduction=reduction)

    @staticmethod
    def zinb_reconstruction_loss(X:            torch.tensor,
                                 total_counts: torch.tensor = None,
                                 logits:       torch.tensor = None,
                                 mu:           torch.tensor = None,
                                 theta:        torch.tensor = None,
                                 gate_logits:  torch.tensor = None,
                                 reduction:    str = "sum"):
        if ((total_counts == None) and (logits == None)):
            if ((mu == None) and (theta == None )):
                raise ValueError
            logits = (mu / theta).log()
            total_counts = theta + 1e-6
            znb = ZeroInflatedNegativeBinomial(
                total_count=total_counts,
                logits=logits,
                gate_logits=gate_logits
            )
        else:
            znb = ZeroInflatedNegativeBinomial(
                total_count=total_counts,
                logits=logits,
                gate_logits=gate_logits
            )
        if reduction == "sum":
            reconst_loss = -znb.log_prob(X).sum(dim=1)
        elif reduction == "mean":
            reconst_loss = -znb.log_prob(X).mean(dim=1)
        elif reduction == "none":
            reconst_loss = -znb.log_prob(X)
        return reconst_loss

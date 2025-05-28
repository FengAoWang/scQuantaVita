import numpy as np
import pandas as pd
import anndata
from DVAE_RBM import DVAE_RBM
import torch
from utils import split_data, multiprocessing_train_fold, worker_function, train_fold, GridSearchConfig, \
    single_train_fold, print_celltype_counts
import random
from dataset_param_dict import dataset_params
import os

import sys


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout  # 保留原控制台输出
        self.log = open(filename, "a", encoding='utf-8')  # 追加模式写入文件

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):  # 避免IO错误
        self.terminal.flush()
        self.log.flush()


# 重定向标准输出和错误
sys.stdout = Logger("output.log")
sys.stderr = sys.stdout  # 将错误输出合并到同一文件


def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')


def set_grid_configs():
    grid_configs = []
    for sample_method in ["gibbs"]:
        for norm_method in ["layer"]:
            for latent_dim in [256]:
                for beta in [0.0001, 0, 0.001, 0.1, 1]:
                    config = GridSearchConfig(normalization_method=norm_method,
                                              sample_method=sample_method,
                                              latent_dim=latent_dim,
                                              batch_size=4096,
                                              seed=512,
                                              beta_kl=beta)  # 这里可以更改seed
                    grid_configs.append(config)
    return grid_configs


if __name__ == "__main__":
    # Load single-cell data`
    output_name = "result"
    dataset_list = dataset_params.keys()
    for dataset_name in dataset_list:
        # if dataset_name == 'HLCA_core':

        print(dataset_name)
        gex_data = anndata.read_h5ad(dataset_params[dataset_name]['file_path'])
        print(f"Data shape: {gex_data.X.shape}")
        print(gex_data.obs.columns)
        print_celltype_counts(gex_data, dataset_params[dataset_name]['labels_key'])

        # split single cell data
        # split_data(dataset_name, gex_data)

        grid_configs = set_grid_configs()
        for config in grid_configs:
            # 设置随机种子（确保实验可重复）
            set_seed(config.seed)

            # 构造保存结果的文件夹名称
            config_folder = str(config)
            output_dir = os.path.join(output_name, dataset_name, config_folder)
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nTraining with config: {config_folder}")

            # 分配设备（示例：按fold循环分配）
            device_list = [0, 0, 0, 0, 0]  # 可根据实际GPU情况调整

            # 构造训练参数列表（将config传递给每个fold）
            training_function_args = [
                (DVAE_RBM, gex_data, dataset_name, fold, dataset_params[dataset_name], config, output_name,
                 device_list[fold], 1)
                for fold in range(1)
            ]

            # results = multiprocessing_train_fold(5, worker_function, training_function_args, train_fold)
            # results = single_train_fold(training_function_args,train_fold)
            # print(results)
            results_dict = {}
            for train_args in training_function_args:
                # 设置随机种子（确保每个fold实验可重复）
                fold_id = train_args[3]
                set_seed(config.seed)
                result = train_fold(*train_args)
                # with open(os.path.join(output_dir, f'{dataset_name}_fold{fold_id}.csv'), "w") as f:
                #     json.dump(result, f)
                results_dict[fold_id] = result
            print(results_dict)
            results = [results_dict[i] for i in range(len(training_function_args))]

            results_df = pd.DataFrame(results,
                                      columns=['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_HOM', 'leiden_FMI',
                                               # 'louvain_ARI', 'louvain_AMI', 'louvain_NMI', 'louvain_HOM', 'louvain_FMI',)
                                               # 'Isolated labels', 'KMeans NMI', 'KMeans ARI', 'Silhouette label',
                                               # 'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity',
                                               # 'PCR comparison', 'Batch correction', 'Bio conservation', 'Total'
                                               ])
            print(results_df)
            results_df.to_csv(os.path.join(output_dir, f'{dataset_name}_clustering.csv'), index=False)
            print(f"Results saved to {output_dir}")

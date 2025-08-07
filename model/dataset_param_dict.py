
dataset_params = {

    # 'SCLC_immune': dict(
    #     # dataset
    #     dataset_name="SCLC_immune",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/SCLC_immune_processed.h5ad"
    # ),
    #

    #
    # 'E10.5_E1S1': dict(
    #     # dataset
    #     dataset_name="E10.5_E1S1",
    #     batch_key="",
    #     labels_key="ground_truth",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/stRNA/E10.5_E1S1_processed.h5ad"
    # ),
    #
    # 'E10.5_E1S2': dict(
    #     # dataset
    #     dataset_name="E10.5_E1S2",
    #     batch_key="",
    #     labels_key="ground_truth",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/stRNA/E10.5_E1S2_processed.h5ad"
    # ),

    # 'Puck_200727': dict(
    #     # dataset
    #     dataset_name="Puck_200727",
    #     batch_key="donor_id",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/stRNA/Puck_200727_08_processed.h5ad"
    # ),

    # 'blood_immune': dict(
    #     # dataset
    #     dataset_name="blood_immune",
    #     batch_key="sample",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/blood_immune_processed.h5ad"
    # ),

    # 'prostate': dict(
    #     # dataset
    #     dataset_name="prostate",
    #     batch_key="Sample",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/prostate_processed.h5ad"
    # ),

    # 'COVID_PBMCs': dict(
    #     # dataset
    #     dataset_name="activate_PBMC",
    #     batch_key="sample",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/COVID_PBMCs_processed.h5ad"
    # ),


    # 'activate_PBMC': dict(
    #     # dataset
    #     dataset_name="activate_PBMC",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/Activated_PBMCs_processed.h5ad"
    # ),

    # 'mBDRC': dict(
    #     # dataset
    #     dataset_name="human_cerebral",
    #     batch_key="assay",
    #     labels_key="broad_cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/mBDRC_processed.h5ad"
    # ),
    # 'hcc': dict(
    #     # dataset
    #     dataset_name="human_cerebral",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/human_cerebral_cortex_processed.h5ad"
    # ),
    
    # 'HIV_PBMC': dict(
    #     # dataset
    #     dataset_name="HIC_PBMC",
    #     batch_key="batch_id",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/HIV_PBMC_processed.h5ad"
    # ),


    # 'BMMC': dict(
    #     # dataset
    #     dataset_name="BMMC",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/BMMC_processed.h5ad"
    # ),
    #
    # 'pbmc12k': dict(
    #     # dataset
    #     dataset_name="pbmc12k",
    #     batch_key="batch",
    #     labels_key="str_labels",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/pbmc_12k_processed.h5ad"
    # ),
    'pancreas': dict(
        # dataset
        dataset_name="pancreas",
        batch_key="tech",
        labels_key="celltype",
        file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/pancreas_processed.h5ad"
    ),
    # #
    # 'fetal_lung': dict(
    #     # dataset
    #     dataset_name="fetal_lung",
    #     batch_key="batch",
    #     labels_key="broad_celltype",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/fetal_lung_processed.h5ad"
    # ),
    # 'immune': dict(
    #     dataset_name="immune",
    #     batch_key="batch",
    #     labels_key="final_annotation",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/immune_processed.h5ad"
    # ),
    # 'HTMCB': dict(
    #     dataset_name="HTMCB",
    #     batch_key="sample_id",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/HTMCB_processed.h5ad",
    # ),
    #
    # 'HLCA_core': dict(
    #     dataset_name="HLCA_core",
    #     batch_key="donor_id",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/HLCA_core_processed.h5ad",
    # ),

    # 'BMMC_multiome': dict(
    #     dataset_name="BMMC_multiome",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/project/single_cell_multimodal/data/filter_data/BMMC/RNA_filter.h5ad"
    # ),



    # 'Lung_atlas': dict(
    #     dataset_name="Lung_atlas",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/Lung_atlas_processed.h5ad",
    # ),

    # 'HEOCA': dict(
    #     dataset_name="HEOCA",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/HEOCA_processed.h5ad",
    # ),
    #

    # 'PurifiedPBMC': dict(
    #     dataset_name="PurifiedPBMC",
    #     batch_key="batch",
    #     labels_key="cell_types",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/PurifiedPBMC_processed.h5ad",
    # ),

}


# training_params = {
#     'latent_dim16': dict(
#         latent_dim=16,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
#     'latent_dim32': dict(
#         latent_dim=32,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
#     'latent_dim64': dict(
#         latent_dim=64,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
#     'latent_dim128': dict(
#         latent_dim=128,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
#     'latent_dim256': dict(
#         latent_dim=256,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
#     'latent_dim512': dict(
#         latent_dim=512,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
# }


VAE_training_params = {
    'batch_size2048': dict(
        latent_dim=128,
        lr=1e-3,
        batch_size=128,
        beta_kl=0.01,
        epochs=150,
        normaliztion='batchnorm'
    ),
}

training_params = {
    'batch_size256': dict(
        latent_dim=256,
        lr=1e-2,
        batch_size=256,
        beta_kl=0.0001,
        epochs=100,
        normaliztion='layernorm'
    ),

    # 'batch_size2048': dict(
    #     latent_dim=256,
    #     lr=1e-2,
    #     batch_size=2048,
    #     beta__kl=0.01,
    #     epochs=500,
    #     normaliztion='layernorm'
    # ),
    # 'batch_size1024': dict(
    #     latent_dim=256,
    #     lr=1e-2,
    #     batch_size=1024,
    #     beta__kl=0.01,
    #     epochs=500,
    #     normaliztion='layernorm'
    # ),
    # 'batch_size512': dict(
    #     latent_dim=256,
    #     lr=1e-2,
    #     batch_size=4096,
    #     beta__kl=0.001,
    #     epochs=500,
    #     normaliztion='layernorm'
    # ),
    #
    # 'batch_size256': dict(
    #     latent_dim=128,
    #     lr=1e-2,
    #     batch_size=2048,
    #     beta__kl=0.01,
    #     epochs=500,
    #     normaliztion='layernorm'
    # ),
    #
    # 'batch_size128': dict(
    #     latent_dim=128,
    #     lr=1e-2,
    #     batch_size=1024,
    #     beta__kl=0.01,
    #     epochs=500,
    #     normaliztion='layernorm'
    # ),
    # 'batch_size64': dict(
    #     latent_dim=128,
    #     lr=1e-2,
    #     batch_size=4096,
    #     beta__kl=0.001,
    #     epochs=500,
    #     normaliztion='layernorm'
    # ),
}

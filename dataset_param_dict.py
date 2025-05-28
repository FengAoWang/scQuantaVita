dataset_params = {
    # 'pancreas': dict(
    # # dataset
    # dataset_name="pancreas",
    # batch_key="tech",
    # labels_key="celltype",
    # file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/pancreas_processed.h5ad"),

    # 'BMMC_': dict(
    #     # dataset
    #     dataset_name="BMMC",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/BMMC_RNA_filter.h5ad"
    # ),
    # 'PBMC_': dict(
    #     # dataset
    #     dataset_name="PBMC",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/PBMC_RNA_filter.h5ad"
    # ),

    'pancreas': dict(
       dataset_name="pancreas",
       batch_key="tech",
       labels_key="celltype",
       file_path="/data/amax/sbchen/vae_rbm_cimtest/data/pancreas_processed.h5ad"
   ),

   # 'pbmc12k': dict(
   #    dataset_name="pbmc12k",
   #    batch_key="batch",
   #    labels_key="str_labels",
   #    file_path="/home/xuany/QVAE/data/pbmc_12k_processed.h5ad"
   # ),

   # 'Lung_atlas': dict(
   #     dataset_name="Lung_atlas",
   #     batch_key="batch",
   #     labels_key="cell_type",
   #     file_path="/home/xuany/QVAE/data/Lung_atlas_processed.h5ad",
   # ),

   # 'pancreas_50': dict(
   #    dataset_name="pancreas_50",
   #    batch_key="tech",
   #    labels_key="celltype",
   #    file_path="/home/xuany/QVAE/data/pancreas_processed_50.2.h5ad"
   # ),

   # 'pancreas_20': dict(
   #    dataset_name="pancreas_20",
   #    batch_key="tech",
   #    labels_key="celltype",
   #    file_path="/home/xuany/QVAE/data/pancreas_processed_20.h5ad"
   # ),
   #
   # 'pancreas_30': dict(
   #    dataset_name="pancreas_30",
   #    batch_key="tech",
   #    labels_key="celltype",
   #    file_path="/home/xuany/QVAE/data/pancreas_processed_30.h5ad"
   # ),

    # 'pancreas_20': dict(
    #     dataset_name="pancreas_20",
    #     batch_key="tech",
    #     labels_key="celltype",
    #     file_path="/home/xuany/QVAE/data/pancreas_processed_20.h5ad"
    # ),

    #
    # 'fetal_lung': dict(
    #     # dataset
    #     dataset_name="fetal_lung",
    #     batch_key="batch",
    #     labels_key="broad_celltype",
    #     file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/fetal_lung_processed.h5ad"
    # ),
    #
    'immune': dict(
        dataset_name="immune",
        batch_key="batch",
        labels_key="final_annotation",
        file_path="/data/amax/sbchen/vae_rbm_cimtest/data/immune_processed.h5ad"
    ),
    #
    # 'BMMC*20': dict(
    #     # dataset
    #     dataset_name="BMMC*20",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/BMMC_RNA_filter*20.h5ad"
    # ),
    #
    #
    # 'HLCA_core': dict(
    #     dataset_name="HLCA_core",
    #     batch_key="donor_id",
    #     labels_key="cell_type",
    #     file_path="/home/xuany/QVAE/data/HLCA_core_processed.h5ad",
    # ),
    # #



}

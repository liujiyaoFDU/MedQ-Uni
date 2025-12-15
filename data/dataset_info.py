# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from interleave_datasets import UnifiedEditIterableDataset, MedicalImageEditingIterableDataset_ver1, CounterfactualMedicalIterableDataset_ver1
from t2i_dataset import T2IIterableDataset,T2IIterableDataset_Ver1
from vlm_dataset import SftJSONLIterableDataset,SftJSONLIterableDataset_with_VisualReconstruction_Ver1,SftJSONLIterableDataset_with_VisualReconstruction_Ver2,SftJSONLIterableDataset_Ver1,SftJSONLIterableDataset_TextOnly


# 数据集注册表, 主要用于数据集的加载和注册.
DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    #---------------------------------------------------------------------------------------------------------------
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver1': SftJSONLIterableDataset_with_VisualReconstruction_Ver1,
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver2': SftJSONLIterableDataset_with_VisualReconstruction_Ver2,
    'SftJSONLIterableDataset_Ver1': SftJSONLIterableDataset_Ver1,
    "T2IIterableDataset_Ver1":T2IIterableDataset_Ver1,
    "SftJSONLIterableDataset_TextOnly": SftJSONLIterableDataset_TextOnly,
    "MedicalImageEditingIterableDataset_ver1": MedicalImageEditingIterableDataset_ver1,
    "CounterfactualMedicalIterableDataset_ver1": CounterfactualMedicalIterableDataset_ver1
}

    # 't2i_pretrain': {
    #     't2i': {siim
    #         'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
    #         'num_files': 10, # number of data units to be sharded across all ranks and workers
    #         'num_total_samples': 1000, # number of total samples in the dataset
    #     },
    # },
    # 'unified_edit':{
    #     'seedxedit_multi': {
    #         'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
    #         'num_files': 10,
    #         'num_total_samples': 1000,
    #         "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
	# 	},
    # },
    # 'vlm_sft': {
    #     'llava_ov': {
	# 		'data_dir': 'your_data_path/bagel_example/vlm/images',
	# 		'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
	# 		'num_total_samples': 1000
	# 	},
    # },`

# 数据集信息, 主要用于数据集的加载和注册. 感觉数据加载更好，应该是从我们的yaml 路径填入.这里主要是注册一下特定的数据及和他对应的名字.
DATASET_INFO = {
    #自己定义的loader ==========================================================================================================================================================================================================================================================================================================================================================================================================================================================
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver1': {
        'CheXpertPlus': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/CheXpertPlus.jsonl',
            'num_total_samples': 1
        },
        'cls_100w_sft_data_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/cls_100w_sft_data_internvl.jsonl',
            'num_total_samples': 1
        },
        'CFP_processed_data_QA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/CFP_processed_data_QA.jsonl',
            'num_total_samples': 1
        },        
        'det2d_vqa_500_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/det2d_vqa_500_internvl.jsonl',
            'num_total_samples': 1
        },
        'gmai_vl_shortanswer': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/gmai_vl_shortanswer.jsonl',
            'num_total_samples': 1
        },
        'ImageClef-2019-VQA-Med': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/ImageClef-2019-VQA-Med.jsonl',
            'num_total_samples': 1
        },
        #------------------------通用
        'llava_v1_5_mix665k_correct_v4': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/llava_v1_5_mix665k_correct_v4.jsonl',
            'num_total_samples': 1
        },
        #------------------------通用
        'LLaVA_Med_Caption': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/LLaVA-Med-Caption.jsonl',
            'num_total_samples': 1
        },
        'Medical_Diff_VQA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Medical_Diff_VQA.jsonl',
            'num_total_samples': 1
        },
        'medication_qa_vqa': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/medicat.jsonl',
            'num_total_samples': 1
        },
        'medpix_single': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/medpix_single.jsonl',
            'num_total_samples': 1
        },
        'mimic_cxr': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/mimic_cxr.jsonl',
            'num_total_samples': 1
        },
        'openI_official': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/openI_official.jsonl',
            'num_total_samples': 1
        },
        'Pathology_VQA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Pathology_VQA.jsonl',
            'num_total_samples': 1
        },
        'PMC_CaseReport': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC_CaseReport.jsonl',
            'num_total_samples': 1
        },
        'PMC_VQA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC_VQA.jsonl',
            'num_total_samples': 1
        },
        'PMC_Inline': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC-Inline.jsonl',
            'num_total_samples': 1
        },
        'PMC_Inline_multi_image': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC-Inline_multi-image.jsonl',
            'num_total_samples': 1
        },
        'PMC_OA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC-OA.jsonl',
            'num_total_samples': 1
        },
        'pubmed_cuhksz': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/pubmed_cuhksz.jsonl',
            'num_total_samples': 1
        },
        'pubmedvision': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/pubmedvision.jsonl',
            'num_total_samples': 1
        },
        'quilt_1m': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/quilt_1m.jsonl',
            'num_total_samples': 1
        },
        'RetinaImageBank': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/RetinaImageBank.jsonl",
            'num_total_samples': 1
        },
        'ROCOV2': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path':"/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/ROCOV2.jsonl",
            'num_total_samples': 1
        },
        'seg2d_vqa_500_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/seg2d_vqa_500_internvl.jsonl",
            'num_total_samples': 1
        },
        'sft_93k_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/sft_93k_internvl.jsonl",
            'num_total_samples': 1
        },
        'VQA_RAD': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/VQA_RAD.jsonl",
            'num_total_samples': 1
        },
        'Slake': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Slake.jsonl",
            'num_total_samples': 1
        },
        # stage2:
        'healthgpt_comprehension_gpt':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/healthgpt/healthgpt_comprehension_gpt.jsonl',
            'num_total_samples': 1
        },
        #---------------visual cot -----------------
        'GMAI_Reasoning10K':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/GMAI_Reasoning10K.jsonl",
            'num_total_samples': 1
        },
        # Train/Val split versions (generated by split_train_val_datasets.py)
        'VQA_RAD_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/VQA_RAD_train.jsonl",
            'num_total_samples': 1
        },
        'VQA_RAD_val': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/VQA_RAD_val.jsonl",
            'num_total_samples': 1
        },
        'Slake_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Slake_train.jsonl",
            'num_total_samples': 1
        },
        'Slake_val': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Slake_val.jsonl",
            'num_total_samples': 1
        },
        'GMAI_Reasoning10K_train':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/GMAI_Reasoning10K_train.jsonl",
            'num_total_samples': 1
        },
        'GMAI_Reasoning10K_val':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/GMAI_Reasoning10K_val.jsonl",
            'num_total_samples': 1
        },
        'RadRBench_CXR_chexpert':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/RadRBench_CXR_chexpert.jsonl",
            'num_total_samples': 1
        },
        'RadRBench_CXR_mimic':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/RadRBench_CXR_mimic.jsonl',
            'num_total_samples': 1
        },
        'wanglab_chest_agent_bench':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/wanglab_chest_agent_bench.jsonl',
            'num_total_samples': 1
        }
    },
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver2': {
        'CheXpertPlus': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/CheXpertPlus.jsonl',
            'num_total_samples': 1
        },
        'cls_100w_sft_data_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/cls_100w_sft_data_internvl.jsonl',
            'num_total_samples': 1
        },
        'CFP_processed_data_QA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/CFP_processed_data_QA.jsonl',
            'num_total_samples': 1
        },        
        'det2d_vqa_500_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/det2d_vqa_500_internvl.jsonl',
            'num_total_samples': 1
        },
        'gmai_vl_shortanswer': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/gmai_vl_shortanswer.jsonl',
            'num_total_samples': 1
        },
        'ImageClef-2019-VQA-Med': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/ImageClef-2019-VQA-Med.jsonl',
            'num_total_samples': 1
        },
        #---------------通用
        'llava_v1_5_mix665k_correct_v4': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/llava_v1_5_mix665k_correct_v4.jsonl',
            'num_total_samples': 1
        },
        #---------------通用
        'LLaVA_Med_Caption': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/LLaVA-Med-Caption.jsonl',
            'num_total_samples': 1
        },
        'Medical_Diff_VQA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Medical_Diff_VQA.jsonl',
            'num_total_samples': 1
        },
        'medication_qa_vqa': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/medicat.jsonl',
            'num_total_samples': 1
        },
        'medpix_single': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/medpix_single.jsonl',
            'num_total_samples': 1
        },
        'mimic_cxr': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/mimic_cxr.jsonl',
            'num_total_samples': 1
        },
        'openI_official': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/openI_official.jsonl',
            'num_total_samples': 1
        },
        'Pathology_VQA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Pathology_VQA.jsonl',
            'num_total_samples': 1
        },
        'PMC_CaseReport': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC_CaseReport.jsonl',
            'num_total_samples': 1
        },
        'PMC_VQA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC_VQA.jsonl',
            'num_total_samples': 1
        },
        'PMC_Inline': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC-Inline.jsonl',
            'num_total_samples': 1
        },
        'PMC_Inline_multi_image': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC-Inline_multi-image.jsonl',
            'num_total_samples': 1
        },
        'PMC_OA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC-OA.jsonl',
            'num_total_samples': 1
        },
        'pubmed_cuhksz': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/pubmed_cuhksz.jsonl',
            'num_total_samples': 1
        },
        'pubmedvision': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/pubmedvision.jsonl',
            'num_total_samples': 1
        },
        'quilt_1m': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/quilt_1m.jsonl',
            'num_total_samples': 1
        },
        'RetinaImageBank': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/RetinaImageBank.jsonl",
            'num_total_samples': 1
        },
        'ROCOV2': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path':"/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/ROCOV2.jsonl",
            'num_total_samples': 1
        },
        'seg2d_vqa_500_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/seg2d_vqa_500_internvl.jsonl",
            'num_total_samples': 1
        },
        'sft_93k_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/sft_93k_internvl.jsonl",
            'num_total_samples': 1
        },
        'VQA_RAD': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/VQA_RAD.jsonl",
            'num_total_samples': 1
        },
        'Slake': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Slake.jsonl",
            'num_total_samples': 1
        },
        # stage2:
        'healthgpt_comprehension_gpt':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/healthgpt/healthgpt_comprehension_gpt.jsonl',
            'num_total_samples': 1
        },
        #---------------visual cot -----------------
        'GMAI_Reasoning10K':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/GMAI_Reasoning10K.jsonl",
            'num_total_samples': 1
        },
        # Train/Val split versions (generated by split_train_val_datasets.py)
        'VQA_RAD_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/VQA_RAD_train.jsonl",
            'num_total_samples': 1
        },
        'VQA_RAD_val': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/VQA_RAD_val.jsonl",
            'num_total_samples': 1
        },
        'Slake_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Slake_train.jsonl",
            'num_total_samples': 1
        },
        'Slake_val': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Slake_val.jsonl",
            'num_total_samples': 1
        },
        'GMAI_Reasoning10K_train':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/GMAI_Reasoning10K_train.jsonl",
            'num_total_samples': 1
        },
        'GMAI_Reasoning10K_val':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/GMAI_Reasoning10K_val.jsonl",
            'num_total_samples': 1
        },
        'RadRBench_CXR_chexpert':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/RadRBench_CXR_chexpert.jsonl",
            'num_total_samples': 1
        },
        'RadRBench_CXR_mimic':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/RadRBench_CXR_mimic.jsonl',
            'num_total_samples': 1
        },
        'wanglab_chest_agent_bench':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/wanglab_chest_agent_bench.jsonl',
            'num_total_samples': 1
        }
    }, 
    'SftJSONLIterableDataset_Ver1': {
        'CheXpertPlus': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/CheXpertPlus.jsonl',
            'num_total_samples': 1
        },
        'CFP_processed_data_QA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/CFP_processed_data_QA.jsonl',
            'num_total_samples': 1
        },                
        'cls_100w_sft_data_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/cls_100w_sft_data_internvl.jsonl',
            'num_total_samples': 1
        },
        'det2d_vqa_500_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/det2d_vqa_500_internvl.jsonl',
            'num_total_samples': 1
        },
        'gmai_vl_shortanswer': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/gmai_vl_shortanswer.jsonl',
            'num_total_samples': 1
        },
        'ImageClef-2019-VQA-Med': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/ImageClef-2019-VQA-Med.jsonl',
            'num_total_samples': 1
        },
        #------通用
        'llava_v1_5_mix665k_correct_v4': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/llava_v1_5_mix665k_correct_v4.jsonl',
            'num_total_samples': 1
        },
        #------通用
        'LLaVA_Med_Caption': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/LLaVA-Med-Caption.jsonl',
            'num_total_samples': 1
        },
        'Medical_Diff_VQA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Medical_Diff_VQA.jsonl',
            'num_total_samples': 1
        },
        'medication_qa_vqa': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/medicat.jsonl',
            'num_total_samples': 1
        },
        'medpix_single': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/medpix_single.jsonl',
            'num_total_samples': 1
        },
        'mimic_cxr': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/mimic_cxr.jsonl',
            'num_total_samples': 1
        },
        'openI_official': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/openI_official.jsonl',
            'num_total_samples': 1
        },
        'Pathology_VQA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Pathology_VQA.jsonl',
            'num_total_samples': 1
        },
        'PMC_CaseReport': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC_CaseReport.jsonl',
            'num_total_samples': 1
        },
        'PMC_VQA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC_VQA.jsonl',
            'num_total_samples': 1
        },
        'PMC_Inline': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC-Inline.jsonl',
            'num_total_samples': 1
        },
        'PMC_Inline_multi_image': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC-Inline_multi-image.jsonl',
            'num_total_samples': 1
        },
        'PMC_OA': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/PMC-OA.jsonl',
            'num_total_samples': 1
        },
        'pubmed_cuhksz': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/pubmed_cuhksz.jsonl',
            'num_total_samples': 1
        },
        'pubmedvision': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/pubmedvision.jsonl',
            'num_total_samples': 1
        },
        'quilt_1m': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/quilt_1m.jsonl',
            'num_total_samples': 1
        },
        'RetinaImageBank': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/RetinaImageBank.jsonl",
            'num_total_samples': 1
        },
        'ROCOV2': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path':"/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/ROCOV2.jsonl",
            'num_total_samples': 1
        },
        'seg2d_vqa_500_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/seg2d_vqa_500_internvl.jsonl",
            'num_total_samples': 1
        },
        'sft_93k_internvl': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/sft_93k_internvl.jsonl",
            'num_total_samples': 1
        },
        'VQA_RAD': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/VQA_RAD.jsonl",
            'num_total_samples': 1
        },
        'Slake': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Slake.jsonl",
            'num_total_samples': 1
        },
        # stage2:
        'healthgpt_comprehension_gpt':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/healthgpt/healthgpt_comprehension_gpt.jsonl',
            'num_total_samples': 1
        },
        #---------------visual cot -----------------
        'GMAI_Reasoning10K':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/GMAI_Reasoning10K.jsonl",
            'num_total_samples': 1
        },
        # Train/Val split versions (generated by split_train_val_datasets.py)
        'VQA_RAD_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/VQA_RAD_train.jsonl",
            'num_total_samples': 1
        },
        'VQA_RAD_val': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/VQA_RAD_val.jsonl",
            'num_total_samples': 1
        },
        'Slake_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Slake_train.jsonl",
            'num_total_samples': 1
        },
        'Slake_val': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/Slake_val.jsonl",
            'num_total_samples': 1
        },
        'GMAI_Reasoning10K_train':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/GMAI_Reasoning10K_train.jsonl",
            'num_total_samples': 1
        },
        'GMAI_Reasoning10K_val':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/GMAI_Reasoning10K_val.jsonl",
            'num_total_samples': 1
        },
        'RadRBench_CXR_chexpert':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/RadRBench_CXR_chexpert.jsonl",
            'num_total_samples': 1
        },
        'RadRBench_CXR_mimic':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/RadRBench_CXR_mimic.jsonl',
            'num_total_samples': 1
        },
        'wanglab_chest_agent_bench':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/multi_model_cot/wanglab_chest_agent_bench.jsonl',
            'num_total_samples': 1
        }
    },
    "T2IIterableDataset_Ver1":{
        "ALLaVA_Caption_LAION_4V":{
            "data_dir": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph",
            "jsonl_path": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/ALLaVA-Caption-LAION-4V.jsonl",
            "num_total_samples": 1
        },
        "CFP_processed_data_Generation":{
            "data_dir": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph",
            "jsonl_path": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_generation_ver1_1/new_CFP_processed_data_Generation.jsonl",
            "num_total_samples": 1
        },
        "CFP_processed_data_Generation2":{
            "data_dir": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph",
            "jsonl_path": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_generation_ver1_1/new_gmaivl_caption_from_clsdataset.jsonl",
            "num_total_samples": 1
        },
        "gmaivl_caption_from_clsdataset":{
            "data_dir": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph",
            "jsonl_path": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver2/gmaivl_caption_from_clsdataset.jsonl",
            "num_total_samples": 1
        },
        "pubmedvision_gen":{
            "data_dir": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph",
            "jsonl_path": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_generation_ver1_1/new_pubmedvision_gen.jsonl",
            "num_total_samples": 1
        },
        #stage2:
        'healthgpt_generation_T2I':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/healthgpt/healthgpt_generation_T2I_modified.jsonl',
            'num_total_samples': 1
        },
        # stage3 enhanced image with thinking:
        'new_CFP_processed_data_Generation_modified2_enhanced':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_generation_thinking_ver1_1/new_CFP_processed_data_Generation_modified2_enhanced.jsonl',
            'num_total_samples': 1
        },
        'new_gmaivl_caption_from_clsdataset_modified2_enhanced':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_generation_thinking_ver1_1/new_gmaivl_caption_from_clsdataset_modified2_enhanced.jsonl',
            'num_total_samples': 1
        },
        'new_pubmedvision_gen_modified3':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_generation_thinking_ver1_1/new_pubmedvision_gen_modified3.jsonl',
            'num_total_samples': 1
        },
        # stage 3 enhanced image
        'new_CFP_processed_data_Generation_modified1_enhanced':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_generation_ver1_1/new_CFP_processed_data_Generation_modified1_enhanced.jsonl',
            'num_total_samples': 1
        },
        'new_gmaivl_caption_from_clsdataset_modified1_enhanced':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_generation_ver1_1/new_gmaivl_caption_from_clsdataset_modified1_enhanced.jsonl',
            'num_total_samples': 1
        },
        'new_pubmedvision_gen_modified1_enhanced':{
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_generation_ver1_1/new_pubmedvision_gen_modified1_enhanced.jsonl',
            'num_total_samples': 1
        }
    },
    "SftJSONLIterableDataset_TextOnly":{
        "ChatDoctor-iCliniq-7.3k_converted":{
            # "data_dir": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph",
            "jsonl_path": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/txt/ChatDoctor-iCliniq-7.3k_converted.jsonl",
            "num_total_samples": 1
        },
        "MedQuad-MedicalQnADataset":{
            # "data_dir": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph",
            "jsonl_path": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/txt/MedQuad-MedicalQnADataset.jsonl",
            "num_total_samples": 1
        },
        # -----cot -----
        "bigbio_pubmed_qa_cot":{
            # "data_dir": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph",
            "jsonl_path": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/txt_cot/bigbio_pubmed_qa_cot.jsonl",
            "num_total_samples": 1
        },
        "medical_r1_distill_sft":{
            # "data_dir": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph",
            "jsonl_path": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/txt_cot/medical_r1_distill_sft.jsonl",
            "num_total_samples": 1
        },
        "UCSC_VLAA_MedReason":{
            # "data_dir": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph",
            "jsonl_path": "/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/txt_cot/UCSC_VLAA_MedReason.jsonl",
            "num_total_samples": 1
        }
    },
    "MedicalImageEditingIterableDataset_ver1": {
        # 医学图像编辑数据集示例配置
        'healthgpt_generation_MT': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/healthgpt/healthgpt_generation_MT.jsonl',  # 需要根据实际情况修改
            'num_total_samples': 1
        },
        'healthgpt_generation_SR': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage2_part1_ver1/healthgpt/healthgpt_generation_SR.jsonl',  # 需要根据实际情况修改
            'num_total_samples': 1
        },         
        # 细胞染色转换
        'he2ihc_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/he2ihc_test.jsonl',  # 需要根据实际情况修改
            'num_total_samples': 1
        },
        'he2ihc_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/he2ihc_train.jsonl',  # 需要根据实际情况修改
            'num_total_samples': 1
        },
        # segmentation data:
        'seg_data': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage1_ver1/seg_data.jsonl',  # 需要根据实际情况修改
            'num_total_samples': 1
        },
        # IXI datasets - 脑部MRI超分辨率数据集
        'ixi_t2_sr_4x_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/ixi_t2_sr_4x_train.jsonl',
            'num_total_samples': 1
        },
        'ixi_t2_sr_4x_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/ixi_t2_sr_4x_test.jsonl',
            'num_total_samples': 1
        },
        'ixi_t1_sr_4x_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/ixi_t1_sr_4x_train.jsonl',
            'num_total_samples': 1
        },
        'ixi_t1_sr_4x_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/ixi_t1_sr_4x_test.jsonl',
            'num_total_samples': 1
        },
        'ixi_t2_v2_sr_4x_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/ixi_t2_v2_sr_4x_train_57228.jsonl',
            'num_total_samples': 1
        },
        'ixi_t2_v2_sr_4x_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/ixi_t2_v2_sr_4x_test_500.jsonl',
            'num_total_samples': 1
        },
        # IXI T1 Medical Quality Enhancement - Multi-task
        # Source: annotation_medq-Uni
        'ixi_t1_medq_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation_medq-Uni/images',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation_medq-Uni/annotation/ixi_t1_sr_4x_train.jsonl',
            'num_total_samples': 58377
        },
        'ixi_t1_medq_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation_medq-Uni/images',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation_medq-Uni/annotation/ixi_t1_sr_4x_test.jsonl',
            'num_total_samples': 302
        },
        # ============================================================================
        # Chest X-ray Segmentation Datasets - ISBI Training
        # Base image directory: /inspire/hdd/global_user/hejunjun-24017/junzhin/data
        # ============================================================================
        'cxlseg_complex': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/seg/chest-x-ray-dataset-with-lung-segmentation-1.0.0/processed_cxlseg/annotations/cxlseg_complex_train.jsonl',
            'num_total_samples': 1
        },
        'cxlseg_simple': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/seg/chest-x-ray-dataset-with-lung-segmentation-1.0.0/processed_cxlseg/annotations/cxlseg_simple_train.jsonl',
            'num_total_samples': 1
        },
        'lung_seg_complex': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/seg/Chest X-ray dataset for lung segmentation/processed/complex/lung_seg_train_complex.jsonl',
            'num_total_samples': 1
        },
        'lung_seg_simple': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/seg/Chest X-ray dataset for lung segmentation/processed/simple/lung_seg_train_simple.jsonl',
            'num_total_samples': 1
        },
        'siim_acr_complex': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/seg/SIIM-ACR/processed/jsonl/train.jsonl',
            'num_total_samples': 1
        },
        'siim_acr_simple': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/seg/SIIM-ACR/processed_simple/jsonl/train_simple.jsonl',
            'num_total_samples': 1
        },
        'vindr_ribcxr_complex': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/seg/VinDr_RibCXR/processed/complex/VinDr_RibCXR_train_complex.jsonl',
            'num_total_samples': 1
        },
        'vindr_ribcxr_simple': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/seg/VinDr_RibCXR/processed/simple/VinDr_RibCXR_train_simple.jsonl',
            'num_total_samples': 1
        }
    },
    "CounterfactualMedicalIterableDataset_ver1": {
        # ---------------------------cxr ---------------------------------------
        # Counterfactual医学影像数据集配置
        'counterfactual_cxr_chexpertplus': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/counterfactual_cxr_chexpertplus.jsonl',
            'num_total_samples': 1
        },
        'counterfactual_cxr_chexpertplus_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/counterfactual_cxr_chexpertplus_train.jsonl',
            'num_total_samples': 1
        },
        'counterfactual_cxr_chexpertplus_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/counterfactual_cxr_chexpertplus_test.jsonl',
            'num_total_samples': 1
        },
        'counterfactual_cxr_mimic_cxr': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/counterfactual_cxr_mimic_cxr.jsonl',
            'num_total_samples': 1
        },
        'counterfactual_cxr_mimic_cxr_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/counterfactual_cxr_mimic_cxr_train.jsonl',
            'num_total_samples': 1
        },
        'counterfactual_cxr_mimic_cxr_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/counterfactual_cxr_mimic_cxr_test.jsonl',
            'num_total_samples': 1
        },
        'modality_trans': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/brats23_train_modality_trans_v2.jsonl',
            'num_total_samples': 1
        },
        # SynthRAD 数据集 - 医学图像模态转换任务
        'synthrad_pelvis_mr_to_ct_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/synthrad_pelvis_mr_to_ct_train.jsonl',
            'num_total_samples': 1
        },
        'synthrad_pelvis_mr_to_ct_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/synthrad_pelvis_mr_to_ct_test.jsonl',
            'num_total_samples': 1
        },
        'synthrad_pelvis_ct_to_mr_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/synthrad_pelvis_ct_to_mr_train.jsonl',
            'num_total_samples': 1
        },
        'synthrad_pelvis_ct_to_mr_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/synthrad_pelvis_ct_to_mr_test.jsonl',
            'num_total_samples': 1
        },
        'synthrad_brain_ct_to_mr_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/synthrad_brain_ct_to_mr_train.jsonl',
            'num_total_samples': 1
        },
        'synthrad_brain_ct_to_mr_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/synthrad_brain_ct_to_mr_test.jsonl',
            'num_total_samples': 1
        },
        'synthrad_brain_mr_to_ct_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/synthrad_brain_mr_to_ct_train.jsonl',
            'num_total_samples': 1
        },
        'synthrad_brain_mr_to_ct_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/synthrad_brain_mr_to_ct_test.jsonl',
            'num_total_samples': 1
        },
        'seg_data_sampled': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/seg_data_sampled_100_per_modality.jsonl',
            'num_total_samples': 1146
        }
    }
}   
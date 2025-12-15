# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Unimedvl Team
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Unimedvl Team.
# Modifications: Removed deprecated UnifiedEditIterableDataset and updated dataset registry for medical applications.
#
# For open source usage:
# - Replace 'path_to_your_xxx_data' with your actual data paths
# - CounterfactualMedicalIterableDataset_ver1 retains real paths as reference implementation
# - All other datasets use placeholder paths for security and privacy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from interleave_datasets import UnifiedEditIterableDataset, MedicalImageEditingIterableDataset_ver1, CounterfactualMedicalIterableDataset_ver1
from t2i_dataset import T2IIterableDataset,T2IIterableDataset_Ver1
from vlm_dataset import SftJSONLIterableDataset,SftJSONLIterableDataset_with_VisualReconstruction_Ver1,SftJSONLIterableDataset_with_VisualReconstruction_Ver2,SftJSONLIterableDataset_Ver1,SftJSONLIterableDataset_TextOnly


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver1': SftJSONLIterableDataset_with_VisualReconstruction_Ver1,
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver2': SftJSONLIterableDataset_with_VisualReconstruction_Ver2,
    'SftJSONLIterableDataset_Ver1': SftJSONLIterableDataset_Ver1,
    "T2IIterableDataset_Ver1":T2IIterableDataset_Ver1,
    "SftJSONLIterableDataset_TextOnly": SftJSONLIterableDataset_TextOnly,
    "MedicalImageEditingIterableDataset_ver1": MedicalImageEditingIterableDataset_ver1,
    "CounterfactualMedicalIterableDataset_ver1": CounterfactualMedicalIterableDataset_ver1
}

DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'path_to_your_t2i_data',
            'num_files': 10,
        },
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'path_to_your_vlm_images',
			'jsonl_path': 'path_to_your_vlm_annotations.jsonl',
		},
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'path_to_your_editing_data',
            'num_files': 10,
            "parquet_info_path": 'path_to_your_parquet_info.json',
		},
    },
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver1': {
        'example_dataset_key': {
            'data_dir': 'path_to_your_medical_images',
            'jsonl_path': 'path_to_your_medical_annotations.jsonl',
        },
    },
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver2': {
        'example_dataset_key': {
            'data_dir': 'path_to_your_medical_images',
            'jsonl_path': 'path_to_your_medical_annotations.jsonl',
        },
    }, 
    'SftJSONLIterableDataset_Ver1': {
        'example_dataset_key': {
            'data_dir': 'path_to_your_medical_images',
            'jsonl_path': 'path_to_your_medical_annotations.jsonl',
        },
    },
    "T2IIterableDataset_Ver1":{
        'example_dataset_key': {
            "data_dir": "path_to_your_generation_images",
            "jsonl_path": "path_to_your_generation_annotations.jsonl",
        },
    },
    "SftJSONLIterableDataset_TextOnly":{
        'example_dataset_key': {
            "jsonl_path": "path_to_your_text_only_data.jsonl",
        },
    },
    "MedicalImageEditingIterableDataset_ver1": {
        'example_dataset_key': {
            'data_dir': 'path_to_your_medical_editing_images',
            'jsonl_path': 'path_to_your_medical_editing_annotations.jsonl',
        },
    },
    "CounterfactualMedicalIterableDataset_ver1": {
        'counterfactual_cxr_chexpertplus_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/counterfactual_cxr_chexpertplus_train.jsonl',
        },
        'counterfactual_cxr_chexpertplus_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/counterfactual_cxr_chexpertplus_test.jsonl',
        },
        'counterfactual_cxr_mimic_cxr_train': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/counterfactual_cxr_mimic_cxr_train.jsonl',
        },
        'counterfactual_cxr_mimic_cxr_test': {
            'data_dir': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
            'jsonl_path': '/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation/stage3_ver1/counterfactual_cxr_mimic_cxr_test.jsonl',
        }
    }
}   
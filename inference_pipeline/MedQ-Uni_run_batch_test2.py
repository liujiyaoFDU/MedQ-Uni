"""
批量测试脚本 - 基于 interactive_image_generator.py 改造
支持：
1. PSNR、SSIM 指标计算
2. 命令行参数指定 annotation 文件、结果路径、image root 路径
3. 保存每个推理结果到 jsonl
4. 保存图像到结果路径的 images 文件夹
5. 计算统计指标（按 main_task_type 分类）
6. 显示进度条
7. 断点继续
"""

import sys
import os
import argparse
import json
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

# Environment setup - 环境变量建议在运行脚本中设置
# 如果需要在代码中设置，可以取消下面的注释
# os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
# os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

import warnings
warnings.filterwarnings('ignore')

import torch
import gc
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights, dispatch_model
from safetensors.torch import load_file, save_file

# UniMedVL imports
ROOT = "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni"
sys.path.append(ROOT)

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    """计算 PSNR"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_value / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 SSIM (使用 skimage)"""
    try:
        from skimage.metrics import structural_similarity as ssim
        # 如果是彩色图像，需要指定 channel_axis
        if len(img1.shape) == 3:
            return ssim(img1, img2, channel_axis=2, data_range=255.0)
        else:
            return ssim(img1, img2, data_range=255.0)
    except ImportError:
        print("Warning: skimage not found, SSIM calculation skipped")
        return -1.0


class ImageGenerator:
    """图像生成器类 - 从 interactive_image_generator.py 复制"""
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.vae_model = None
        self.tokenizer = None
        self.vae_transform = None
        self.vit_transform = None
        self.new_token_ids = None
        self.inferencer = None
        self.loaded = False

    def set_seed(self, seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def convert_checkpoint_to_bf16(self, input_path, output_path):
        if not os.path.exists(input_path):
            return False

        state_dict = load_file(input_path, device="cpu")
        first_key = next(iter(state_dict))

        if state_dict[first_key].dtype == torch.bfloat16:
            if input_path != output_path:
                shutil.copy(input_path, output_path)
            del state_dict
            return True

        bf16_state_dict = {key: tensor.to(torch.bfloat16) for key, tensor in state_dict.items()}
        del state_dict
        gc.collect()

        save_file(bf16_state_dict, output_path)
        del bf16_state_dict
        gc.collect()
        return True

    def select_checkpoint_path(self, model_path: str) -> str:
        """Prefer model.safetensors by default; fall back to ema if missing."""
        use_model_checkpoint = self.config.get('use_model_checkpoint', True)
        enable_auto_bf16_conversion = self.config.get('enable_auto_bf16_conversion', True)

        def try_variant(use_model_flag: bool) -> Optional[str]:
            original_checkpoint_name = "model.safetensors" if use_model_flag else "ema.safetensors"
            bf16_checkpoint_name = "model_bf16.safetensors" if use_model_flag else "ema_bf16.safetensors"

            bf16_checkpoint_path = os.path.join(model_path, bf16_checkpoint_name)
            original_checkpoint_path = os.path.join(model_path, original_checkpoint_name)

            if os.path.exists(bf16_checkpoint_path):
                return bf16_checkpoint_path

            if os.path.exists(original_checkpoint_path) and enable_auto_bf16_conversion:
                success = self.convert_checkpoint_to_bf16(original_checkpoint_path, bf16_checkpoint_path)
                return bf16_checkpoint_path if success else original_checkpoint_path

            if os.path.exists(original_checkpoint_path):
                return original_checkpoint_path

            return None

        preferred_path = try_variant(use_model_checkpoint)
        if preferred_path:
            return preferred_path

        fallback_path = try_variant(not use_model_checkpoint)
        if fallback_path:
            fallback_type = "ema" if use_model_checkpoint else "model"
            print(f"Preferred checkpoint not found, falling back to {fallback_type} checkpoint")
            return fallback_path

        raise FileNotFoundError(
            f"No valid checkpoint found in {model_path}. "
            f"Tried model/ema safetensors (bf16 and original)."
        )

    def create_cpu_device_map(self, model):
        cpu_device_map = {}
        for name, _ in model.named_parameters():
            cpu_device_map[name] = "cpu"

        cpu_device_map.update({
            'language_model': 'cpu', 'vit_model': 'cpu', 'time_embedder': 'cpu',
            'latent_pos_embed': 'cpu', 'vae2llm': 'cpu', 'llm2vae': 'cpu',
            'connector': 'cpu', 'vit_pos_embed': 'cpu', 'vae_model': 'cpu'
        })
        return cpu_device_map

    def load_weights_progressively(self, model, vae_model, model_path):
        cpu_device_map = self.create_cpu_device_map(model)

        if not model_path:
            raise ValueError("model_path required")

        final_checkpoint_path = self.select_checkpoint_path(model_path)

        print(f"Loading checkpoint file: {final_checkpoint_path}")
        model = load_checkpoint_and_dispatch(
            model, checkpoint=final_checkpoint_path, device_map=cpu_device_map,
            offload_buffers=False, dtype=torch.bfloat16, force_hooks=False
        )

        torch.cuda.empty_cache()
        gc.collect()

        return model, vae_model

    def deploy_to_gpu_unified(self, model, vae_model, target_device="cuda:0"):
        if not torch.cuda.is_available():
            return model, vae_model

        if torch.cuda.device_count() == 1:
            device_map = infer_auto_device_map(
                model, max_memory={0: self.config.get('max_mem_per_gpu', '40GiB')},
                no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            )

            same_device_modules = [
                'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed',
                'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed'
            ]

            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for module_name in same_device_modules:
                if module_name in device_map:
                    device_map[module_name] = first_device
        else:
            device_map = {name: target_device for name, _ in model.named_parameters()}
            first_device = target_device

        model = dispatch_model(model, device_map=device_map)
        vae_model = vae_model.to(device=first_device, dtype=torch.bfloat16)

        return model, vae_model

    def load_model(self):
        if self.loaded:
            print("Model already loaded")
            return

        print("Loading Bagel model (may take 5-10 minutes)...")
        self.set_seed(self.config.get('seed', 42))

        model_path = self.config.get('model_path')
        if not model_path:
            raise ValueError("model_path required")

        print(f"Checkpoint path: {model_path}")

        # Load configs
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # Load VAE
        ae_path = os.path.join(model_path, "ae.safetensors")
        vae_path = os.path.join(model_path, "vae_model.safetensors")

        if os.path.exists(ae_path):
            print(f"Loading VAE from autoencoder: {ae_path}")
            vae_model, vae_config = load_ae(local_path=ae_path)
        elif os.path.exists(vae_path):
            print(f"Loading VAE directly from: {vae_path}")
            vae_model, vae_config = load_ae(local_path=None)
            vae_state_dict = load_file(vae_path, device="cpu")
            vae_state_dict = {key.replace("module.", ""): value for key, value in vae_state_dict.items()}
            missing, unexpected = vae_model.load_state_dict(vae_state_dict, strict=False, assign=True)
            if missing:
                print(f"Warning: Missing keys in VAE model: {missing}")
            if unexpected:
                print(f"Warning: Unexpected keys in VAE model: {unexpected}")
        else:
            raise FileNotFoundError(f"Neither {ae_path} nor {vae_path} found in {model_path}")

        vae_model = vae_model.cpu().to(torch.bfloat16)

        # Bagel config
        config = BagelConfig(
            visual_gen=True, visual_und=True,
            llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
            vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
            latent_patch_size=2, max_latent_size=64,
        )

        # Initialize empty model
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config, vae_model=vae_model)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids

        # Image transforms
        vae_size = self.config.get('vae_transform_size', (1024, 32, 16))
        vit_size = self.config.get('vit_transform_size', (980, 387, 14))
        self.vae_transform = ImageTransform(vae_size[0], vae_size[1], vae_size[2])
        self.vit_transform = ImageTransform(vit_size[0], vit_size[1], vit_size[2])

        # Load weights
        if self.config.get('enable_cpu_loading', True):
            model, vae_model = self.load_weights_progressively(model, vae_model, model_path)
            torch.cuda.empty_cache()
            gc.collect()

        # Deploy to GPU
        target_device = f"cuda:{self.config.get('target_gpu_device', '0')}" if torch.cuda.is_available() else "cpu"
        model, vae_model = self.deploy_to_gpu_unified(model, vae_model, target_device)

        model = model.eval()
        self.model = model
        self.vae_model = vae_model

        # Create inferencer
        self.inferencer = InterleaveInferencer(
            model=self.model, vae_model=self.vae_model, tokenizer=self.tokenizer,
            vae_transform=self.vae_transform, vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids
        )

        self.loaded = True
        print("Model loaded successfully")


class BatchTester:
    """批量测试类"""
    def __init__(self, args):
        self.args = args
        self.generator = None
        self.results = []
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        # 结果文件路径
        self.results_jsonl_path = self.output_dir / "results.jsonl"
        self.stats_json_path = self.output_dir / "statistics.json"
        
        # 加载已有结果（用于断点继续）
        self.processed_samples = set()
        if self.results_jsonl_path.exists():
            print(f"Found existing results, loading for resume...")
            with open(self.results_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line)
                    self.processed_samples.add(result['sample_id'])
                    self.results.append(result)
            print(f"Loaded {len(self.processed_samples)} existing results")
    
    def load_annotation(self) -> List[Dict]:
        """加载 annotation 文件"""
        annotations = []
        with open(self.args.annotation_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line)
                sample['sample_id'] = idx  # 添加唯一 ID
                annotations.append(sample)
                
                # 如果指定了样本数量限制，则只加载前N个样本
                if self.args.num_samples > 0 and len(annotations) >= self.args.num_samples:
                    break
        
        return annotations
    
    def get_image_path(self, relative_path: str) -> str:
        """根据相对路径获取完整图像路径"""
        return os.path.join(self.args.image_root, relative_path)
    
    def extract_instruction(self, message: List[Dict]) -> str:
        """从 message 中提取指令"""
        for msg in message:
            if msg['from'] == 'human':
                # 移除 <image> 标记
                return msg['value'].replace('<image>', '').strip()
        return ""

    def infer_understanding_text(self, image: Image.Image, instruction: str) -> str:
        """按训练预处理方式推理理解文本（避免 VAE resize 预处理影响 ViT 输入）。"""
        inferencer = self.generator.inferencer
        gen_context = inferencer.init_gen_context()

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            gen_context = inferencer.update_context_image(
                pil_img2rgb(image),
                gen_context,
                vae=False,
                vit=True,
            )
            gen_context = inferencer.update_context_text(instruction, gen_context)
            understanding_text = inferencer.gen_text(
                gen_context,
                max_length=800,
                do_sample=self.args.text_do_sample,
                temperature=self.args.text_temperature,
            )

        return understanding_text

    def infer_edit_image(self, image: Image.Image, instruction: str, image_shapes: tuple) -> Image.Image:
        """按训练预处理方式推理生成编辑结果（VAE/VIT 都基于原图各自 resize）。"""
        inferencer = self.generator.inferencer
        gen_context = inferencer.init_gen_context()
        cfg_img_context = deepcopy(gen_context)
        cfg_text_context = deepcopy(gen_context)

        input_lists = [image, instruction]

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = inferencer.update_context_text(input_term, gen_context)
                    cfg_img_context = inferencer.update_context_text(input_term, cfg_img_context)
                elif isinstance(input_term, Image.Image):
                    gen_context = inferencer.update_context_image(
                        pil_img2rgb(input_term),
                        gen_context,
                        vae=True,
                        vit=True,
                    )
                    cfg_text_context = deepcopy(gen_context)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            edited_image = inferencer.gen_image(
                image_shapes,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=self.args.cfg_text_scale,
                cfg_img_scale=self.args.cfg_img_scale,
                cfg_interval=[0.0, 1.0],
                cfg_renorm_min=0.0,
                cfg_renorm_type="text_channel",
                timestep_shift=self.args.timestep_shift,
                num_timesteps=self.args.num_timesteps,
            )

        return edited_image
    
    def process_single_sample(self, sample: Dict) -> Optional[Dict]:
        """处理单个样本"""
        sample_id = sample['sample_id']
        
        # 检查是否已处理（断点继续）
        if sample_id in self.processed_samples:
            return None
        
        try:
            # 获取输入和真值图像路径
            input_relative_path = sample['input_img'][0]['path']
            output_relative_path = sample['output_img'][0]['path']
            
            input_path = self.get_image_path(input_relative_path)
            gt_path = self.get_image_path(output_relative_path)
            
            if not os.path.exists(input_path):
                print(f"Warning: Input image not found: {input_path}")
                return None
            
            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth image not found: {gt_path}")
                return None
            
            # 加载图像
            input_image = Image.open(input_path)
            gt_image = Image.open(gt_path)
            
            # 转换为 RGB
            if input_image.mode != 'RGB':
                if input_image.mode == 'RGBA':
                    background = Image.new('RGB', input_image.size, (255, 255, 255))
                    background.paste(input_image, mask=input_image.split()[-1])
                    input_image = background
                else:
                    input_image = input_image.convert('RGB')
            
            if gt_image.mode != 'RGB':
                if gt_image.mode == 'RGBA':
                    background = Image.new('RGB', gt_image.size, (255, 255, 255))
                    background.paste(gt_image, mask=gt_image.split()[-1])
                    gt_image = background
                else:
                    gt_image = gt_image.convert('RGB')

            # 检查并修正图像尺寸 - 确保不小于最小边要求
            # 从配置中获取最小边要求
            vae_min_size = self.generator.config.get('vae_transform_size', (2048, 128, 16))[1]  # 128
            vit_min_size = self.generator.config.get('vit_transform_size', (518, 224, 14))[1]    # 224
            min_required_size = max(vae_min_size, vit_min_size)  # 取较大值作为最小边要求

            original_width, original_height = input_image.size
            min_current_size = min(original_width, original_height)

            if min_current_size < min_required_size:
                # 图像的最小边小于要求，需要resize
                scale_factor = min_required_size / min_current_size
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)

                print(f"Warning [Sample {sample_id}]: Input image size ({original_width}x{original_height}) "
                      f"has a dimension smaller than minimum required size ({min_required_size}). "
                      f"Resizing to ({new_width}x{new_height}) to meet minimum requirements.")

                input_image = input_image.resize((new_width, new_height), Image.BICUBIC)

                # 同样处理GT图像
                gt_original_width, gt_original_height = gt_image.size
                gt_min_current_size = min(gt_original_width, gt_original_height)

                if gt_min_current_size < min_required_size:
                    gt_scale_factor = min_required_size / gt_min_current_size
                    gt_new_width = int(gt_original_width * gt_scale_factor)
                    gt_new_height = int(gt_original_height * gt_scale_factor)

                    print(f"Warning [Sample {sample_id}]: GT image size ({gt_original_width}x{gt_original_height}) "
                          f"has a dimension smaller than minimum required size ({min_required_size}). "
                          f"Resizing to ({gt_new_width}x{gt_new_height}) to meet minimum requirements.")

                    gt_image = gt_image.resize((gt_new_width, gt_new_height), Image.BICUBIC)

            # 提取指令（移除 <image> 标记）
            instruction = self.extract_instruction(sample['message'])
            
            # 设置随机种子
            if self.args.seed > 0:
                self.generator.set_seed(self.args.seed)
            
            # 记录开始时间
            start_time = time.time()
            
            # 推理 - Understanding mode（图像预处理与训练一致：ViT 基于原图 resize）
            understanding_text = self.infer_understanding_text(input_image, instruction)
            
            # 构建最终指令（与 interactive_image_generator.py 保持一致）
            final_instruction = f"{instruction}\n\n{understanding_text}"

            # 图像编辑 - 使用实际transform后的尺寸确保100%对齐
            actual_size = self.generator.inferencer.get_actual_transformed_size(input_image)

            edited_image = self.infer_edit_image(input_image, final_instruction, actual_size)
            
            if edited_image.mode != 'RGB':
                edited_image = edited_image.convert('RGB')
            
            # 记录结束时间
            inference_time = time.time() - start_time
            
            # 保存生成的图像 - 使用层级路径结构
            # 例如：input_img 是 "IXI-T1/process/high_res/IXI322-IOP-0891-T1/021_noise.png"
            # 则保存为 "images/IXI-T1/process/high_res/IXI322-IOP-0891-T1/021_noise_input.png"等
            input_path_parts = Path(input_relative_path)
            output_path_parts = Path(output_relative_path)
            
            # 获取基础文件名（完整的文件名，不含扩展名）
            base_filename = input_path_parts.stem
            
            # 构建保存路径（使用 input_img 的目录结构）
            save_dir = self.images_dir / input_path_parts.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存三张图像到同一目录，统一命名规范
            # 1. 输入图像（使用 _input 后缀）
            input_filename = f"{base_filename}_input.png"
            input_save_path = save_dir / input_filename
            input_image.save(input_save_path)
            input_relative_save_path = input_save_path.relative_to(self.images_dir)
            
            # 2. 真值图像（使用 _GT 后缀）
            gt_filename = f"{base_filename}_GT.png"
            gt_save_path = save_dir / gt_filename
            gt_image.save(gt_save_path)
            gt_relative_save_path = gt_save_path.relative_to(self.images_dir)
            
            # 3. 预测图像（使用 _pred 后缀）
            pred_filename = f"{base_filename}_pred.png"
            pred_save_path = save_dir / pred_filename
            edited_image.save(pred_save_path)
            pred_relative_save_path = pred_save_path.relative_to(self.images_dir)
            
            # 转换为 numpy 数组进行指标计算
            edited_np = np.array(edited_image)
            gt_np = np.array(gt_image)

            # 确保尺寸一致 - 使用actual_size确保精确对齐
            if edited_np.shape != gt_np.shape:
                # 使用BICUBIC插值（与transform一致）调整GT图像尺寸
                # actual_size格式: (height, width), PIL.Image.resize需要: (width, height)
                gt_image_resized = gt_image.resize(
                    (actual_size[1], actual_size[0]),
                    Image.BICUBIC
                )
                gt_np = np.array(gt_image_resized)
            
            # 计算 PSNR 和 SSIM
            psnr = calculate_psnr(edited_np, gt_np)
            ssim = calculate_ssim(edited_np, gt_np)
            
            # 构建结果字典 - 包含完整的 input_img、output_img 和 pred_img 信息
            result = {
                'sample_id': sample_id,
                'main_task_type': sample['main_task_type'],
                'degrade_type': sample.get('degrade_type', ''),
                # 原始路径信息
                'input_img': sample['input_img'],
                'output_img': sample['output_img'],
                # 保存的图像路径（在 images/ 文件夹下）
                'saved_input_img': [{
                    'path': str(input_relative_save_path),
                    'height': input_image.height,
                    'width': input_image.width
                }],
                'saved_output_img': [{
                    'path': str(gt_relative_save_path),
                    'height': gt_image.height,
                    'width': gt_image.width
                }],
                'pred_img': [{
                    'path': str(pred_relative_save_path),
                    'height': edited_image.height,
                    'width': edited_image.width
                }],
                # 保存完整的 message
                'message': sample['message'],
                # 性能指标
                'psnr': float(psnr) if not np.isinf(psnr) else 999.99,
                'ssim': float(ssim),
                'inference_time': inference_time,
                # 理解文本
                'understanding_text': understanding_text,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self):
        """运行批量测试"""
        # 初始化模型配置
        config = {
            "model_path": self.args.model_path,
            "target_gpu_device": self.args.target_gpu_device,
            "max_mem_per_gpu": self.args.max_mem_per_gpu,
            "enable_cpu_loading": True,
            'use_model_checkpoint': self.args.use_model_checkpoint,
            "enable_auto_bf16_conversion": True,
            "offload_folder": "/tmp/bagel_offload",
            "seed": self.args.seed,
            # Align with `configs/train_ixi_t1_medq_ver1.yaml`
            # VAE: max=1024, min=512, stride=16
            # ViT: max=518,  min=224, stride=14
            "vae_transform_size": (2048, 128, 16),
            "vit_transform_size": (518, 224, 14),
            "text_do_sample": self.args.text_do_sample,
            "text_temperature": self.args.text_temperature,
        }
        
        # 加载模型
        print("Initializing model...")
        self.generator = ImageGenerator(config)
        self.generator.load_model()
        
        # 加载 annotation
        print(f"Loading annotation from: {self.args.annotation_file}")
        annotations = self.load_annotation()
        
        # 显示样本数量信息
        if self.args.num_samples > 0:
            print(f"Sample limit: {self.args.num_samples} (partial testing mode)")
        else:
            print(f"Sample limit: ALL (full testing mode)")
        
        print(f"Total samples loaded: {len(annotations)}")
        print(f"Already processed: {len(self.processed_samples)}")
        print(f"Remaining: {len(annotations) - len(self.processed_samples)}")
        
        # 打开结果文件（追加模式）
        results_file = open(self.results_jsonl_path, 'a', encoding='utf-8')
        
        # 处理每个样本
        with tqdm(total=len(annotations), initial=len(self.processed_samples), desc="Processing") as pbar:
            for sample in annotations:
                if sample['sample_id'] in self.processed_samples:
                    pbar.update(1)
                    continue
                
                result = self.process_single_sample(sample)
                
                if result is not None:
                    # 保存单个结果到 jsonl
                    results_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    results_file.flush()
                    
                    # 添加到内存结果列表
                    self.results.append(result)
                    self.processed_samples.add(sample['sample_id'])
                
                pbar.update(1)
        
        results_file.close()
        
        # 计算统计指标
        self.calculate_statistics()
        
        print(f"\nTesting completed!")
        print(f"Results saved to: {self.results_jsonl_path}")
        print(f"Statistics saved to: {self.stats_json_path}")
        print(f"Images saved to: {self.images_dir}")
    
    def calculate_statistics(self):
        """计算并保存统计指标"""
        if not self.results:
            print("No results to calculate statistics")
            return
        
        # 总体统计
        psnr_values = [r['psnr'] for r in self.results if r['psnr'] < 999.0]
        ssim_values = [r['ssim'] for r in self.results if r['ssim'] >= 0]
        time_values = [r['inference_time'] for r in self.results]
        
        overall_stats = {
            'total_samples': len(self.results),
            'psnr_mean': float(np.mean(psnr_values)) if psnr_values else 0.0,
            'psnr_std': float(np.std(psnr_values)) if psnr_values else 0.0,
            'ssim_mean': float(np.mean(ssim_values)) if ssim_values else 0.0,
            'ssim_std': float(np.std(ssim_values)) if ssim_values else 0.0,
            'avg_inference_time': float(np.mean(time_values)) if time_values else 0.0,
        }
        
        # 按 main_task_type 分类统计
        task_type_stats = {}
        task_types = set(r['main_task_type'] for r in self.results)
        
        for task_type in task_types:
            task_results = [r for r in self.results if r['main_task_type'] == task_type]
            task_psnr = [r['psnr'] for r in task_results if r['psnr'] < 999.0]
            task_ssim = [r['ssim'] for r in task_results if r['ssim'] >= 0]
            
            task_type_stats[task_type] = {
                'count': len(task_results),
                'psnr_mean': float(np.mean(task_psnr)) if task_psnr else 0.0,
                'psnr_std': float(np.std(task_psnr)) if task_psnr else 0.0,
                'ssim_mean': float(np.mean(task_ssim)) if task_ssim else 0.0,
                'ssim_std': float(np.std(task_ssim)) if task_ssim else 0.0,
            }
        
        # 组合统计结果
        statistics = {
            'overall': overall_stats,
            'by_task_type': task_type_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到 JSON 文件
        with open(self.stats_json_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        # 打印统计结果
        print("\n" + "="*60)
        print("OVERALL STATISTICS")
        print("="*60)
        print(f"Total samples: {overall_stats['total_samples']}")
        print(f"PSNR: {overall_stats['psnr_mean']:.2f} ± {overall_stats['psnr_std']:.2f}")
        print(f"SSIM: {overall_stats['ssim_mean']:.4f} ± {overall_stats['ssim_std']:.4f}")
        print(f"Avg inference time: {overall_stats['avg_inference_time']:.2f}s")
        
        print("\n" + "="*60)
        print("STATISTICS BY TASK TYPE")
        print("="*60)
        for task_type, stats in task_type_stats.items():
            print(f"\n{task_type} (n={stats['count']}):")
            print(f"  PSNR: {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f}")
            print(f"  SSIM: {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch testing script for UniMedVL")
    
    # 输入输出路径
    parser.add_argument('--annotation_file', type=str, 
                       default='annotation/ixi_t1_sr_4x_test.jsonl',
                       help='Path to annotation JSONL file')
    parser.add_argument('--image_root', type=str,
                       default='/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/images/medical_data_from_s_server_ceph',
                       help='Root directory for images')
    parser.add_argument('--output_dir', type=str,
                       default='test_results',
                       help='Output directory for results')
    
    # 模型路径
    parser.add_argument('--model_path', type=str,
                       default='/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/MedQ-Uni/output/ixi_t1_medq_1ep_ver1/0000600',
                       help='Path to model checkpoint')

    # Checkpoint selection
    parser.add_argument('--use_model_checkpoint', dest='use_model_checkpoint', action='store_true',
                       help='Use model.safetensors as the base checkpoint (default).')
    parser.add_argument('--use_ema_checkpoint', dest='use_model_checkpoint', action='store_false',
                       help='Use ema.safetensors instead of model.safetensors.')
    parser.set_defaults(use_model_checkpoint=True)
    
    # GPU 设置
    parser.add_argument('--target_gpu_device', type=str, default='0',
                       help='Target GPU device')
    parser.add_argument('--max_mem_per_gpu', type=str, default='40GiB',
                       help='Maximum memory per GPU')
    
    # 生成参数
    parser.add_argument('--cfg_text_scale', type=float, default=4.0,
                       help='CFG scale for text')
    parser.add_argument('--cfg_img_scale', type=float, default=2.0,
                       help='CFG scale for image')
    parser.add_argument('--num_timesteps', type=int, default=50,
                       help='Number of diffusion timesteps')
    parser.add_argument('--timestep_shift', type=float, default=1.0,
                       help='Timestep shift for diffusion')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # 文本生成参数
    parser.add_argument('--text_do_sample', action='store_true',
                       help='Use sampling for text generation (default: greedy)')
    parser.add_argument('--text_temperature', type=float, default=0.3,
                       help='Temperature for text sampling')
    
    # 样本数量控制
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='Number of samples to test (default: -1 for all samples)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print("="*60)
    print("Batch Testing Script for UniMedVL")
    print("="*60)
    print(f"Annotation file: {args.annotation_file}")
    print(f"Image root: {args.image_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model path: {args.model_path}")
    print("="*60)
    
    tester = BatchTester(args)
    tester.run()

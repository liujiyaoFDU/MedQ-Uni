"""
参数管理模块 - 精确控制MOT架构的参数冻结和优化
提供参数分组、冻结、统计和验证功能
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def freeze_mot_branch(model: nn.Module, branch: str) -> int:
    """
    冻结MOT模型的特定分支（Understanding或Generation专家）

    Args:
        model: BAGEL模型实例
        branch: 'und' (Understanding专家) 或 'gen' (Generation专家)

    Returns:
        冻结的参数数量
    """
    if not hasattr(model, 'language_model'):
        logger.warning("模型没有language_model属性，跳过MOT分支冻结")
        return 0

    frozen_count = 0

    for name, param in model.language_model.named_parameters():
        should_freeze = False

        if branch == 'gen':
            # 冻结Generation专家：参数名包含'moe_gen'
            if 'moe_gen' in name:
                should_freeze = True
        elif branch == 'und':
            # 冻结Understanding专家：参数名不包含'moe_gen'（但属于language_model）
            # 排除embedding和lm_head等共享组件
            if 'moe_gen' not in name:
                # 只冻结Transformer层中的参数
                if any(key in name for key in ['layers.', 'self_attn.', 'mlp.']):
                    should_freeze = True

        if should_freeze and param.requires_grad:
            param.requires_grad = False
            frozen_count += param.numel()

    logger.info(f"冻结MOT {branch.upper()}分支: {frozen_count:,} 参数")
    return frozen_count


def classify_parameters(model: nn.Module) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
    """
    将模型参数分类到不同的组件

    Returns:
        字典，键为组件名，值为(参数名, 参数对象)的列表
    """
    param_groups = {
        'vae': [],
        'vit': [],
        'mot_understanding': [],
        'mot_generation': [],
        'llm_embeddings': [],
        'llm_head': [],
        'llm_other': [],
    }

    for name, param in model.named_parameters():
        if name.startswith('vae_model.'):
            param_groups['vae'].append((name, param))
        elif name.startswith('vit_model.'):
            param_groups['vit'].append((name, param))
        elif name.startswith('language_model.'):
            # 区分MOT的两个分支
            if 'moe_gen' in name:
                param_groups['mot_generation'].append((name, param))
            elif any(key in name for key in ['layers.', 'self_attn.', 'mlp.']):
                param_groups['mot_understanding'].append((name, param))
            elif 'embed_tokens' in name:
                param_groups['llm_embeddings'].append((name, param))
            elif 'lm_head' in name:
                param_groups['llm_head'].append((name, param))
            else:
                param_groups['llm_other'].append((name, param))
        else:
            param_groups['llm_other'].append((name, param))

    return param_groups


def create_param_groups_v2(model: nn.Module, training_args) -> List[Dict]:
    """
    创建优化器参数组（改进版）- 只添加requires_grad=True的参数

    Args:
        model: FSDP包装后的模型
        training_args: 训练参数配置

    Returns:
        参数组列表，每个组包含 {'params', 'lr', 'name'}
    """
    param_groups = []
    assigned_params_set = set()

    # 首先对所有参数进行分类
    classified = classify_parameters(model)

    # 1. VAE参数组（如果不冻结）
    if hasattr(model, "vae_model") and model.vae_model is not None:
        vae_trainable = [p for n, p in classified['vae'] if p.requires_grad]
        if vae_trainable:
            vae_lr = getattr(training_args, 'vae_lr', training_args.lr)
            param_groups.append({
                'params': vae_trainable,
                'lr': vae_lr,
                'name': 'vae_model'
            })
            assigned_params_set.update(id(p) for p in vae_trainable)
            logger.info(f"VAE参数组: {len(vae_trainable)} 参数, lr={vae_lr}")

    # 2. ViT参数组（如果不冻结）
    if hasattr(model, "vit_model") and model.vit_model is not None:
        vit_trainable = [p for n, p in classified['vit'] if p.requires_grad]
        if vit_trainable:
            vit_lr = getattr(training_args, 'vit_lr', training_args.lr)
            param_groups.append({
                'params': vit_trainable,
                'lr': vit_lr,
                'name': 'vit_model'
            })
            assigned_params_set.update(id(p) for p in vit_trainable)
            logger.info(f"ViT参数组: {len(vit_trainable)} 参数, lr={vit_lr}")

    # 3. MOT Generation专家参数组（如果不冻结）
    mot_gen_trainable = [p for n, p in classified['mot_generation'] if p.requires_grad]
    if mot_gen_trainable:
        mot_gen_lr = getattr(training_args, 'mot_gen_lr',
                             getattr(training_args, 'llm_lr', training_args.lr))
        param_groups.append({
            'params': mot_gen_trainable,
            'lr': mot_gen_lr,
            'name': 'mot_generation'
        })
        assigned_params_set.update(id(p) for p in mot_gen_trainable)
        logger.info(f"MOT Generation参数组: {len(mot_gen_trainable)} 参数, lr={mot_gen_lr}")

    # 4. MOT Understanding专家参数组（如果不冻结）
    mot_und_trainable = [p for n, p in classified['mot_understanding'] if p.requires_grad]
    if mot_und_trainable:
        mot_und_lr = getattr(training_args, 'mot_und_lr',
                            getattr(training_args, 'llm_lr', training_args.lr))
        param_groups.append({
            'params': mot_und_trainable,
            'lr': mot_und_lr,
            'name': 'mot_understanding'
        })
        assigned_params_set.update(id(p) for p in mot_und_trainable)
        logger.info(f"MOT Understanding参数组: {len(mot_und_trainable)} 参数, lr={mot_und_lr}")

    # 5. LLM其他参数（embeddings, lm_head等）
    llm_other_params = []
    for category in ['llm_embeddings', 'llm_head', 'llm_other']:
        for n, p in classified[category]:
            if p.requires_grad and id(p) not in assigned_params_set:
                llm_other_params.append(p)

    if llm_other_params:
        llm_lr = getattr(training_args, 'llm_lr', training_args.lr)
        param_groups.append({
            'params': llm_other_params,
            'lr': llm_lr,
            'name': 'llm_other'
        })
        logger.info(f"LLM其他参数组: {len(llm_other_params)} 参数, lr={llm_lr}")

    if not param_groups:
        raise ValueError("没有可训练的参数！请检查freeze配置。")

    return param_groups


def count_parameters(params: List[nn.Parameter]) -> int:
    """计算参数总数"""
    return sum(p.numel() for p in params)


def format_param_count(count: int) -> str:
    """格式化参数数量显示"""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.1f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    else:
        return str(count)


def print_param_statistics(model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                          rank: int = 0):
    """
    打印详细的参数统计信息

    Args:
        model: 模型实例
        optimizer: 优化器实例（可选）
        rank: 当前进程的rank（只在rank 0打印）
    """
    if rank != 0:
        return

    classified = classify_parameters(model)

    # 统计各组件的参数
    stats = {}
    for component, params_list in classified.items():
        total = sum(p.numel() for n, p in params_list)
        trainable = sum(p.numel() for n, p in params_list if p.requires_grad)
        stats[component] = {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }

    # 统计优化器中的参数
    optimizer_params = {}
    if optimizer is not None:
        for group in optimizer.param_groups:
            group_name = group.get('name', 'unnamed')
            optimizer_params[group_name] = {
                'count': len(group['params']),
                'total': sum(p.numel() for p in group['params']),
                'lr': group['lr']
            }

    # 打印表格
    print("\n" + "=" * 80)
    print("参数统计详情".center(80))
    print("=" * 80)

    # 组件级统计
    print(f"\n{'组件':<20} {'总参数':<15} {'可训练参数':<15} {'冻结参数':<15}")
    print("-" * 80)

    component_names = {
        'vae': 'VAE',
        'vit': 'ViT',
        'mot_understanding': 'MOT-Understanding',
        'mot_generation': 'MOT-Generation',
        'llm_embeddings': 'LLM-Embeddings',
        'llm_head': 'LLM-Head',
        'llm_other': 'LLM-Other'
    }

    total_all = 0
    total_trainable = 0
    total_frozen = 0

    for component, display_name in component_names.items():
        stat = stats[component]
        if stat['total'] > 0:  # 只显示非空组件
            print(f"{display_name:<20} {format_param_count(stat['total']):<15} "
                  f"{format_param_count(stat['trainable']):<15} "
                  f"{format_param_count(stat['frozen']):<15}")
            total_all += stat['total']
            total_trainable += stat['trainable']
            total_frozen += stat['frozen']

    print("-" * 80)
    print(f"{'总计':<20} {format_param_count(total_all):<15} "
          f"{format_param_count(total_trainable):<15} "
          f"{format_param_count(total_frozen):<15}")

    # 优化器参数组统计
    if optimizer_params:
        print("\n" + "-" * 80)
        print("优化器参数组详情:")
        print("-" * 80)

        total_in_optimizer = 0
        for i, (group_name, info) in enumerate(optimizer_params.items()):
            print(f"  [{i}] {group_name}: {format_param_count(info['total'])} 参数, "
                  f"lr={info['lr']:.2e}")
            total_in_optimizer += info['total']

        print("-" * 80)
        print(f"优化器管理的总参数: {format_param_count(total_in_optimizer)}")

        # 验证：优化器中的参数应该等于可训练参数
        if total_in_optimizer != total_trainable:
            print(f"\n⚠️  警告: 优化器参数({format_param_count(total_in_optimizer)}) "
                  f"!= 可训练参数({format_param_count(total_trainable)})")
        else:
            print(f"\n✓ 验证通过: 优化器只管理可训练参数")

    print("=" * 80 + "\n")


def validate_param_freezing(model: nn.Module, initial_params: Dict[str, torch.Tensor],
                            rank: int = 0) -> bool:
    """
    验证冻结的参数是否真的没有变化

    Args:
        model: 当前模型
        initial_params: 初始参数的副本 {name: tensor}
        rank: 当前进程rank

    Returns:
        True如果所有冻结参数未变化，否则False
    """
    if rank != 0:
        return True

    all_valid = True
    changed_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:  # 冻结的参数
            if name in initial_params:
                initial = initial_params[name]
                if not torch.equal(param.data, initial):
                    changed_params.append(name)
                    all_valid = False

    if not all_valid:
        logger.warning(f"检测到 {len(changed_params)} 个冻结参数发生了变化:")
        for name in changed_params[:5]:  # 只显示前5个
            logger.warning(f"  - {name}")
        if len(changed_params) > 5:
            logger.warning(f"  ... 还有 {len(changed_params) - 5} 个")

    return all_valid


def save_initial_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    保存初始参数的副本（用于后续验证）

    Args:
        model: 模型实例

    Returns:
        参数名到参数值的字典
    """
    initial_params = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:  # 只保存冻结的参数
            initial_params[name] = param.data.clone()

    return initial_params


def print_memory_stats(rank: int = 0, prefix: str = ""):
    """
    打印当前GPU显存统计

    Args:
        rank: 当前进程rank
        prefix: 打印前缀
    """
    if rank != 0:
        return

    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n{'=' * 60}")
    print(f"GPU显存统计 - {prefix}".center(60))
    print(f"{'=' * 60}")
    print(f"  当前分配: {allocated:.2f} GB")
    print(f"  当前保留: {reserved:.2f} GB")
    print(f"  峰值分配: {max_allocated:.2f} GB")
    print(f"{'=' * 60}\n")


def get_optimizer_param_names(optimizer: torch.optim.Optimizer) -> Set[int]:
    """
    获取优化器中所有参数的ID集合

    Args:
        optimizer: 优化器实例

    Returns:
        参数ID的集合
    """
    param_ids = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            param_ids.add(id(param))
    return param_ids


def verify_param_groups_integrity(model: nn.Module, optimizer: torch.optim.Optimizer,
                                 rank: int = 0) -> bool:
    """
    验证参数组的完整性：
    1. 优化器中的所有参数都应该是requires_grad=True
    2. 所有requires_grad=True的参数都应该在优化器中

    Args:
        model: 模型实例
        optimizer: 优化器实例
        rank: 当前进程rank

    Returns:
        True如果验证通过，否则False
    """
    if rank != 0:
        return True

    optimizer_param_ids = get_optimizer_param_names(optimizer)

    # 检查1: 优化器中是否有frozen参数
    frozen_in_optimizer = []
    for name, param in model.named_parameters():
        if not param.requires_grad and id(param) in optimizer_param_ids:
            frozen_in_optimizer.append(name)

    # 检查2: 是否有trainable参数不在优化器中
    trainable_not_in_optimizer = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in optimizer_param_ids:
            trainable_not_in_optimizer.append(name)

    all_valid = True

    if frozen_in_optimizer:
        logger.error(f"发现 {len(frozen_in_optimizer)} 个冻结参数在优化器中:")
        for name in frozen_in_optimizer[:5]:
            logger.error(f"  - {name}")
        all_valid = False

    if trainable_not_in_optimizer:
        logger.error(f"发现 {len(trainable_not_in_optimizer)} 个可训练参数不在优化器中:")
        for name in trainable_not_in_optimizer[:5]:
            logger.error(f"  - {name}")
        all_valid = False

    if all_valid:
        logger.info("✓ 参数组完整性验证通过")

    return all_valid

#!/usr/bin/env python3
"""
JSONL数据集验证脚本
用于验证医学图像数据集JSONL文件的完整性、正确性，并生成详细统计报告
"""

import os
import sys
import json
import argparse
import csv
import copy
import time
import glob
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from PIL import Image, ImageFile
except ImportError:
    print("错误：需要安装 Pillow 库。请运行：pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("错误：需要安装 tqdm 库。请运行：pip install tqdm")
    sys.exit(1)

# PIL 图像处理配置（参考 data/data_utils.py）
Image.MAX_IMAGE_PIXELS = 20000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================================
# 默认配置
# ============================================================================
DEFAULT_JSONL_DIR = "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1"
DEFAULT_IMAGE_BASE_PATH = "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/images"
DEFAULT_OUTPUT_DIR = "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/validation_reports"


# ============================================================================
# 验证函数
# ============================================================================

def validate_required_fields(entry: Dict[str, Any], line_num: int) -> List[str]:
    """
    验证必需字段是否存在且格式正确

    Args:
        entry: JSONL数据条目
        line_num: 行号

    Returns:
        错误列表，如果没有错误则返回空列表
    """
    errors = []

    # 检查 main_task_type
    if "main_task_type" not in entry:
        errors.append("缺失字段：main_task_type")
    elif not isinstance(entry["main_task_type"], str) or not entry["main_task_type"].strip():
        errors.append("main_task_type 必须是非空字符串")

    # 检查 input_img
    if "input_img" not in entry:
        errors.append("缺失字段：input_img")
    elif not isinstance(entry["input_img"], list) or len(entry["input_img"]) == 0:
        errors.append("input_img 必须是非空列表")
    else:
        for idx, img_info in enumerate(entry["input_img"]):
            if not isinstance(img_info, dict):
                errors.append(f"input_img[{idx}] 必须是字典类型")
                continue
            if "path" not in img_info:
                errors.append(f"input_img[{idx}] 缺失 path 字段")
            if "height" not in img_info:
                errors.append(f"input_img[{idx}] 缺失 height 字段")
            if "width" not in img_info:
                errors.append(f"input_img[{idx}] 缺失 width 字段")

    # 检查 output_img
    if "output_img" not in entry:
        errors.append("缺失字段：output_img")
    elif not isinstance(entry["output_img"], list) or len(entry["output_img"]) == 0:
        errors.append("output_img 必须是非空列表")
    else:
        for idx, img_info in enumerate(entry["output_img"]):
            if not isinstance(img_info, dict):
                errors.append(f"output_img[{idx}] 必须是字典类型")
                continue
            if "path" not in img_info:
                errors.append(f"output_img[{idx}] 缺失 path 字段")
            if "height" not in img_info:
                errors.append(f"output_img[{idx}] 缺失 height 字段")
            if "width" not in img_info:
                errors.append(f"output_img[{idx}] 缺失 width 字段")

    # 检查 message
    if "message" not in entry:
        errors.append("缺失字段：message")
    elif not isinstance(entry["message"], list) or len(entry["message"]) < 2:
        errors.append("message 必须是包含至少2个元素的列表")

    return errors


def validate_image_path(
    img_info: Dict[str, Any],
    image_base_path: str,
    img_type: str
) -> Tuple[bool, str, Tuple[int, int]]:
    """
    验证图像文件是否存在、可读，并返回实际分辨率

    Args:
        img_info: 图像信息字典（包含path, height, width）
        image_base_path: 图像文件基础路径
        img_type: 图像类型标识（用于错误消息）

    Returns:
        (是否通过验证, 错误消息, 实际分辨率(width, height))
    """
    if "path" not in img_info:
        return False, f"{img_type}: 缺失 path 字段", (0, 0)

    # 拼接绝对路径
    rel_path = img_info["path"]
    abs_path = os.path.join(image_base_path, rel_path)

    # 检查文件是否存在
    if not os.path.exists(abs_path):
        return False, f"{img_type}: 文件不存在 - {rel_path}", (0, 0)

    # 尝试打开图像并读取分辨率
    try:
        with Image.open(abs_path) as img:
            actual_width, actual_height = img.size

            # 验证分辨率是否匹配
            expected_width = img_info.get("width")
            expected_height = img_info.get("height")

            if expected_width is not None and expected_height is not None:
                if actual_width != expected_width or actual_height != expected_height:
                    return False, (
                        f"{img_type}: 分辨率不匹配 - "
                        f"记录: {expected_width}x{expected_height}, "
                        f"实际: {actual_width}x{actual_height} - {rel_path}"
                    ), (actual_width, actual_height)

            return True, "", (actual_width, actual_height)

    except Exception as e:
        return False, f"{img_type}: 无法打开图像 - {rel_path} (错误: {str(e)})", (0, 0)


def validate_message_format(entry: Dict[str, Any]) -> List[str]:
    """
    验证 message 字段格式是否正确

    Args:
        entry: JSONL数据条目

    Returns:
        错误列表，如果没有错误则返回空列表
    """
    errors = []

    if "message" not in entry:
        return ["缺失字段：message"]

    messages = entry["message"]

    if not isinstance(messages, list):
        return ["message 必须是列表类型"]

    if len(messages) < 2:
        return ["message 必须至少包含2个元素（human + gpt）"]

    has_human = False
    has_gpt = False

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"message[{idx}] 必须是字典类型")
            continue

        # 检查必需字段
        if "from" not in msg:
            errors.append(f"message[{idx}] 缺失 from 字段")
        else:
            from_value = msg["from"]
            if from_value not in ["human", "gpt"]:
                errors.append(f"message[{idx}] from 字段必须是 'human' 或 'gpt'，当前为: {from_value}")
            if from_value == "human":
                has_human = True
            elif from_value == "gpt":
                has_gpt = True

        if "value" not in msg:
            errors.append(f"message[{idx}] 缺失 value 字段")
        elif not isinstance(msg["value"], str) or not msg["value"].strip():
            errors.append(f"message[{idx}] value 必须是非空字符串")

    # 确保至少有一个 human 和一个 gpt 消息
    if not has_human:
        errors.append("message 中缺少 human 消息")
    if not has_gpt:
        errors.append("message 中缺少 gpt 消息")

    return errors


def validate_single_entry(
    entry: Dict[str, Any],
    line_num: int,
    image_base_path: str,
    verbose: bool = False
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    验证单个数据条目

    Args:
        entry: JSONL数据条目
        line_num: 行号
        image_base_path: 图像文件基础路径
        verbose: 是否详细输出

    Returns:
        (是否通过验证, 错误列表, 元数据字典)
    """
    all_errors = []
    metadata = {
        "main_task_type": entry.get("main_task_type", "unknown"),
        "degrade_type": entry.get("degrade_type", "N/A"),
        "input_resolutions": [],
        "output_resolutions": []
    }

    # 1. 验证必需字段
    field_errors = validate_required_fields(entry, line_num)
    all_errors.extend(field_errors)

    # 2. 验证消息格式
    message_errors = validate_message_format(entry)
    all_errors.extend(message_errors)

    # 3. 验证图像（如果字段存在）
    if "input_img" in entry and isinstance(entry["input_img"], list):
        for idx, img_info in enumerate(entry["input_img"]):
            if isinstance(img_info, dict):
                is_valid, error_msg, resolution = validate_image_path(
                    img_info, image_base_path, f"input_img[{idx}]"
                )
                if not is_valid:
                    all_errors.append(error_msg)
                else:
                    metadata["input_resolutions"].append(resolution)

    if "output_img" in entry and isinstance(entry["output_img"], list):
        for idx, img_info in enumerate(entry["output_img"]):
            if isinstance(img_info, dict):
                is_valid, error_msg, resolution = validate_image_path(
                    img_info, image_base_path, f"output_img[{idx}]"
                )
                if not is_valid:
                    all_errors.append(error_msg)
                else:
                    metadata["output_resolutions"].append(resolution)

    passed = len(all_errors) == 0

    if verbose and not passed:
        print(f"  [行 {line_num}] 发现 {len(all_errors)} 个问题")

    return passed, all_errors, metadata


# ============================================================================
# 统计和报告生成
# ============================================================================

class ValidationStatistics:
    """验证统计信息收集器"""

    def __init__(self):
        self.total_files = 0
        self.total_entries = 0
        self.passed_entries = 0
        self.failed_entries = 0

        # 按任务类型统计（使用普通字典，避免序列化问题）
        self.task_type_stats = {}
        self.degrade_type_stats = {}

        # 分辨率统计
        self.input_widths = []
        self.input_heights = []
        self.output_widths = []
        self.output_heights = []

        # 问题详情
        self.error_details = []

        # 问题类型计数
        self.error_type_counts = {}

    def add_entry(
        self,
        jsonl_file: str,
        line_num: int,
        passed: bool,
        errors: List[str],
        metadata: Dict[str, Any]
    ):
        """添加一个条目的验证结果"""
        self.total_entries += 1

        task_type = metadata.get("main_task_type", "unknown")
        degrade_type = metadata.get("degrade_type", "N/A")

        # 更新任务类型统计
        if task_type not in self.task_type_stats:
            self.task_type_stats[task_type] = {"total": 0, "passed": 0}
        if degrade_type not in self.degrade_type_stats:
            self.degrade_type_stats[degrade_type] = {"total": 0, "passed": 0}

        self.task_type_stats[task_type]["total"] += 1
        self.degrade_type_stats[degrade_type]["total"] += 1

        if passed:
            self.passed_entries += 1
            self.task_type_stats[task_type]["passed"] += 1
            self.degrade_type_stats[degrade_type]["passed"] += 1

            # 收集分辨率信息
            for w, h in metadata.get("input_resolutions", []):
                self.input_widths.append(w)
                self.input_heights.append(h)
            for w, h in metadata.get("output_resolutions", []):
                self.output_widths.append(w)
                self.output_heights.append(h)
        else:
            self.failed_entries += 1

            # 记录错误详情
            self.error_details.append({
                "jsonl_file": os.path.basename(jsonl_file),
                "line_num": line_num,
                "task_type": task_type,
                "degrade_type": degrade_type,
                "errors": errors
            })

            # 分类错误类型
            for error in errors:
                if "缺失字段" in error or "缺少" in error:
                    self.error_type_counts["缺失字段"] = self.error_type_counts.get("缺失字段", 0) + 1
                elif "文件不存在" in error:
                    self.error_type_counts["图像文件不存在"] = self.error_type_counts.get("图像文件不存在", 0) + 1
                elif "无法打开图像" in error:
                    self.error_type_counts["图像不可读"] = self.error_type_counts.get("图像不可读", 0) + 1
                elif "分辨率不匹配" in error:
                    self.error_type_counts["分辨率不匹配"] = self.error_type_counts.get("分辨率不匹配", 0) + 1
                elif "message" in error.lower():
                    self.error_type_counts["Message格式错误"] = self.error_type_counts.get("Message格式错误", 0) + 1
                else:
                    self.error_type_counts["其他错误"] = self.error_type_counts.get("其他错误", 0) + 1

    def merge(self, other: 'ValidationStatistics'):
        """合并另一个统计对象"""
        self.total_files += other.total_files
        self.total_entries += other.total_entries
        self.passed_entries += other.passed_entries
        self.failed_entries += other.failed_entries

        # 合并任务类型统计
        for task_type, stats in other.task_type_stats.items():
            if task_type not in self.task_type_stats:
                self.task_type_stats[task_type] = {"total": 0, "passed": 0}
            self.task_type_stats[task_type]["total"] += stats["total"]
            self.task_type_stats[task_type]["passed"] += stats["passed"]

        for degrade_type, stats in other.degrade_type_stats.items():
            if degrade_type not in self.degrade_type_stats:
                self.degrade_type_stats[degrade_type] = {"total": 0, "passed": 0}
            self.degrade_type_stats[degrade_type]["total"] += stats["total"]
            self.degrade_type_stats[degrade_type]["passed"] += stats["passed"]

        # 合并分辨率列表
        self.input_widths.extend(other.input_widths)
        self.input_heights.extend(other.input_heights)
        self.output_widths.extend(other.output_widths)
        self.output_heights.extend(other.output_heights)

        # 合并错误详情
        self.error_details.extend(other.error_details)

        # 合并错误类型计数
        for error_type, count in other.error_type_counts.items():
            self.error_type_counts[error_type] = self.error_type_counts.get(error_type, 0) + count

    def _calc_stats(self, values: List[int]) -> Dict[str, float]:
        """计算统计量"""
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0, "std": 0.0}

        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std = variance ** 0.5

        return {
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "std": std
        }

    def generate_report(self, max_errors: int = None) -> str:
        """生成文本格式的验证报告"""
        report = []
        report.append("=" * 70)
        report.append("JSONL 数据集验证报告")
        report.append("=" * 70)
        report.append("")

        # 总体统计
        report.append("[总体统计]")
        report.append(f"- 扫描的JSONL文件：{self.total_files} 个")
        report.append(f"- 总数据条目：{self.total_entries:,} 条")
        report.append(f"- 通过验证：{self.passed_entries:,} 条 ({self.passed_entries/max(self.total_entries, 1)*100:.2f}%)")
        report.append(f"- 验证失败：{self.failed_entries:,} 条 ({self.failed_entries/max(self.total_entries, 1)*100:.2f}%)")
        report.append("")

        # 按任务类型统计
        if self.task_type_stats:
            report.append("[按任务类型统计]")
            report.append("main_task_type:")
            for task_type in sorted(self.task_type_stats.keys()):
                stats = self.task_type_stats[task_type]
                pass_rate = stats["passed"] / max(stats["total"], 1) * 100
                report.append(f"  - {task_type}: {stats['total']:,} 条 (通过率: {pass_rate:.2f}%)")
            report.append("")

        if len(self.degrade_type_stats) > 1 or "N/A" not in self.degrade_type_stats:
            report.append("degrade_type:")
            for degrade_type in sorted(self.degrade_type_stats.keys()):
                if degrade_type == "N/A":
                    continue
                stats = self.degrade_type_stats[degrade_type]
                pass_rate = stats["passed"] / max(stats["total"], 1) * 100
                report.append(f"  - {degrade_type}: {stats['total']:,} 条 (通过率: {pass_rate:.2f}%)")
            report.append("")

        # 分辨率统计
        if self.input_widths:
            report.append("[图像分辨率统计]")

            input_w_stats = self._calc_stats(self.input_widths)
            input_h_stats = self._calc_stats(self.input_heights)
            report.append("Input Images:")
            report.append(f"  - 最小: {input_w_stats['min']}x{input_h_stats['min']}")
            report.append(f"  - 最大: {input_w_stats['max']}x{input_h_stats['max']}")
            report.append(f"  - 平均: {input_w_stats['mean']:.0f}x{input_h_stats['mean']:.0f}")
            report.append(f"  - 标准差: {input_w_stats['std']:.0f}x{input_h_stats['std']:.0f}")
            report.append("")

            if self.output_widths:
                output_w_stats = self._calc_stats(self.output_widths)
                output_h_stats = self._calc_stats(self.output_heights)
                report.append("Output Images:")
                report.append(f"  - 最小: {output_w_stats['min']}x{output_h_stats['min']}")
                report.append(f"  - 最大: {output_w_stats['max']}x{output_h_stats['max']}")
                report.append(f"  - 平均: {output_w_stats['mean']:.0f}x{output_h_stats['mean']:.0f}")
                report.append(f"  - 标准差: {output_w_stats['std']:.0f}x{output_h_stats['std']:.0f}")
                report.append("")

        # 错误类型统计
        if self.error_type_counts:
            report.append("[错误类型统计]")
            for error_type in sorted(self.error_type_counts.keys()):
                count = self.error_type_counts[error_type]
                report.append(f"  - {error_type}: {count:,} 次")
            report.append("")

        # 问题详情
        if self.error_details:
            num_to_show = len(self.error_details) if max_errors is None else min(max_errors, len(self.error_details))
            report.append(f"[问题详情] (显示 {num_to_show}/{len(self.error_details)} 条)")
            report.append("")

            for detail in self.error_details[:num_to_show]:
                report.append(f"文件: {detail['jsonl_file']}, 行号: {detail['line_num']}")
                report.append(f"  - 任务类型: {detail['task_type']}")
                if detail['degrade_type'] != "N/A":
                    report.append(f"  - 退化类型: {detail['degrade_type']}")
                report.append("  - 问题:")
                for error in detail['errors']:
                    report.append(f"    * {error}")
                report.append("")

            if len(self.error_details) > num_to_show:
                report.append(f"... 还有 {len(self.error_details) - num_to_show} 条问题未显示")
                report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def save_csv_report(self, output_path: str):
        """保存CSV格式的详细错误报告"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "JSONL文件名", "行号", "任务类型", "退化类型", "错误数量", "错误详情"
            ])

            if self.error_details:
                for detail in self.error_details:
                    errors_str = " | ".join(detail['errors'])
                    writer.writerow([
                        detail['jsonl_file'],
                        detail['line_num'],
                        detail['task_type'],
                        detail['degrade_type'],
                        len(detail['errors']),
                        errors_str
                    ])

        if self.error_details:
            print(f"CSV报告已保存到: {output_path}")
        else:
            print(f"CSV报告已保存到: {output_path} (无错误条目)")


class AutoSaver:
    """自动保存管理器（支持时间和数量双触发机制）"""

    def __init__(
        self,
        output_dir: str,
        save_interval: int = 300,
        save_every_n: int = 10000,
        keep_n: int = 3
    ):
        """
        初始化自动保存管理器

        Args:
            output_dir: 报告输出目录
            save_interval: 自动保存时间间隔（秒），默认300秒（5分钟）
            save_every_n: 每处理N条自动保存，默认10000条
            keep_n: 保留最新N个进度文件，默认3个
        """
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.save_every_n = save_every_n
        self.keep_n = keep_n

        self.last_save_time = time.time()
        self.last_save_count = 0

    def should_save(self, current_count: int) -> bool:
        """
        判断是否应该保存（时间或数量触发）

        Args:
            current_count: 当前已处理的条目数

        Returns:
            是否应该保存
        """
        now = time.time()
        time_trigger = (now - self.last_save_time) >= self.save_interval
        count_trigger = (current_count - self.last_save_count) >= self.save_every_n

        if time_trigger or count_trigger:
            self.last_save_time = now
            self.last_save_count = current_count
            return True
        return False

    def save_progress(
        self,
        stats: ValidationStatistics,
        max_errors: int = None
    ) -> Tuple[str, str]:
        """
        保存中间进度报告

        Args:
            stats: 统计信息对象
            max_errors: 最多显示多少条错误

        Returns:
            (TXT报告路径, CSV报告路径)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        count = stats.total_entries

        prefix = f"progress_entries_{count:08d}"

        # 保存TXT报告
        txt_path = os.path.join(
            self.output_dir,
            f"validation_report_{prefix}_{timestamp}.txt"
        )
        report = stats.generate_report(max_errors=max_errors)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # 保存CSV报告
        csv_path = os.path.join(
            self.output_dir,
            f"validation_report_{prefix}_{timestamp}.csv"
        )
        stats.save_csv_report(csv_path)

        # 清理旧文件
        self.cleanup_old_progress_files()

        return txt_path, csv_path

    def save_final(
        self,
        stats: ValidationStatistics,
        max_errors: int = None
    ) -> Tuple[str, str]:
        """
        保存最终报告

        Args:
            stats: 统计信息
            max_errors: 最多显示多少条错误

        Returns:
            (TXT报告路径, CSV报告路径)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存文本报告
        txt_path = os.path.join(self.output_dir, f"validation_report_final_{timestamp}.txt")
        report = stats.generate_report(max_errors=max_errors)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # 保存CSV报告
        csv_path = os.path.join(self.output_dir, f"validation_report_final_{timestamp}.csv")
        stats.save_csv_report(csv_path)

        return txt_path, csv_path

    def cleanup_old_progress_files(self):
        """清理旧的进度文件，保留最新的N个"""
        try:
            # 查找所有progress文件
            txt_pattern = os.path.join(self.output_dir, "validation_report_progress_*.txt")
            csv_pattern = os.path.join(self.output_dir, "validation_report_progress_*.csv")

            txt_files = glob.glob(txt_pattern)
            csv_files = glob.glob(csv_pattern)

            # 分别处理TXT和CSV文件
            for files_list in [txt_files, csv_files]:
                if len(files_list) > self.keep_n:
                    # 按修改时间排序（最新的在前）
                    files_list.sort(key=os.path.getmtime, reverse=True)
                    # 删除旧文件
                    for old_file in files_list[self.keep_n:]:
                        try:
                            os.remove(old_file)
                        except OSError:
                            pass
        except Exception:
            pass


class GracefulKiller:
    """优雅退出处理器（处理Ctrl+C等中断信号）"""

    def __init__(self, stats: ValidationStatistics, auto_saver: AutoSaver = None):
        """
        初始化优雅退出处理器

        Args:
            stats: 统计信息对象
            auto_saver: 自动保存管理器（可选）
        """
        self.stats = stats
        self.auto_saver = auto_saver
        self.kill_now = False

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        """处理退出信号"""
        if self.kill_now:
            sys.exit(1)

        self.kill_now = True
        print("\n\n检测到中断信号，正在保存当前进度...")

        if self.auto_saver:
            try:
                txt_path, csv_path = self.auto_saver.save_progress(self.stats)
                print(f"进度已保存到: {txt_path}")
                print(f"CSV已保存到: {csv_path}")
            except Exception as e:
                print(f"保存失败: {e}")

        print("退出程序")
        sys.exit(0)


# ============================================================================
# 多线程工作函数
# ============================================================================

def validate_file_worker(
    jsonl_file_path: str,
    image_base_path: str,
    verbose: bool = False
) -> ValidationStatistics:
    """
    工作进程函数：验证单个JSONL文件的所有条目

    Args:
        jsonl_file_path: JSONL文件路径（字符串，便于序列化）
        image_base_path: 图像文件基础路径
        verbose: 是否详细输出

    Returns:
        ValidationStatistics 对象，包含该文件的所有验证统计信息
    """
    # 创建本地统计对象
    stats = ValidationStatistics()
    stats.total_files = 1

    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)

                # 验证条目
                passed, errors, metadata = validate_single_entry(
                    entry, line_num, image_base_path, verbose
                )

                # 直接添加到本地统计对象
                stats.add_entry(jsonl_file_path, line_num, passed, errors, metadata)

            except json.JSONDecodeError as e:
                error_msg = f"JSON解析错误: {str(e)}"
                stats.add_entry(
                    jsonl_file_path, line_num, False,
                    [error_msg],
                    {"main_task_type": "unknown", "degrade_type": "N/A"}
                )

    except Exception as e:
        # 如果整个文件读取失败，返回空统计对象
        pass

    return stats


# ============================================================================
# 主流程
# ============================================================================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="JSONL数据集验证工具 - 验证医学图像数据集的完整性和正确性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认云端路径（自动保存TXT和CSV报告）
  python validate_jsonl_dataset.py

  # 自定义路径
  python validate_jsonl_dataset.py --jsonl_dir /path/to/jsonl --image_base_path /path/to/images

  # 禁用进度条并限制错误显示（适合重定向到日志）
  python validate_jsonl_dataset.py --max_errors 50 --no-progress

  # 详细模式（自动禁用进度条）
  python validate_jsonl_dataset.py --verbose

注意: 脚本会自动在output_dir中保存TXT和CSV两种格式的报告
        """
    )

    parser.add_argument(
        '--jsonl_dir',
        type=str,
        default=DEFAULT_JSONL_DIR,
        help=f'JSONL文件所在目录 (默认: {DEFAULT_JSONL_DIR})'
    )

    parser.add_argument(
        '--image_base_path',
        type=str,
        default=DEFAULT_IMAGE_BASE_PATH,
        help=f'图像文件的基础路径 (默认: {DEFAULT_IMAGE_BASE_PATH})'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'验证报告输出目录（自动保存TXT和CSV报告） (默认: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--max_errors',
        type=int,
        default=None,
        help='终端最多显示多少条错误详情 (默认: 显示全部)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出模式（显示每个条目的验证状态）'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='禁用进度条（适用于日志记录场景）'
    )

    # 多线程相关参数
    parser.add_argument(
        '--max_workers',
        type=int,
        default=16,
        help='并发线程数 (默认: 4，设为1禁用多线程)'
    )

    # 自动保存相关参数
    parser.add_argument(
        '--enable_autosave',
        action='store_true',
        default=True,
        help='启用自动保存中间结果 (默认: 启用)'
    )

    parser.add_argument(
        '--no_autosave',
        dest='enable_autosave',
        action='store_false',
        help='禁用自动保存'
    )

    parser.add_argument(
        '--save_interval',
        type=int,
        default=300,
        help='自动保存时间间隔（秒） (默认: 300秒=5分钟)'
    )

    parser.add_argument(
        '--save_every_n',
        type=int,
        default=10000,
        help='每处理N条自动保存 (默认: 10000条)'
    )

    parser.add_argument(
        '--keep_progress_files',
        type=int,
        default=3,
        help='保留最新N个进度文件 (默认: 3个)'
    )

    return parser.parse_args()


def main():
    """主函数（多线程版本）"""
    args = parse_arguments()

    # 检查JSONL目录是否存在
    if not os.path.exists(args.jsonl_dir):
        print(f"错误：JSONL目录不存在: {args.jsonl_dir}")
        sys.exit(1)

    # 检查图像基础路径是否存在
    if not os.path.exists(args.image_base_path):
        print(f"警告：图像基础路径不存在: {args.image_base_path}")
        print("将继续验证，但所有图像文件检查都会失败")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输出目录: {args.output_dir}")

    # 查找所有JSONL文件
    jsonl_files = list(Path(args.jsonl_dir).glob("*.jsonl"))

    if not jsonl_files:
        print(f"错误：在 {args.jsonl_dir} 中未找到任何 .jsonl 文件")
        sys.exit(1)

    print(f"找到 {len(jsonl_files)} 个JSONL文件")
    print(f"图像基础路径: {args.image_base_path}")
    print(f"并发线程数: {args.max_workers}")
    if args.enable_autosave:
        print(f"自动保存: 每{args.save_interval}秒 或 每{args.save_every_n}条")
    else:
        print("自动保存: 禁用")
    print("开始验证...\n")

    # 初始化统计收集器（线程安全版本）
    stats = ValidationStatistics()
    stats.total_files = len(jsonl_files)

    # 初始化自动保存管理器
    auto_saver = None
    if args.enable_autosave:
        auto_saver = AutoSaver(
            output_dir=args.output_dir,
            save_interval=args.save_interval,
            save_every_n=args.save_every_n,
            keep_n=args.keep_progress_files
        )

    # 初始化优雅退出处理器
    killer = GracefulKiller(stats, auto_saver)

    # 配置进度条
    use_progress = not args.no_progress and not args.verbose

    # 使用多进程处理
    if args.max_workers > 1:
        # 多进程模式
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有文件任务（注意：传递字符串路径而不是Path对象）
            future_to_file = {
                executor.submit(
                    validate_file_worker,
                    str(jsonl_file),
                    args.image_base_path,
                    args.verbose
                ): jsonl_file
                for jsonl_file in sorted(jsonl_files)
            }

            # 使用进度条跟踪完成的文件
            if use_progress:
                pbar = tqdm(total=len(jsonl_files), desc="处理JSONL文件", unit="file")

            # 处理完成的任务
            for future in as_completed(future_to_file):
                jsonl_file = future_to_file[future]

                try:
                    # 获取验证结果（ValidationStatistics 对象）
                    file_stats = future.result()

                    # 合并到主统计对象
                    stats.merge(file_stats)

                    # 更新进度条
                    if use_progress:
                        pbar.update(1)
                        pbar.set_postfix({"已处理条目": stats.total_entries})

                    # 检查是否需要自动保存
                    if auto_saver and auto_saver.should_save(stats.total_entries):
                        txt_path, csv_path = auto_saver.save_progress(
                            stats, args.max_errors
                        )
                        msg = f"已保存中间结果: {stats.total_entries} 条目 -> {os.path.basename(txt_path)}"
                        if use_progress:
                            tqdm.write(msg)
                        else:
                            print(msg)

                except Exception as e:
                    error_msg = f"处理文件 {jsonl_file.name} 时出错: {e}"
                    if use_progress:
                        tqdm.write(error_msg)
                    else:
                        print(error_msg)

            if use_progress:
                pbar.close()

    else:
        # 单进程模式（向后兼容）
        jsonl_files_iter = sorted(jsonl_files)
        if use_progress:
            jsonl_files_iter = tqdm(
                jsonl_files_iter,
                desc="处理JSONL文件",
                unit="file"
            )

        for jsonl_file in jsonl_files_iter:
            if args.verbose:
                print(f"处理: {jsonl_file.name}")

            try:
                # 调用worker函数获取统计对象
                file_stats = validate_file_worker(
                    str(jsonl_file),
                    args.image_base_path,
                    args.verbose
                )

                # 合并到主统计对象
                stats.merge(file_stats)

                # 检查是否需要自动保存
                if auto_saver and auto_saver.should_save(stats.total_entries):
                    txt_path, csv_path = auto_saver.save_progress(
                        stats, args.max_errors
                    )
                    msg = f"已保存中间结果: {stats.total_entries} 条目"
                    if use_progress:
                        tqdm.write(msg)
                    else:
                        print(msg)

            except Exception as e:
                error_msg = f"处理文件 {jsonl_file.name} 时出错: {e}"
                if use_progress:
                    tqdm.write(error_msg)
                else:
                    print(error_msg)
                continue

            if args.verbose:
                print(f"  完成")
                print()

    # 生成并显示报告
    print("\n验证完成！\n")
    report = stats.generate_report(max_errors=args.max_errors)
    print(report)

    # 保存最终报告
    if auto_saver:
        txt_path, csv_path = auto_saver.save_final(stats, args.max_errors)
        print(f"\n最终TXT报告已保存到: {txt_path}")
        print(f"最终CSV报告已保存到: {csv_path}")
    else:
        # 使用原来的保存方式
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_report_path = os.path.join(args.output_dir, f"validation_report_{timestamp}.txt")
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n文本报告已保存到: {txt_report_path}")

        csv_path = os.path.join(args.output_dir, f"validation_report_{timestamp}.csv")
        stats.save_csv_report(csv_path)


if __name__ == "__main__":
    main()

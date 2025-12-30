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
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime

try:
    from PIL import Image, ImageFile
except ImportError:
    print("错误：需要安装 Pillow 库。请运行：pip install Pillow")
    sys.exit(1)

# PIL 图像处理配置（参考 data/data_utils.py）
Image.MAX_IMAGE_PIXELS = 20000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================================
# 默认配置
# ============================================================================
DEFAULT_JSONL_DIR = "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/jsonl"
DEFAULT_IMAGE_BASE_PATH = "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/images"
DEFAULT_OUTPUT_DIR = "./validation_reports"


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

        # 按任务类型统计
        self.task_type_stats = defaultdict(lambda: {"total": 0, "passed": 0})
        self.degrade_type_stats = defaultdict(lambda: {"total": 0, "passed": 0})

        # 分辨率统计
        self.input_widths = []
        self.input_heights = []
        self.output_widths = []
        self.output_heights = []

        # 问题详情
        self.error_details = []

        # 问题类型计数
        self.error_type_counts = defaultdict(int)

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
                    self.error_type_counts["缺失字段"] += 1
                elif "文件不存在" in error:
                    self.error_type_counts["图像文件不存在"] += 1
                elif "无法打开图像" in error:
                    self.error_type_counts["图像不可读"] += 1
                elif "分辨率不匹配" in error:
                    self.error_type_counts["分辨率不匹配"] += 1
                elif "message" in error.lower():
                    self.error_type_counts["Message格式错误"] += 1
                else:
                    self.error_type_counts["其他错误"] += 1

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
        if not self.error_details:
            print(f"没有错误需要保存到CSV")
            return

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "JSONL文件名", "行号", "任务类型", "退化类型", "错误数量", "错误详情"
            ])

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

        print(f"CSV报告已保存到: {output_path}")


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
  # 使用默认云端路径
  python validate_jsonl_dataset.py

  # 自定义路径
  python validate_jsonl_dataset.py --jsonl_dir /path/to/jsonl --image_base_path /path/to/images

  # 生成CSV报告
  python validate_jsonl_dataset.py --save_csv --max_errors 50
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
        help=f'验证报告输出目录 (默认: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--save_csv',
        action='store_true',
        help='保存CSV格式的详细错误报告'
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

    return parser.parse_args()


def main():
    """主函数"""
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
    if args.save_csv:
        os.makedirs(args.output_dir, exist_ok=True)

    # 查找所有JSONL文件
    jsonl_files = list(Path(args.jsonl_dir).glob("*.jsonl"))

    if not jsonl_files:
        print(f"错误：在 {args.jsonl_dir} 中未找到任何 .jsonl 文件")
        sys.exit(1)

    print(f"找到 {len(jsonl_files)} 个JSONL文件")
    print(f"图像基础路径: {args.image_base_path}")
    print("开始验证...\n")

    # 初始化统计收集器
    stats = ValidationStatistics()
    stats.total_files = len(jsonl_files)

    # 遍历每个JSONL文件
    for jsonl_file in sorted(jsonl_files):
        print(f"处理: {jsonl_file.name}")

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)

                    # 验证条目
                    passed, errors, metadata = validate_single_entry(
                        entry, line_num, args.image_base_path, args.verbose
                    )

                    # 添加到统计
                    stats.add_entry(str(jsonl_file), line_num, passed, errors, metadata)

                except json.JSONDecodeError as e:
                    error_msg = f"JSON解析错误: {str(e)}"
                    stats.add_entry(
                        str(jsonl_file), line_num, False,
                        [error_msg],
                        {"main_task_type": "unknown", "degrade_type": "N/A"}
                    )
                    if args.verbose:
                        print(f"  [行 {line_num}] {error_msg}")

        except Exception as e:
            print(f"  错误：无法读取文件 - {str(e)}")
            continue

        if args.verbose:
            print(f"  完成，共 {len(lines)} 条")
        print()

    # 生成并显示报告
    print("\n验证完成！\n")
    report = stats.generate_report(max_errors=args.max_errors)
    print(report)

    # 保存CSV报告
    if args.save_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = os.path.join(args.output_dir, f"validation_report_{timestamp}.csv")
        stats.save_csv_report(csv_path)


if __name__ == "__main__":
    main()

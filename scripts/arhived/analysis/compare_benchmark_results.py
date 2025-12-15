#!/usr/bin/env python3
"""
Benchmark Results Comparison Script

This script compares the performance of Baseline (single-expert) and MoT (full architecture)
experiments by analyzing the training_metrics.csv files.

Usage:
    python scripts/compare_benchmark_results.py

Output:
    - output/baseline_vs_mot_comparison.png (visualization)
    - output/comparison_summary.txt (detailed statistics)
    - Console output (comparison table)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ===== Configuration =====
BASELINE_CSV = "output/benchmark_baseline_vqa_1000steps/logs/training_metrics.csv"
MOT_CSV = "output/benchmark_mot_vqa_1000steps/logs/training_metrics.csv"
OUTPUT_PLOT = "output/baseline_vs_mot_comparison.png"
OUTPUT_SUMMARY = "output/comparison_summary.txt"
WARMUP_STEPS = 100  # Exclude warmup period from statistics

def check_files_exist():
    """Check if required CSV files exist."""
    missing_files = []
    if not os.path.exists(BASELINE_CSV):
        missing_files.append(BASELINE_CSV)
    if not os.path.exists(MOT_CSV):
        missing_files.append(MOT_CSV)

    if missing_files:
        print("❌ Error: Missing CSV files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease run the benchmark experiments first:")
        print("   bash scripts/benchmark_baseline_vqa.sh")
        print("   bash scripts/benchmark_mot_vqa.sh")
        sys.exit(1)

def load_data():
    """Load CSV files and basic validation."""
    print("📂 Loading data...")
    print(f"   Baseline: {BASELINE_CSV}")
    print(f"   MoT:      {MOT_CSV}")

    df_baseline = pd.read_csv(BASELINE_CSV)
    df_mot = pd.read_csv(MOT_CSV)

    print(f"   Baseline: {len(df_baseline)} steps loaded")
    print(f"   MoT:      {len(df_mot)} steps loaded")

    return df_baseline, df_mot

def compute_stats(df, name):
    """Compute statistics from dataframe."""
    stats = {
        'name': name,
        'total_steps': len(df),
        'avg_ce_loss': df['ce_loss'].mean(),
        'std_ce_loss': df['ce_loss'].std(),
        'avg_mse_loss': df['mse_loss'].mean(),
        'avg_sec_per_step': df['sec_per_step'].mean(),
        'std_sec_per_step': df['sec_per_step'].std(),
        'min_sec_per_step': df['sec_per_step'].min(),
        'max_sec_per_step': df['sec_per_step'].max(),
        'avg_peak_memory_gb': df['peak_memory_gb'].mean(),
        'max_peak_memory_gb': df['peak_memory_gb'].max(),
        'min_peak_memory_gb': df['peak_memory_gb'].min(),
        'final_ce_loss': df['ce_loss'].iloc[-10:].mean(),  # Last 10 steps average
    }
    return stats

def print_comparison_table(baseline_stats, mot_stats):
    """Print comparison table to console."""
    print("\n" + "="*90)
    print("Baseline vs MoT 实验结果对比")
    print("="*90)
    print(f"{'指标':<35} {'Baseline':<20} {'MoT':<20} {'差异':<15}")
    print("-"*90)

    metrics = [
        ('总步数', 'total_steps', False),
        ('平均 CE Loss', 'avg_ce_loss', False),
        ('最终 CE Loss (last 10 steps)', 'final_ce_loss', False),
        ('平均 Sec/Step', 'avg_sec_per_step', True),
        ('Sec/Step 标准差', 'std_sec_per_step', False),
        ('最小 Sec/Step', 'min_sec_per_step', False),
        ('最大 Sec/Step', 'max_sec_per_step', False),
        ('平均峰值 GPU Memory (GB)', 'avg_peak_memory_gb', True),
        ('最大峰值 GPU Memory (GB)', 'max_peak_memory_gb', True),
    ]

    for metric_name, key, compute_overhead in metrics:
        baseline_val = baseline_stats[key]
        mot_val = mot_stats[key]

        if compute_overhead and baseline_val > 0:
            diff_pct = ((mot_val - baseline_val) / baseline_val) * 100
            print(f"{metric_name:<35} {baseline_val:<20.4f} {mot_val:<20.4f} {diff_pct:>+14.2f}%")
        else:
            diff_abs = mot_val - baseline_val
            if key in ['avg_ce_loss', 'final_ce_loss']:
                print(f"{metric_name:<35} {baseline_val:<20.6f} {mot_val:<20.6f} {diff_abs:>+14.6f}")
            else:
                print(f"{metric_name:<35} {baseline_val:<20.4f} {mot_val:<20.4f} {diff_abs:>+14.4f}")

    print("="*90)

    # Calculate and display key overheads
    time_overhead = ((mot_stats['avg_sec_per_step'] - baseline_stats['avg_sec_per_step'])
                     / baseline_stats['avg_sec_per_step']) * 100
    memory_overhead = ((mot_stats['max_peak_memory_gb'] - baseline_stats['max_peak_memory_gb'])
                       / baseline_stats['max_peak_memory_gb']) * 100

    print("\n🔍 关键开销总结：")
    print(f"   时间开销 (Sec/Step):    {time_overhead:+.2f}%")
    print(f"   显存开销 (Peak Memory): {memory_overhead:+.2f}%")
    print(f"   Loss 差异 (Final CE):   {((mot_stats['final_ce_loss'] - baseline_stats['final_ce_loss']) / baseline_stats['final_ce_loss']) * 100:+.2f}%")
    print("="*90 + "\n")

def plot_comparison(df_baseline, df_mot, baseline_stats, mot_stats):
    """Generate comparison plots."""
    print("📊 生成对比图表...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Set Chinese font (fallback to default if not available)
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    # 1. Sec/Step Comparison
    axes[0, 0].plot(df_baseline['step'], df_baseline['sec_per_step'],
                    label='Baseline', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(df_mot['step'], df_mot['sec_per_step'],
                    label='MoT', alpha=0.7, linewidth=1.5)
    axes[0, 0].axvline(x=WARMUP_STEPS, color='red', linestyle='--',
                       label='Warmup End', alpha=0.5)
    axes[0, 0].set_xlabel('Training Step', fontsize=12)
    axes[0, 0].set_ylabel('Sec/Step', fontsize=12)
    axes[0, 0].set_title('Training Speed Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Peak Memory Comparison
    axes[0, 1].plot(df_baseline['step'], df_baseline['peak_memory_gb'],
                    label='Baseline', alpha=0.7, linewidth=1.5)
    axes[0, 1].plot(df_mot['step'], df_mot['peak_memory_gb'],
                    label='MoT', alpha=0.7, linewidth=1.5)
    axes[0, 1].axvline(x=WARMUP_STEPS, color='red', linestyle='--',
                       label='Warmup End', alpha=0.5)
    axes[0, 1].set_xlabel('Training Step', fontsize=12)
    axes[0, 1].set_ylabel('Peak GPU Memory (GB)', fontsize=12)
    axes[0, 1].set_title('GPU Memory Usage Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. CE Loss Comparison
    axes[1, 0].plot(df_baseline['step'], df_baseline['ce_loss'],
                    label='Baseline', alpha=0.7, linewidth=1.5)
    axes[1, 0].plot(df_mot['step'], df_mot['ce_loss'],
                    label='MoT', alpha=0.7, linewidth=1.5)
    axes[1, 0].axvline(x=WARMUP_STEPS, color='red', linestyle='--',
                       label='Warmup End', alpha=0.5)
    axes[1, 0].set_xlabel('Training Step', fontsize=12)
    axes[1, 0].set_ylabel('CE Loss', fontsize=12)
    axes[1, 0].set_title('Cross-Entropy Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Summary Bar Chart
    metrics_names = ['Sec/Step\n(lower better)', 'Peak Memory (GB)\n(lower better)']
    baseline_vals = [baseline_stats['avg_sec_per_step'], baseline_stats['max_peak_memory_gb']]
    mot_vals = [mot_stats['avg_sec_per_step'], mot_stats['max_peak_memory_gb']]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = axes[1, 1].bar(x - width/2, baseline_vals, width,
                           label='Baseline', alpha=0.8, color='#4CAF50')
    bars2 = axes[1, 1].bar(x + width/2, mot_vals, width,
                           label='MoT', alpha=0.8, color='#FF9800')

    axes[1, 1].set_ylabel('Value', fontsize=12)
    axes[1, 1].set_title('Average Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names, fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"   ✅ 对比图已保存: {OUTPUT_PLOT}")

def save_summary(baseline_stats, mot_stats):
    """Save detailed summary to file."""
    print("💾 保存详细统计...")

    os.makedirs(os.path.dirname(OUTPUT_SUMMARY) if os.path.dirname(OUTPUT_SUMMARY) else '.', exist_ok=True)

    with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Baseline vs MoT 实验详细结果\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write("="*80 + "\n\n")

        f.write(f"## Baseline 统计 (步骤 {WARMUP_STEPS}-{baseline_stats['total_steps']})\n")
        f.write("-"*80 + "\n")
        for key, val in baseline_stats.items():
            if isinstance(val, float):
                f.write(f"  {key}: {val:.6f}\n")
            else:
                f.write(f"  {key}: {val}\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"## MoT 统计 (步骤 {WARMUP_STEPS}-{mot_stats['total_steps']})\n")
        f.write("-"*80 + "\n")
        for key, val in mot_stats.items():
            if isinstance(val, float):
                f.write(f"  {key}: {val:.6f}\n")
            else:
                f.write(f"  {key}: {val}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("## MoT 相对 Baseline 的开销\n")
        f.write("-"*80 + "\n")

        time_overhead = ((mot_stats['avg_sec_per_step'] - baseline_stats['avg_sec_per_step'])
                         / baseline_stats['avg_sec_per_step']) * 100
        memory_overhead = ((mot_stats['max_peak_memory_gb'] - baseline_stats['max_peak_memory_gb'])
                           / baseline_stats['max_peak_memory_gb']) * 100
        loss_diff = ((mot_stats['final_ce_loss'] - baseline_stats['final_ce_loss'])
                     / baseline_stats['final_ce_loss']) * 100

        f.write(f"  时间开销 (Sec/Step):        {time_overhead:+.2f}%\n")
        f.write(f"  显存开销 (Peak Memory):     {memory_overhead:+.2f}%\n")
        f.write(f"  Loss 差异 (Final CE Loss): {loss_diff:+.2f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("## 结论\n")
        f.write("-"*80 + "\n")

        if time_overhead > 0:
            f.write(f"MoT 架构相比 Baseline 慢 {time_overhead:.2f}%，")
        else:
            f.write(f"MoT 架构相比 Baseline 快 {-time_overhead:.2f}%，")

        if memory_overhead > 0:
            f.write(f"多用 {memory_overhead:.2f}% 显存。\n")
        else:
            f.write(f"少用 {-memory_overhead:.2f}% 显存。\n")

        f.write(f"两者的最终 Loss 差异为 {loss_diff:+.2f}%，")
        if abs(loss_diff) < 1.0:
            f.write("训练效果基本一致。\n")
        elif loss_diff < 0:
            f.write("MoT 略优于 Baseline。\n")
        else:
            f.write("Baseline 略优于 MoT。\n")

        f.write("\n这些开销主要来自：\n")
        f.write("  1. VAE 模型的初始化和内存占用\n")
        f.write("  2. 额外的投影层 (vae2llm, llm2vae, timestep_embedder)\n")
        f.write("  3. MoE 路由机制的计算开销\n")
        f.write("  4. 生成分支专家的参数和计算\n")

        f.write("="*80 + "\n")

    print(f"   ✅ 详细统计已保存: {OUTPUT_SUMMARY}")

def main():
    """Main function."""
    print("\n" + "="*80)
    print("📊 Benchmark Results Comparison")
    print("="*80 + "\n")

    # Check files
    check_files_exist()

    # Load data
    df_baseline, df_mot = load_data()

    # Filter warmup period
    print(f"\n🔍 过滤 warmup 阶段 (前 {WARMUP_STEPS} 步)...")
    df_baseline_stable = df_baseline[df_baseline['step'] >= WARMUP_STEPS].copy()
    df_mot_stable = df_mot[df_mot['step'] >= WARMUP_STEPS].copy()
    print(f"   Baseline: {len(df_baseline_stable)} 步用于统计")
    print(f"   MoT:      {len(df_mot_stable)} 步用于统计")

    # Compute statistics
    print("\n📈 计算统计指标...")
    baseline_stats = compute_stats(df_baseline_stable, "Baseline")
    mot_stats = compute_stats(df_mot_stable, "MoT")

    # Print comparison
    print_comparison_table(baseline_stats, mot_stats)

    # Plot comparison
    plot_comparison(df_baseline, df_mot, baseline_stats, mot_stats)

    # Save summary
    save_summary(baseline_stats, mot_stats)

    print("\n✅ 分析完成！")
    print(f"   📊 可视化图表: {OUTPUT_PLOT}")
    print(f"   📄 详细报告:   {OUTPUT_SUMMARY}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

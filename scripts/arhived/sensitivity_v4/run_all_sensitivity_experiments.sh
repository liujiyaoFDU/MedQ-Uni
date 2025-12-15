#!/usr/bin/env bash
# ============================================================================
# Master Script: Run All 5 Sensitivity Experiments + Analysis
# ============================================================================
# Purpose: Sequentially run all sensitivity experiments and generate comparison
# Total estimated time: ~1.5-2 hours (5 experiments × 15-20 min each)
# ============================================================================



# ===== Environment Setup =====
echo "============================================================================"
echo "  Loss Weight Sensitivity Experiments - Master Script"
echo "============================================================================"
echo ""
echo "[STEP 0] Setting up environment..."

source /inspire/hdd/global_user/hejunjun-24017/junzhin/.bashrc
cd /inspire/hdd/global_user/hejunjun-24017/junzhin/projects/Uni-MedVL

eval "$(conda shell.bash hook)"
conda activate bagel

echo "   ✓ Environment activated: unimedvl_gzy"
echo "   ✓ Working directory: $(pwd)"
echo ""

# ===== Verify Scripts Exist =====
SCRIPTS=(
    "scripts/sensitivity_v4/sensitivity_exp1_ce4.0_mse1.0.sh"
    "scripts/sensitivity_v4/sensitivity_exp2_ce2.0_mse1.0.sh"
    "scripts/sensitivity_v4/sensitivity_exp3_ce1.0_mse1.0.sh"
    "scripts/sensitivity_v4/sensitivity_exp4_ce0.5_mse1.0.sh"
    "scripts/sensitivity_v4/sensitivity_exp5_ce0.25_mse1.0.sh"
    "scripts/compare_sensitivity_5experiments.py"
)

echo "[VERIFICATION] Checking if all scripts exist..."
MISSING=0
for script in "${SCRIPTS[@]}"; do
    if [[ ! -f "$script" ]]; then
        echo "   ✗ Missing: $script"
        MISSING=1
    else
        echo "   ✓ Found: $script"
    fi
done

if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "❌ Error: Some scripts are missing. Please create them first."
    exit 1
fi

echo ""
echo "✅ All scripts verified!"
echo ""

# ===== Record Start Time =====
START_TIME=$(date +%s)
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================================"
echo "  Starting Experiments at: ${START_TIMESTAMP}"
echo "============================================================================"
echo ""

# ===== Run Experiment 1 =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/5] Running Experiment 1: ce_weight=4.0 (Understanding-Heavy)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
EXP1_START=$(date +%s)

bash scripts/sensitivity_v4/sensitivity_exp1_ce4.0_mse1.0.sh

EXP1_END=$(date +%s)
EXP1_DURATION=$((EXP1_END - EXP1_START))
echo ""
echo "✅ Experiment 1 completed in $((EXP1_DURATION / 60)) min $((EXP1_DURATION % 60)) sec"
echo ""

# ===== Run Experiment 2 =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/5] Running Experiment 2: ce_weight=2.0 (Understanding-Moderate)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
EXP2_START=$(date +%s)

bash scripts/sensitivity_v4/sensitivity_exp2_ce2.0_mse1.0.sh

EXP2_END=$(date +%s)
EXP2_DURATION=$((EXP2_END - EXP2_START))
echo ""
echo "✅ Experiment 2 completed in $((EXP2_DURATION / 60)) min $((EXP2_DURATION % 60)) sec"
echo ""

# ===== Run Experiment 3 =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/5] Running Experiment 3: ce_weight=1.0 (Balanced)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
EXP3_START=$(date +%s)

bash scripts/sensitivity_v4/sensitivity_exp3_ce1.0_mse1.0.sh

EXP3_END=$(date +%s)
EXP3_DURATION=$((EXP3_END - EXP3_START))
echo ""
echo "✅ Experiment 3 completed in $((EXP3_DURATION / 60)) min $((EXP3_DURATION % 60)) sec"
echo ""

# ===== Run Experiment 4 =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/5] Running Experiment 4: ce_weight=0.5 (Generation-Moderate)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
EXP4_START=$(date +%s)

bash scripts/sensitivity_v4/sensitivity_exp4_ce0.5_mse1.0.sh

EXP4_END=$(date +%s)
EXP4_DURATION=$((EXP4_END - EXP4_START))
echo ""
echo "✅ Experiment 4 completed in $((EXP4_DURATION / 60)) min $((EXP4_DURATION % 60)) sec"
echo ""

# ===== Run Experiment 5 =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/5] Running Experiment 5: ce_weight=0.25 (Generation-Heavy - Paper Setting)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
EXP5_START=$(date +%s)

bash scripts/sensitivity_v4/sensitivity_exp5_ce0.25_mse1.0.sh

EXP5_END=$(date +%s)
EXP5_DURATION=$((EXP5_END - EXP5_START))
echo ""
echo "✅ Experiment 5 completed in $((EXP5_DURATION / 60)) min $((EXP5_DURATION % 60)) sec"
echo ""

# ===== Calculate Total Training Time =====
TOTAL_TRAINING_TIME=$((EXP5_END - START_TIME))

echo "============================================================================"
echo "  All 5 Experiments Completed!"
echo "============================================================================"
echo ""
echo "📊 Training Time Summary:"
echo "   Experiment 1 (ce=4.0):  $((EXP1_DURATION / 60)) min $((EXP1_DURATION % 60)) sec"
echo "   Experiment 2 (ce=2.0):  $((EXP2_DURATION / 60)) min $((EXP2_DURATION % 60)) sec"
echo "   Experiment 3 (ce=1.0):  $((EXP3_DURATION / 60)) min $((EXP3_DURATION % 60)) sec"
echo "   Experiment 4 (ce=0.5):  $((EXP4_DURATION / 60)) min $((EXP4_DURATION % 60)) sec"
echo "   Experiment 5 (ce=0.25): $((EXP5_DURATION / 60)) min $((EXP5_DURATION % 60)) sec"
echo "   ────────────────────────────────────────"
echo "   Total Training Time:    $((TOTAL_TRAINING_TIME / 60)) min $((TOTAL_TRAINING_TIME % 60)) sec"
echo ""

# ===== Verify CSV Files =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Verifying CSV Output Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CSV_FILES=(
    "output/sensitivity_exp1_ce4.0_mse1.0_1000steps_v4_noclip_lr0.01/logs/training_metrics.csv"
    "output/sensitivity_exp2_ce2.0_mse1.0_1000steps_v4_noclip_lr0.01/logs/training_metrics.csv"
    "output/sensitivity_exp3_ce1.0_mse1.0_1000steps_v4_noclip_lr0.01/logs/training_metrics.csv"
    "output/sensitivity_exp4_ce0.5_mse1.0_1000steps_v4_noclip_lr0.01/logs/training_metrics.csv"
    "output/sensitivity_exp5_ce0.25_mse1.0_1000steps_v4_noclip_lr0.01/logs/training_metrics.csv"
)

ALL_CSV_EXIST=1
for csv in "${CSV_FILES[@]}"; do
    if [[ -f "$csv" ]]; then
        LINES=$(wc -l < "$csv")
        echo "   ✓ $csv ($LINES lines)"
    else
        echo "   ✗ Missing: $csv"
        ALL_CSV_EXIST=0
    fi
done

echo ""

if [[ $ALL_CSV_EXIST -eq 0 ]]; then
    echo "⚠️  Warning: Some CSV files are missing. Analysis may fail."
    echo ""
    echo "❌ Skipping analysis step. Please check experiment logs for errors."
    echo ""
    exit 1
fi

# ===== Run Analysis Script =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running Comparison Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ANALYSIS_START=$(date +%s)

python scripts/compare_sensitivity_5experiments.py

ANALYSIS_END=$(date +%s)
ANALYSIS_DURATION=$((ANALYSIS_END - ANALYSIS_START))

echo ""
echo "✅ Analysis completed in ${ANALYSIS_DURATION} seconds"
echo ""

# ===== Final Summary =====
END_TIME=$(date +%s)
END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "============================================================================"
echo "  🎉 ALL TASKS COMPLETED SUCCESSFULLY!"
echo "============================================================================"
echo ""
echo "⏱️  Time Summary:"
echo "   Started at:        ${START_TIMESTAMP}"
echo "   Finished at:       ${END_TIMESTAMP}"
echo "   Total duration:    $((TOTAL_DURATION / 60)) min $((TOTAL_DURATION % 60)) sec"
echo ""
echo "📁 Output Files:"
echo "   📊 Visualization:  output/sensitivity_loss_weight_5exp_comparison.png"
echo "   📄 Report:         output/sensitivity_5exp_report.txt"
echo ""
echo "📊 Experiment Data:"
for csv in "${CSV_FILES[@]}"; do
    echo "   📈 $csv"
done
echo ""
echo "============================================================================"
echo "  Next Steps:"
echo "============================================================================"
echo "   1. View the comparison plot:"
echo "      open output/sensitivity_loss_weight_5exp_comparison.png"
echo ""
echo "   2. Read the detailed report:"
echo "      cat output/sensitivity_5exp_report.txt"
echo ""
echo "   3. Use the LaTeX table code in the report for your paper"
echo ""
echo "   4. Key findings will show:"
echo "      • CE Loss correlation with ce_weight (Pearson r and p-value)"
echo "      • MSE Loss correlation with ce_weight"
echo "      • Training efficiency stability (<2% variance)"
echo "      • Paper setting (ce=0.25) optimality for generation task"
echo ""
echo "============================================================================"
echo ""

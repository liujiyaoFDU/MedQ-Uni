#!/usr/bin/env bash
# ============================================================================
# Setup DeepSpeed environment on cluster (offline, no internet needed)
# 在集群上离线创建 DeepSpeed 环境
# ============================================================================
# Usage: bash scripts/setup_deepspeed_env.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_SRC="${SCRIPT_DIR}/.venv"
VENV_DST="${SCRIPT_DIR}/.venv_deepspeed"
WHEELS_DIR="${SCRIPT_DIR}/deepspeed_wheels"

echo "============================================"
echo "[Setup] Creating DeepSpeed venv from existing .venv"
echo "[Setup] Source: ${VENV_SRC}"
echo "[Setup] Target: ${VENV_DST}"
echo "============================================"

# Step 1: Check source venv exists
if [[ ! -d "${VENV_SRC}" ]]; then
    echo "[ERROR] Source venv not found: ${VENV_SRC}"
    exit 1
fi

# Step 2: Check wheels exist
if [[ ! -d "${WHEELS_DIR}" ]] || [[ -z "$(ls ${WHEELS_DIR}/*.whl 2>/dev/null)" ]]; then
    echo "[ERROR] Wheel files not found in: ${WHEELS_DIR}"
    exit 1
fi

# Step 3: Copy venv (skip if already exists)
if [[ -d "${VENV_DST}" ]]; then
    echo "[INFO] ${VENV_DST} already exists, skipping copy"
else
    echo "[INFO] Copying venv... (this may take a moment)"
    cp -a "${VENV_SRC}" "${VENV_DST}"
    echo "[INFO] Copy done"
fi

# Step 4: Fix activate scripts (hardcoded VIRTUAL_ENV path)
echo "[INFO] Fixing activate scripts..."
for f in activate activate.csh activate.fish activate.nu activate.bat; do
    fpath="${VENV_DST}/bin/${f}"
    if [[ -f "${fpath}" ]]; then
        sed -i "s|${VENV_SRC}|${VENV_DST}|g" "${fpath}"
    fi
done
echo "[INFO] Activate scripts fixed"

# Step 5: Bootstrap pip if missing
if [[ ! -f "${VENV_DST}/bin/pip" ]] && [[ ! -f "${VENV_DST}/bin/pip3" ]]; then
    echo "[INFO] Bootstrapping pip..."
    "${VENV_DST}/bin/python" -m ensurepip --upgrade 2>/dev/null || true
fi

# Find pip executable
if [[ -f "${VENV_DST}/bin/pip" ]]; then
    PIP="${VENV_DST}/bin/pip"
elif [[ -f "${VENV_DST}/bin/pip3" ]]; then
    PIP="${VENV_DST}/bin/pip3"
else
    PIP="${VENV_DST}/bin/python -m pip"
fi

# Step 6: Install deepspeed from local wheels (offline)
echo "[INFO] Installing DeepSpeed from local wheels..."
${PIP} install --no-index --find-links="${WHEELS_DIR}" deepspeed 2>&1
echo "[INFO] Installation done"

# Step 7: Verify
echo "============================================"
echo "[Verify] Testing DeepSpeed import..."
"${VENV_DST}/bin/python" -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__} installed successfully')"
echo "============================================"
echo "[Done] Environment ready at: ${VENV_DST}"
echo "[Done] Training script can now be run:"
echo "  bash scripts/training/train_sft_stage1_medq_unif_multinode_sr_ssim_loss_v3_deepspeed.sh"
echo "============================================"

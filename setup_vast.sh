#!/bin/bash
# TOPOLOGY OVER SUBSTANCE — GPU Benchmark Setup
# Run this on the Vast.ai instance after SSH in

set -e

echo "============================================"
echo "  CHESTOHEDRON DEFINITIVE BENCHMARK SETUP"
echo "============================================"

# Install deps
pip install torch numpy --quiet
echo "  [OK] PyTorch + NumPy installed"

# Clone repo
cd /root
git clone https://github.com/hugoboss23-5/jibblets.git
cd jibblets
git checkout claude/neural-networks-software-hardware-OBP5d
echo "  [OK] Repo cloned, on neural-networks branch"

# Verify CUDA
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"

# Quick sanity check first (2 seeds, 5 epochs)
echo ""
echo "  Running QUICK sanity check..."
python master_benchmark.py --quick

echo ""
echo "  Quick check passed. Starting FULL benchmark..."
echo "  (7 archs × 2 datasets × 10 seeds = 140 runs)"
echo ""

# Full benchmark
python master_benchmark.py 2>&1 | tee benchmark_log.txt

# Push results back
git add -A
git commit -m "Definitive benchmark: 7 archs × 2 datasets × 10 seeds"
git push origin claude/neural-networks-software-hardware-OBP5d

echo ""
echo "  DONE. Results committed and pushed."

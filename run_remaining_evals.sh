#!/bin/bash
set -e

echo "==========================================================="
echo "STEP 1: RUNNING ROBUSTNESS EVALUATION (MODALITY & NOISE)"
echo "==========================================================="
python3 run_robustness.py

echo "==========================================================="
echo "STEP 2: RUNNING GRAD-CAM & TOKEN SIMILARITY VISUALIZATION"
echo "==========================================================="
python3 run_gradcam.py

echo "==========================================================="
echo "STEP 3: RUNNING ADVANCED BENCHMARKS (LATENCY, ECE, ERRORS)"
echo "==========================================================="
python3 run_advanced_benchmarks.py

echo "==========================================================="
echo "ALL EVALUATIONS COMPLETED SUCCESSFULLY!"
echo "==========================================================="

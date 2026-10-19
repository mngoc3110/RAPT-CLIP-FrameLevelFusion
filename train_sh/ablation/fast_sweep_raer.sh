#!/bin/bash
# ==============================================================================
# RAPT-CLIP: Master Fast Screening & Feature-Level Ablation Engine for RAER
# ==============================================================================
# Usage:
#   1. Full Ablation Matrix Sweep (Runs all 9 variants in ~2 minutes):
#        bash train_sh/ablation/fast_sweep_raer.sh --sweep
#
#   2. Run Single Variant (e.g. No LDAM, No Adapter, GFI):
#        bash train_sh/ablation/fast_sweep_raer.sh --config No_LDAM
#
#   3. Extract Features Only (Cache ViT-B/16 tokens for reuse):
#        bash train_sh/ablation/fast_sweep_raer.sh --extract-only
#
#   4. Force Re-extraction of Features:
#        bash train_sh/ablation/fast_sweep_raer.sh --force-extract --sweep
# ==============================================================================

set -e

# Defaults
MODE="sweep"
CONFIG="Full_RAPT_CLIP"
EPOCHS=20
DEVICE="auto"
FORCE_EXTRACT=false
EXTRACT_ONLY=false
BATCH_SIZE=16
WORKERS=4
FEATURES_DIR="cached_features/raer"
OUT_MD="outputs/raer_fast_ablation_summary.md"

# Parse CLI arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --sweep) MODE="sweep"; shift ;;
    --config) MODE="single"; CONFIG="$2"; shift 2 ;;
    --extract-only) EXTRACT_ONLY=true; shift ;;
    --force-extract) FORCE_EXTRACT=true; shift ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --features-dir) FEATURES_DIR="$2"; shift 2 ;;
    --out-md) OUT_MD="$2"; shift 2 ;;
    -h|--help)
      echo "RAPT-CLIP Fast RAER Ablation Runner"
      echo "Options:"
      echo "  --sweep             Run complete 9-experiment ablation matrix (default)"
      echo "  --config <NAME>     Run single config (e.g. Full_RAPT_CLIP, No_LDAM, Fusion_GFI)"
      echo "  --extract-only      Run feature extraction and exit"
      echo "  --force-extract     Re-extract features even if cached files exist"
      echo "  --epochs <N>        Epochs per ablation experiment (default: 20)"
      echo "  --device <DEV>      Device: auto, cuda, mps, cpu (default: auto)"
      echo "  --features-dir <P>  Cache directory (default: cached_features/raer)"
      echo "  --out-md <PATH>     Markdown report output (default: outputs/raer_fast_ablation_summary.md)"
      exit 0
      ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Display Header
echo "=============================================================================="
echo "  🚀 RAPT-CLIP: RAER ULTRA-FAST FEATURE-LEVEL ABLATION SUITE"
echo "=============================================================================="
echo "  • Mode:             $([ "$EXTRACT_ONLY" = true ] && echo 'Feature Extraction Only' || ([ "$MODE" = 'sweep' ] && echo 'Full Matrix Sweep' || echo "Single Config ($CONFIG)"))"
echo "  • Epochs:           $EPOCHS"
echo "  • Device:           $DEVICE"
echo "  • Features Cache:   $FEATURES_DIR"
echo "  • Report Markdown:  $OUT_MD"
echo "=============================================================================="

mkdir -p "$FEATURES_DIR"
mkdir -p outputs

TRAIN_FEATS="$FEATURES_DIR/raer_train_features.pt"
TEST_FEATS="$FEATURES_DIR/raer_test_features.pt"

# ------------------------------------------------------------------------------
# Step 1: Feature Extraction / Cache Check
# ------------------------------------------------------------------------------
NEEDS_EXTRACT=false
if [ ! -f "$TRAIN_FEATS" ] || [ ! -f "$TEST_FEATS" ] || [ "$FORCE_EXTRACT" = true ]; then
  NEEDS_EXTRACT=true
fi

if [ "$NEEDS_EXTRACT" = true ]; then
  echo ""
  echo "=> [STEP 1/2] Extracting and Caching Frozen CLIP Features for RAER..."
  echo "   (This runs once, producing ~170MB .pt files for instant 300x ablations)"
  
  python tools/extract_raer_features.py \
    --root-dir "." \
    --out-dir "$FEATURES_DIR" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --device "$DEVICE"

  if [ "$EXTRACT_ONLY" = true ]; then
    echo ""
    echo "✅ [SUCCESS] Feature extraction completed successfully in $FEATURES_DIR!"
    echo "   You can now run ablations anytime with:"
    echo "   bash train_sh/ablation/fast_sweep_raer.sh --sweep"
    exit 0
  fi
else
  echo ""
  echo "=> [STEP 1/2] Found existing pre-computed feature cache in '$FEATURES_DIR'!"
  echo "   • Train features: $(ls -lh "$TRAIN_FEATS" | awk '{print $5}')"
  echo "   • Test features:  $(ls -lh "$TEST_FEATS" | awk '{print $5}')"
  echo "   (Skipping extraction. Use --force-extract to recompute if needed)."
fi

if [ "$EXTRACT_ONLY" = true ]; then
  echo "=> Extract-only requested and cache is already up to date. Exiting."
  exit 0
fi

# ------------------------------------------------------------------------------
# Step 2: Feature-Level Ablation Execution
# ------------------------------------------------------------------------------
echo ""
echo "=> [STEP 2/2] Running Feature-Level Ablations..."

CMD="python tools/fast_ablation_raer.py \
  --features-dir $FEATURES_DIR \
  --epochs $EPOCHS \
  --device $DEVICE \
  --out-md $OUT_MD"

if [ "$MODE" = "sweep" ]; then
  echo "   Running full sweep across 9 ablation configurations..."
  CMD="$CMD --sweep"
else
  echo "   Running single configuration: $CONFIG..."
  CMD="$CMD --config $CONFIG"
fi

$CMD | tee outputs/fast_ablation_raer.log

echo ""
echo "=============================================================================="
echo "  🎉 ABLATION RUN COMPLETED SUCCESSFULLY!"
echo "  • Execution log:    outputs/fast_ablation_raer.log"
echo "  • Markdown Summary: $OUT_MD"
echo "=============================================================================="

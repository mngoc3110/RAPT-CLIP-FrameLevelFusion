#!/bin/bash
# ==============================================================================
# run_4seeds_raer.sh: Tự động chạy 4 seeds mới [7, 123, 456, 789] trên RAER
# (Seed 42 đã có sẵn checkpoint SOTA và được tự động tổng hợp vào bảng cuối cùng)
# ==============================================================================

GPU_ID=${1:-0}
WORKERS=${2:-4}

echo "=============================================================================="
echo "🎯 RAPT-CLIP: KHỞI ĐỘNG TIẾN TRÌNH HUẤN LUYỆN 4 SEEDS [7, 123, 456, 789]"
echo "   • Thiết bị GPU: cuda:$GPU_ID"
echo "   • Số Workers:   $WORKERS"
echo "   • Seed 42:      Đã có sẵn kết quả (UAR 77.66%), sẽ tự động ghép vào báo cáo."
echo "=============================================================================="

# Gọi trực tiếp bộ điều phối Python
python3 tools/run_multi_seed_raer.py \
    --seeds 7 123 456 789 \
    --gpu $GPU_ID \
    --workers $WORKERS

echo "🎉 HOÀN TẤT TOÀN BỘ TIẾN TRÌNH 5 SEEDS! Xem kết quả tại: outputs/raer_5seeds_summary.md"


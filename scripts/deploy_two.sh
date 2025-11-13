#!/bin/bash
# Deploy two-hands with independent checkpoints
# Usage:
#   ./scripts/deploy_two.sh CACHE                    # Same checkpoint for both hands
#   ./scripts/deploy_two.sh CACHE_RIGHT CACHE_LEFT  # Different checkpoints

CACHE_RIGHT=$1
CACHE_LEFT=$2

if [ -z "$CACHE_RIGHT" ]; then
    echo "Usage: $0 <cache_right> [cache_left]"
    echo ""
    echo "Examples:"
    echo "  $0 ckpt_name_01                    # Same checkpoint for both hands"
    echo "  $0 ckpt_name_01 ckpt_name_02       # Different checkpoints"
    exit 1
fi

# Build checkpoint paths
CHECKPOINT_RIGHT="outputs/AllegroHandHora/${CACHE_RIGHT}/stage2_nn/best.pth"

if [ -z "$CACHE_LEFT" ]; then
    # Use same checkpoint for both hands
    echo "🧠🧠 Using same checkpoint for both hands: ${CACHE_RIGHT}"
    python run_two.py +checkpoint_right="${CHECKPOINT_RIGHT}"
else
    # Use different checkpoints
    CHECKPOINT_LEFT="outputs/AllegroHandHora/${CACHE_LEFT}/stage2_nn/best.pth"
    echo "🧠 Right hand: ${CACHE_RIGHT}"
    echo "🧠 Left hand:  ${CACHE_LEFT}"
    python run_two.py +checkpoint_right="${CHECKPOINT_RIGHT}" +checkpoint_left="${CHECKPOINT_LEFT}"
fi

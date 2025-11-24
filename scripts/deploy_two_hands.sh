#!/bin/bash
# Deploy two hands with hand-specific checkpoints
# Right hand loads from RightAllegroHandHora
# Left hand loads from LeftAllegroHandHora
#
# Usage:
#   ./scripts/deploy_two_hands.sh CACHE                    # Same cache name for both hands
#   ./scripts/deploy_two_hands.sh CACHE_RIGHT CACHE_LEFT  # Different cache names

CACHE_RIGHT=$1
CACHE_LEFT=$2

if [ -z "$CACHE_RIGHT" ]; then
    echo "Usage: $0 <cache_right> [cache_left]"
    echo ""
    echo "Examples:"
    echo "  $0 experiment_01                    # Same cache name, different hand directories"
    echo "  $0 exp_right exp_left               # Different cache names"
    echo ""
    echo "Checkpoint paths:"
    echo "  Right: outputs/RightAllegroHandHora/<cache_right>/stage2_nn/best.pth"
    echo "  Left:  outputs/LeftAllegroHandHora/<cache_left>/stage2_nn/best.pth"
    exit 1
fi

# Build checkpoint paths from hand-specific directories
CHECKPOINT_RIGHT="outputs/RightAllegroHandHora/${CACHE_RIGHT}/stage2_nn/best.pth"

if [ -z "$CACHE_LEFT" ]; then
    # Use same cache name but from different hand directories
    CACHE_LEFT="${CACHE_RIGHT}"
    CHECKPOINT_LEFT="outputs/LeftAllegroHandHora/${CACHE_LEFT}/stage2_nn/best.pth"
    echo "🧠 Using cache name '${CACHE_RIGHT}' for both hands"
    echo "   Right: RightAllegroHandHora/${CACHE_RIGHT}"
    echo "   Left:  LeftAllegroHandHora/${CACHE_LEFT}"
else
    # Use different cache names
    CHECKPOINT_LEFT="outputs/LeftAllegroHandHora/${CACHE_LEFT}/stage2_nn/best.pth"
    echo "🧠 Right hand: RightAllegroHandHora/${CACHE_RIGHT}"
    echo "🧠 Left hand:  LeftAllegroHandHora/${CACHE_LEFT}"
fi

echo ""
echo "Checkpoint paths:"
echo "  Right: ${CHECKPOINT_RIGHT}"
echo "  Left:  ${CHECKPOINT_LEFT}"
echo ""

# Verify checkpoints exist
if [ ! -f "${CHECKPOINT_RIGHT}" ]; then
    echo "❌ Error: Right hand checkpoint not found: ${CHECKPOINT_RIGHT}"
    exit 1
fi

if [ ! -f "${CHECKPOINT_LEFT}" ]; then
    echo "❌ Error: Left hand checkpoint not found: ${CHECKPOINT_LEFT}"
    exit 1
fi

echo "✓ Both checkpoints found"
echo ""
echo "Starting deployment..."

python run.py +checkpoint_right="${CHECKPOINT_RIGHT}" +checkpoint_left="${CHECKPOINT_LEFT}"

#!/bin/bash
# Deploy single hand with checkpoint
# Right hand loads from RightAllegroHandHora
#
# Usage:
#   ./scripts/deploy_one_hand.sh CACHE
#
# Example:
#   ./scripts/deploy_one_hand.sh my_experiment
#
# Checkpoint path:
#   outputs/RightAllegroHandHora/<cache>/stage2_nn/best.pth

CACHE=$1

if [ -z "$CACHE" ]; then
    echo "Usage: $0 <cache_name>"
    echo ""
    echo "Example:"
    echo "  $0 my_experiment"
    echo ""
    echo "Checkpoint path:"
    echo "  outputs/RightAllegroHandHora/<cache_name>/stage2_nn/best.pth"
    exit 1
fi

CHECKPOINT="outputs/RightAllegroHandHora/${CACHE}/stage2_nn/best.pth"

echo "🧠 Single hand deployment"
echo "   Cache: ${CACHE}"
echo "   Checkpoint: ${CHECKPOINT}"
echo ""

# Verify checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "❌ Error: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "✓ Checkpoint found"
echo ""
echo "Starting deployment..."

python run.py checkpoint="${CHECKPOINT}"


set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== Réentraînement lancé le $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_DIR/retrain.log"
cd "$PROJECT_DIR"
PYTHON=$(command -v python3 || command -v python)
"$PYTHON" scripts/train_model.py >> "$LOG_DIR/retrain.log" 2>&1
echo "=== Terminé le $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_DIR/retrain.log"

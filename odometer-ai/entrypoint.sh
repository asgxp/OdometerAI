#!/bin/bash
set -e

echo "🚀 Starting AI Odometer..."

BASE_DIR="/sdms/ai-odometer"
MODEL_DIR="$BASE_DIR/models"
DEFAULT_DIR="$BASE_DIR/default_models"

# สร้างโฟลเดอร์ถ้ายังไม่มี
mkdir -p "$MODEL_DIR"

# ถ้าว่าง → copy default
if [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "📦 Models empty → copying from default..."
    cp -r "$DEFAULT_DIR"/* "$MODEL_DIR"/
else
    echo "✅ Models exist → skip copy"
fi

# Run main command
exec "$@"

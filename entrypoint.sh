#!/bin/bash
# entrypoint.sh — robust, no set -e at top

echo "🔧 Starting entrypoint..."

# Ensure HF_HOME exists
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
mkdir -p "$HF_HOME" 2>/dev/null || true

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ WARNING: HF_TOKEN not set — gated models (e.g. FLUX.1-schnell) will fail!"
else
    echo "🔑 HF_TOKEN is set (length: ${#HF_TOKEN})"
    # Try login via Python (most reliable)
    if python -c "
import os, sys
token = os.getenv('HF_TOKEN')
if not token:
    sys.exit(1)
try:
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print('✅ HF login via Python succeeded')
except Exception as e:
    print(f'❌ HF login failed: {e}', file=sys.stderr)
    sys.exit(1)
"; then
    echo "✅ Hugging Face login completed"
else
    echo "❌ HF login failed — but continuing (model may still load if token passed to from_pretrained)"
fi
fi

# Launch app
echo "🚀 Starting Uvicorn..."
exec "$@"
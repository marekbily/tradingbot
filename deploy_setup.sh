#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/trading-bot}"
SERVICE_NAME="trading-bot"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python 3 is required but was not found."
    exit 1
  fi
fi

if [ ! -d "$PROJECT_DIR" ]; then
  echo "Project directory not found: $PROJECT_DIR"
  echo "Clone or upload the repository first, then rerun this script."
  exit 1
fi

cd "$PROJECT_DIR"

sudo apt-get update
sudo apt-get install -y git build-essential python3-venv python3-pip

if [ ! -d venv ]; then
  "$PYTHON_BIN" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip

pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0"
pip install \
  numpy \
  pandas \
  stable-baselines3 \
  gymnasium \
  shimmy \
  python-dotenv \
  requests \
  scikit-learn \
  scipy \
  metaapi-cloud-sdk

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Fill in METAAPI_TOKEN and METAAPI_ACCOUNT_ID before starting the service."
fi

USER_NAME="${SUDO_USER:-${USER:-$(id -un)}}"
SERVICE_FILE="/tmp/${SERVICE_NAME}.service"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=DRL Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}
Environment="PYTHONPATH=${PROJECT_DIR}"
ExecStart=${PROJECT_DIR}/venv/bin/python ${PROJECT_DIR}/live_trade_metaapi.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo cp "$SERVICE_FILE" /etc/systemd/system/${SERVICE_NAME}.service
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}

echo "Setup complete. Edit .env, then start the bot with: sudo systemctl start ${SERVICE_NAME}"

set -euo pipefail
cd /workspace || mkdir -p /workspace && cd /workspace
if [ ! -d tradingbot ]; then git clone https://github.com/marekbily/tradingbot.git; fi
cd tradingbot
python3 -m venv .venv || true
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || true
mkdir -p train
nohup bash -lc "source .venv/bin/activate && python train/train_ultimate_150.py --steps 1000000 --device cuda --batch-size 128" > train/run_training.log 2>&1 &
echo $! > train/train.pid
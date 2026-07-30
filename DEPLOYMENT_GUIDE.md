# ForpsiCloud VPS Deployment Guide

This guide is tailored to your ForpsiCloud VPS:

- Ubuntu Server 24.04 LTS 64-bit
- 1 vCPU
- 1 GB RAM
- 20 GB disk
- Public IP: 194.182.83.34

The bot should run with MetaAPI on the VPS, not MT5 desktop. Use `live_trade_metaapi.py`.
For MetaAPI paper trading, you still need a broker demo MT4/MT5 account connected inside MetaAPI.

## 1. Connect To The VPS

From your local machine, SSH into the server using the username Forpsi gave you:

```bash
ssh YOUR_USER@194.182.83.34
```

If you log in as `root`, use that instead of `YOUR_USER`.

## 2. Upload The Repository

If the repository is already on GitHub, clone it on the server:

```bash
git clone YOUR_REPO_URL ~/trading-bot
cd ~/trading-bot
```

If you want to upload the current workspace directly, copy the folder from your PC to the VPS:

```bash
scp -r tradingbot YOUR_USER@194.182.83.34:~/trading-bot
```

## 3. Prepare The Server

On the VPS, run the setup script already included in this repo:

```bash
cd ~/trading-bot
chmod +x deploy_setup.sh
./deploy_setup.sh
```

The script will:

- install system packages
- create `venv`
- install `requirements.txt`
- create `.env` from `.env.example` if needed
- install and enable the `trading-bot` systemd service

## 4. Configure Secrets And Trading Settings

Edit `~/trading-bot/.env` on the VPS and set at least:

```bash
METAAPI_TOKEN=your_metaapi_token_here
METAAPI_ACCOUNT_ID=your_account_id_here
```

If you have no MetaAPI account yet, create one at [MetaAPI](https://metaapi.cloud/), then add a broker demo account there and copy the token and account ID into `.env`.

Optional but useful values are already defined in `.env.example`:

```bash
SYMBOL=XAUUSD
TIMEFRAME=1h
VOLUME=0.01
MODEL_PATH=train/ppo_xauusd_latest.zip
MAX_RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05
MAX_POSITIONS=3
```

## 5. Start The Bot

```bash
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```

If the service is healthy, you should see `active (running)`.

## 6. Manage The Bot

```bash
sudo systemctl stop trading-bot
sudo systemctl start trading-bot
sudo systemctl restart trading-bot
sudo journalctl -u trading-bot -f
sudo journalctl -u trading-bot -n 100
```

## 7. Updating Code Later

After you change the code locally, upload the repo again and restart the service:

```bash
scp -r tradingbot YOUR_USER@194.182.83.34:~/trading-bot
ssh YOUR_USER@194.182.83.34
cd ~/trading-bot
sudo systemctl restart trading-bot
```

## 8. What The Service Runs

The service starts:

```bash
~/trading-bot/venv/bin/python ~/trading-bot/live_trade_metaapi.py
```

The service file is included as `trading-bot.service`, and the installer rewrites it for the current user and project path.

## 9. Notes For This VPS Size

The 1 GB RAM VPS should be fine for this bot because the live loop is lightweight:

- it loads one model
- fetches market data periodically
- sends occasional MetaAPI requests
- does not need a GUI or MT5 terminal

If you later want MT5 desktop trading, use a Windows machine or a different remote setup. For this VPS, MetaAPI is the correct path.

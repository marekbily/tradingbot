# tmux Quick Reference

A compact cheat-sheet for starting persistent training sessions with `tmux`, rescuing running processes with `reptyr`, and common commands to reconnect, monitor, and control jobs.

## Install

- Ubuntu (root or sudo):

```bash
apt-get update
apt-get install -y tmux reptyr
```

- Termux (Android):

```bash
pkg install tmux
# reptyr may not be available on Termux
```

## Start a session (interactive)

```bash
cd /workspace/tradingbot
tmux new -s training
# inside tmux:
source .venv/bin/activate
python train/train_ultimate_150.py --steps 1000000 --device cuda --batch-size 32 |& tee train/run_training.log
# detach: Ctrl-B then D
```

## Start a detached session (one-liner)

```bash
cd /workspace/tradingbot
tmux new -s training -d bash -lc "source .venv/bin/activate && python train/train_ultimate_150.py --steps 1000000 --device cuda --batch-size 32 > train/run_training.log 2>&1 & echo \$! > train/train.pid"
```

## List / attach / detach

- List sessions: `tmux ls`
- Attach: `tmux attach -t training` or `tmux a -t training`
- Detach from inside tmux: `Ctrl-B` then `D`
- Kill session: `tmux kill-session -t training`

## Rescue a running process (move into tmux)

1. Find PID:

```bash
pgrep -a -f 'train/train_ultimate_150.py' || ps aux | grep '[p]ython.*train_ultimate_150.py'
```

2. Create detached tmux and reparent with `reptyr`:

```bash
tmux new -s training -d
PID=<the-pid>
reptyr -T $PID    # try -T, then plain reptyr $PID if it fails
tmux attach -t training
```

Notes: `reptyr` may require root privileges or special ptrace settings on some hosts. If `reptyr` is not available, the reliable fallback is to stop and restart under tmux.

## Logs, PID, and basic monitoring

- Save PID when starting: `echo $! > train/train.pid`
- Show PID: `cat train/train.pid`
- Tail logs: `tail -n 200 train/run_training.log`
- Follow logs: `tail -f train/run_training.log` or `less +F train/run_training.log`
- GPU monitor: `watch -n 5 nvidia-smi`

## Stopping the job

- Graceful: `kill $(cat train/train.pid)` or `pkill -f 'train/train_ultimate_150.py'`
- Force: `pkill -9 -f 'train/train_ultimate_150.py'`

## TensorBoard (optional)

Start in tmux so it remains available after logout:

```bash
source .venv/bin/activate
tensorboard --logdir train/tensorboard --bind_all --port 6006 > train/tensorboard.log 2>&1 & echo $! > train/tensorboard.pid
```

Forward port locally to view in your browser:

```bash
ssh -L 6006:localhost:6006 -i <key> <user>@<host>
# then open http://localhost:6006
```

## Quick tips

- Use `|& tee` to both view and write logs from inside tmux.
- Prefer `tmux` for persistent sessions rather than `nohup` if you want interactive inspection later.
- Keep `train/train.pid` and `train/run_training.log` in the repo workspace for easy inspection.

---

File: `docs/tmux_quick_reference.md`

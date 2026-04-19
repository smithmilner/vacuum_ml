# Remote GPU Training

## Architecture

- **MacBook** — control interface, Claude Code, code editing
- **Windows Gaming PC** — RTX 4070 Ti SUPER (16GB VRAM), runs all training
- **WSL2 (Ubuntu)** — Linux environment on the PC where code actually runs
- **Tailscale** — private mesh VPN connecting both machines

```
MacBook → SSH (Tailscale) → Windows → WSL2 (Ubuntu) → GPU
```

## Connecting

```bash
ssh <windows-username>@<tailscale-ip>
# Lands directly in WSL shell (ForceCommand wsl configured on server)
```

## First-Time Setup (WSL)

```bash
cd ~/workspace
git clone <repo-url> vacuum-ml
cd vacuum-ml

pip install uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Running Training

Always use tmux so training survives SSH disconnects:

```bash
tmux new -s train
.venv/bin/python -m vacuum_ml.training.train --timesteps 5000000 --envs 16
# Detach: Ctrl+b then d
# Reattach: tmux attach -t train
```

### Recommended envs count

SB3 auto-detects CUDA — no code changes needed. The neural network runs on GPU, but Shapely env stepping stays on CPU. More parallel envs keeps the GPU better utilized:

| Setting | When to use |
|---|---|
| `--envs 8` | Quick experiments |
| `--envs 16` | Standard training runs |
| `--envs 32` | If GPU utilization is low (check via `nvidia-smi`) |

Monitor GPU utilization during training:
```bash
watch -n 1 nvidia-smi
```

## Workflow

1. Edit code on MacBook (Claude Code)
2. `git push` from MacBook
3. `git pull` on remote PC (in tmux)
4. Launch training in tmux, detach
5. Check progress by SSH-ing back in and running `tmux attach -t train`

## Notes

- Work in `~/workspace`, not `/mnt/c/...` (much faster I/O)
- Use cloud GPUs only if 16GB VRAM becomes a bottleneck or parallelism is needed
- `caffeinate` not needed on remote — tmux keeps the session alive regardless

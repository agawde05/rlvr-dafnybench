# RLVR-DafnyBench: Reinforcement Learning from Verifiable Rewards on DafnyBench

A **reproducible RL training and evaluation pipeline** for the **DafnyBench** formal verification benchmark, built with **uv** and **Docker**.

---

## 🚀 Quick Start

```bash
git clone https://github.com/<your-org-or-user>/rlvr-dafnybench.git
cd rlvr-dafnybench
code .
```

When prompted by VS Code, click **“Reopen in Container.”**

Then run:

```bash
uv sync
uv run scripts/get_data.py
uv run scripts/train_model.py
uv run scripts/evaluate_model.py
uv run pytest
```

---

## 🧭 Dev Container Setup

Requires:

* **Docker Desktop** (with WSL2 enabled on Windows)
* **VS Code** + Dev Container extension
* **uv** and **ruff**

### Install Prerequisites

#### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install ruff@latest
```

#### Windows (PowerShell)

```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv tool install ruff@latest
```

Ensure:

```bash
docker --version
uv --version
ruff --version
```

---

## 🗂️ Project Structure

| Path                        | Description                            |
| --------------------------- | -------------------------------------- |
| `src/rlvr_dafnybench/`      | Core RL + verification code            |
| `scripts/`                  | Training, evaluation, and data scripts |
| `data/DafnyBench/`          | Dafny benchmark data                   |
| `models/`                   | Saved checkpoints                      |
| `experiments/`              | Experiment logs                        |
| `config/train_config.yaml`  | Training configuration                 |
| `pyproject.toml`, `uv.lock` | Dependency management                  |

---

## 🧩 Requirements

| Component | Version                                |
| --------- | -------------------------------------- |
| Python    | ≥ 3.12                                 |
| Docker    | ≥ 24.0                                 |
| VS Code   | ≥ 1.85                                 |
| uv        | ≥ 0.4.0                                |
| Ruff      | ≥ 0.14.0                               |
| OS        | Ubuntu 22.04+ / macOS / Windows (WSL2) |

---

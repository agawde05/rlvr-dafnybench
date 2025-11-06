# RLVR-DafnyBench: Reinforcement Learning from Verifiable Rewards on DafnyBench

This repository provides a **reproducible RL training and evaluation pipeline** for the **DafnyBench** formal verification benchmark, built around **uv** and **Docker**.

It is designed for **reproducibility**, **collaboration**, and **cross-platform development**.  
Supports both:
- **VS Code Dev Container workflow (recommended)** - for automatic, consistent environments  
- **Local Development (not recommended)** - for debugging 
---

## 📘 Table of Contents
1. [🔧 Initial Setup](#-initial-setup)
   - [Install Docker](#install-docker)
   - [Install VS Code + Extensions](#install-vs-code--extensions)
   - [Install uv and Ruff](#install-uv-and-ruff)
2. [🧭 Dev Container Workflow](#-dev-container-workflow)
3. [⚙️ Local Development](#-local-development)
4. [🧹 Cleanup & Troubleshooting](#-cleanup--troubleshooting)
5. [🗂️ Project Structure](#-project-structure)
6. [💡 Development Tips](#-development-tips)
7. [🧩 System Requirements](#-system-requirements)
8. [🚀 Quick Start](#-quick-start)

---

## 🔧 Initial Setup

### Install Docker

#### 🐧 Linux / WSL2 (Ubuntu)
```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

#### 🍎 macOS

Download **Docker Desktop for Mac** from
[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

Verify installation:

```bash
docker --version
docker compose version
```

#### 🪟 Windows (with WSL2)

1. Install **Docker Desktop for Windows** from
   [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. In **Docker Desktop**, open **Settings → General**, ensure **Use the WSL 2 based engine** is checked.
   Then go to **Settings → Resources → WSL Integration**, and enable **Ubuntu (or your preferred WSL2 distro)**.
3. Verify setup:

   ```powershell
   docker --version
   docker compose version
   wsl --status
   ```

---

### Install VS Code + Extensions

Install **[VS Code](https://code.visualstudio.com/)** and these extensions:

* **Docker** (`ms-azuretools.vscode-docker`)
* **Container Tools** (`ms-azuretools.vscode-containers`)
* **Dev Containers** (`ms-vscode-remote.remote-containers`)

> On WSL2, VS Code may prompt to install **Remote - WSL**; accept it.

---

### Install uv and Ruff

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install ruff@latest
```

Windows (PowerShell):

```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv tool install ruff@latest
```

Check versions:

```bash
uv --version
ruff --version
```

---

## 🧭 Dev Container Workflow

The main workflow is inside the **VS Code Dev Container**, ensuring identical environments for all teammates.

### 1. Prerequisites

Install:

* **Docker Desktop** (Linux/macOS/Windows WSL2)
* **VS Code** with extensions above

Verify:

```bash
docker --version
docker compose version
```

### 2. Open the Project

```bash
git clone https://github.com/agawde05/rlvr-dafnybench.git
cd rlvr-dafnybench
code .
```

When prompted:

> "Reopen in Container"

Click **Yes** - VS Code will:

* Build the Docker image
* Start the container with all dependencies
* Mount your repo at `/workspace`

You’ll see a shell:

```bash
(rlvr-dafnybench) root@<container-id>:/workspace#
```

### 3. Run the Pipeline

```bash
uv sync
uv run scripts/get_data.py
uv run scripts/train_model.py
uv run scripts/evaluate_model.py
uv run pytest
```

This ensures reproducibility with no local dependency issues.

---

## ⚙️ Local Development

### Local Environment Setup

```bash
echo "PYTHONPATH=.:./src" >> .env
echo "DAFNY_PATH=/opt/dafny/dafny" >> .env
echo "HF_HOME=./.cache/huggingface" >> .env
export $(grep -v '^#' .env | xargs)
```

PowerShell alternative:

```powershell
Get-Content .env | ForEach-Object { if ($_ -match '([^=]+)=(.*)') { setx $matches[1] $matches[2] } }
```

### Running Locally

```bash
uv pip install -e .
uv run pytest -q
uv run scripts/get_data.py
uv run scripts/train_model.py
uv run scripts/evaluate_model.py
```

---

## 🧹 Cleanup & Troubleshooting

```bash
docker compose -f docker-compose.dev.yml down -v
docker system prune -f
docker volume prune -f
rm -rf .pytest_cache .ruff_cache .uv __pycache__ */__pycache__
```

Clear generated data:

```bash
rm -rf data/DafnyBench models/* experiments/* results/*
```

If needed:

```bash
sudo rm -rf .pytest_cache
```

Windows:

```powershell
Remove-Item -Recurse -Force .pytest_cache,.ruff_cache
```

---

## 🗂️ Project Structure

| Path                                | Purpose                                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `README.md`                         | Full user-facing guide                                                                                |
| `docs/dev_notes.md`                 | Developer notes and collaboration details                                                             |
| `pyproject.toml`, `uv.lock`         | Dependency management                                                                                 |
| `docker/`, `docker-compose.dev.yml` | Docker configuration and compose setup                                                                |
| `config/train_config.yaml`          | Experiment and training configuration                                                                 |
| `src/rlvr_dafnybench/`              | Core RL + verification code (`training.py`, `evaluation.py`, `utils.py`, `verifier.py`)               |
| `scripts/`                          | Entrypoints for training, evaluation, and data (`get_data.py`, `train_model.py`, `evaluate_model.py`) |
| `data/DafnyBench/`                  | Dataset and benchmark directory                                                                       |
| `experiments/`                      | Experiment logs and outputs                                                                           |
| `models/`                           | Saved model checkpoints                                                                               |
| `notebooks/`                        | Analysis and visualization notebooks                                                                  |
| `tests/`                            | Unit tests                                                                                            |

---

## 💡 Development Tips

* Work **inside the container**
* Run all commands via `uv run`
* Use `ruff` + `pytest` before commits
* Don’t commit `.env`, `.venv`, or `/data`

---

## 🧩 System Requirements

| Component   | Version                                | Notes            |
| ----------- | -------------------------------------- | ---------------- |
| **Python**  | ≥ 3.12                                 | Managed by `uv`  |
| **Docker**  | ≥ 24.0                                 | Required         |
| **VS Code** | ≥ 1.85                                 | With extensions  |
| **uv**      | ≥ 0.4.0                                | Package manager  |
| **Ruff**    | ≥ 0.14.0                               | Formatter/linter |
| **OS**      | Ubuntu 22.04+ / macOS / Windows (WSL2) | All supported    |

---

## 🚀 Quick Start

```bash
git clone https://github.com/agawde05/rlvr-dafnybench.git
cd rlvr-dafnybench
code .

# Reopen in container
uv sync
uv run scripts/get_data.py
uv run scripts/train_model.py
uv run scripts/evaluate_model.py
uv run pytest
```
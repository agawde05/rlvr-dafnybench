# Project Structure (Developer Notes)

This repository implements **RL from Verifiable Rewards (RLVR)** on **DafnyBench**.

### Root Files

| Path                         | Purpose                                           |
| ---------------------------- | ------------------------------------------------- |
| `README.md`                  | User guide (how to install, run training/eval)    |
| `pyproject.toml` / `uv.lock` | Python project + dependency management (via `uv`) |

### Core Code

| Path                   | Purpose                                    |
| ---------------------- | ------------------------------------------ |
| `src/rlvr_dafnybench/` | Main Python package                        |
| `training.py`          | RL training loop (GRPO/PPO, model updates) |
| `evaluation.py`        | Evaluate models on DafnyBench              |
| `verifier.py`          | Low-level Dafny invocation + parsing       |
| `utils.py`             | Shared helpers/utilities                   |

### RL Environment

| Path                       | Purpose                                                                                 |
| -------------------------- | --------------------------------------------------------------------------------------- |
| `env/dafny_environment.py` | "Environment" wrapper for RL — steps programs through Dafny, returns rewards & metadata |

### Configs + Data

| Path                       | Purpose                                               |
| -------------------------- | ----------------------------------------------------- |
| `config/train_config.yaml` | Training hyperparameters & experiment settings        |
| `data/`                    | Stores downloaded datasets & dumps (ignored from git) |

### Scripts (Entry Points)

| File                        | Purpose                          |
| --------------------------- | -------------------------------- |
| `scripts/get_data.py`       | Download/prepare DafnyBench data |
| `scripts/train_model.py`    | Launch RL training run           |
| `scripts/evaluate_model.py` | Evaluate trained model           |
| `scripts/launch_local.sh`   | Local training entry (CPU)       |
| `scripts/launch_docker.sh`  | Development container run script |
| `scripts/launch_hpc.sh`     | HPC scheduler submit wrapper     |

### Research Workflow Folders

| Path           | Purpose                          |
| -------------- | -------------------------------- |
| `notebooks/`   | Analysis, debugging, exploration |
| `experiments/` | Logs, configs, run metadata      |
| `models/`      | Saved model checkpoints          |

### Tests

| Path                     | Purpose                       |
| ------------------------ | ----------------------------- |
| `tests/test_env.py`      | Verifier/env behavior tests   |
| `tests/test_training.py` | Training pipeline smoke tests |

---

### Development Guidelines

* Python code lives under `src/`
* Scripts call into `src/` (no logic in scripts)
* Config-driven training (edit YAML, not code)
* Env wrapper ensures Dafny logic is isolated/testable
* All heavy files (models/data) stay out of git

# Task Breakdown for Two-Stage Dafny RLVR Pipeline

## Data & Pairing
- Wire up `load_dafny_bench` to read DafnyBench (or proxy) into `DafnyExample` objects with ids/spec/header/body/annotated_body fields.
- Implement `split_examples` with reproducible seeds and stratification hooks (e.g., by length or construct type).
- Build `build_pairings_for_writer` (header/spec → body targets) and `build_pairings_for_annotator` (body → annotated_body) including validation that headers/bodies are non-empty and compilable.
- Add collators for each role that pack inputs/labels, respect tokenizer eos/pad ids, and produce attention masks.

## Prompt & Formatting
- Finalize prompt templates for writer vs annotator; add few-shot scaffolds; include stop sequences to avoid spilling past method bodies.
- Add prompt fusing utilities (e.g., insert context header) and style variants (strict vs permissive).

## Model Setup
- Choose base LM (e.g., Llama-3 or Qwen2.5 Coder) and tokenizer; expose dtype/device map knobs.
- Implement LoRA/PEFT wrapping with rank/alpha/dropout config plus optional gradient checkpointing.
- Add generation config defaults for RL sampling (temperature/top-p/min-p, max tokens, stop seqs).

## Verifier & Rewards
- Implement `run_dafny_verify` using local Dafny binary or docker, with timeout handling, temp-file plumbing, and structured logs.
- Define reward adapters: binary {0,1} primary; optional partials for syntax/parse failures, and KL-regularization bookkeeping.
- Add batched verification helper with parallelism guardrails and output caching for deterministic code strings.

## Training Loops
- Implement SFT loops for writer and annotator (optimizer/scheduler hooks; logging of loss/token counts).
- Implement GRPO: sampling `G` candidates, computing group-normalized advantages, clipped objective + KL to reference model, and ref model update cadence.
- Add curriculum/alternating training mode where annotator RL continues while writer remains frozen, with option to RL-tune writer on downstream cascade reward.

## Cascade Execution
- Implement `cascade_sample` that runs writer → annotator → verifier, returns structured `CascadeResult`, and handles failure bucketing (syntax, timeout, semantic).
- Add joint reward computation for annotator-only or writer+annotator credit assignment (self-play / assisted).

## Evaluation & Analysis
- Implement evaluation for each role: writer exact-match/bleu and functional coverage; annotator verify rate; cascade verify rate and latency.
- Add qualitative logging (n-best lists, diffs vs gold, verifier logs) and reward trajectory plots.

## Experiment Harness & Logging
- Build `run_experiment` driver that stitches configs, seeds, loaders, training/eval, and checkpointing.
- Add config schemas (TrainConfig/GenConfig) and serialization; capture system metadata (Dafny version, hardware).
- Integrate lightweight logging (stdout + JSONL/CSV) with paths for checkpoints and verifier logs.

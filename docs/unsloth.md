# unsloth backend (single GPU)

Guide for `use_unsloth: true`. It loads the model through [unsloth](https://github.com/unslothai/unsloth)'s
`FastModel` and swaps TRL's trainers for unsloth's patched versions, in all four scripts (`sft.py`, `dpo.py`,
`grpo.py`, `distill.py`). It is a **backend switch**: hyperparameters, the prepared training data and the saved
checkpoints are the same as with `use_unsloth: false`.

## Install and run

unsloth is an optional extra (pinned to `unsloth`/`unsloth-zoo` 2026.10.3):

```bash
$ uv sync --extra unsloth

$ python sft.py --config configs/SFT/qwen3-0.6B-sft-unsloth.yaml
$ python grpo.py --config configs/GRPO/qwen3-0.6B-grpo-unsloth.yaml

# Or switch it on for any config
$ python dpo.py --config configs/DPO/qwen3-0.6B-dpo.yaml --use_unsloth true --use_peft true --lora_dropout 0.0
$ python distill.py --use_unsloth true --use_peft true --lora_dropout 0.0
```

unsloth officially caps `datasets`, `transformers` and `trl` below this repo's pins. `pyproject.toml` keeps the
core pins with a `[tool.uv] override-dependencies` entry (so `uv sync --extra unsloth` changes none of them and
`uv sync` without the extra is unaffected); the combination was validated on GPU with all four scripts. When you
bump `datasets`, `transformers` or `trl` in the core dependencies, bump the override with them.

Run the script directly with `python`. Multi-process `accelerate launch` and DeepSpeed are refused with an
error: unsloth's patches are single-GPU here.

## Supported models

The switch is matched on the checkpoint's `config.json` `model_type`:

| `model_type` | Checkpoints | vLLM generation (GRPO / distillation) |
|---|---|---|
| `qwen3` | Qwen3 dense (`dnotitia/Qwen3-*`, `Qwen/Qwen3-*`) | unsloth's engine |
| `qwen3_moe` | Qwen3-30B-A3B, Qwen3-235B-A22B | HF generate (slower) |
| `qwen3_5` | Qwen3.5 dense, Qwen3.6-27B, Qwen3.8-27B | unsloth's engine |
| `qwen3_5_moe` | Qwen3.5-35B-A3B / 122B-A10B / 397B-A17B, Qwen3.6-35B-A3B | HF generate (slower) |

Any other model type (e.g. Qwen3.8-Flash-Next, which is `qwen4_exp`) logs a warning and runs the standard
path; unsloth is then not even imported. MoE models generate with HF generate because unsloth's vLLM engine
cannot serve LoRA on the expert weights, which unsloth always adapts on MoE models
([vllm-project/vllm#41754](https://github.com/vllm-project/vllm/issues/41754)).

## When it helps

Measured on one B200 against this repo's defaults (FlashAttention-2, Liger kernels for SFT/DPO):

| Workload (one B200) | `use_unsloth: false` | `use_unsloth: true` | Step time | Peak memory |
|---|---|---|---|---|
| SFT LoRA, Qwen3-8B, 4k, batch 2 | 0.82 s/step, 20.2 GiB | 0.55 s/step, 17.8 GiB | **33% faster** | **−12%** |
| SFT LoRA, Qwen3-8B, 16k | 10.25 s/step, 24.8 GiB | 7.92 s/step, 19.8 GiB | **23% faster** | **−20%** |
| SFT QLoRA (4-bit), Qwen3-8B, 4k, batch 2 | 0.73 s/step, 10.1 GiB | 0.50 s/step, 8.9 GiB | **32% faster** | **−12%** |
| SFT LoRA, Qwen3-30B-A3B (MoE), attention + experts, 2k | 0.44 s/step, 63.7 GiB ¹ | 0.33 s/step, 62.3 GiB | **25% faster** | −2% |
| DPO LoRA, Qwen3-8B, 2k | 2.75 s/step, 34.5 GiB ³ | 2.50 s/step, 17.5 GiB | about the same | **−49%** |
| SFT full fine-tuning, Qwen3-4B, 4k | 0.24 s/step, 31.0 GiB | 0.24 s/step, 30.9 GiB | same | same |
| GRPO LoRA + vLLM, Qwen3-4B, 8×1024 tokens | 4.14 s/step, 72.8 GiB ² | 7.14 s/step, 64.9 GiB ² | 72% slower | −11% |
| Distillation LoRA + vLLM, Qwen3-0.6B ← 1.7B | 0.78 s/step, 49.5 GiB ² | 0.89 s/step, 48.2 GiB ² | 14% slower | −3% |

¹ The default config crashes on this model (Liger's MoE SwiGLU kernel and LoRA on the expert weights don't mix),
so this baseline runs with `use_liger_kernel: false`; with unsloth the default config just works.
² Includes the colocated vLLM engine (`vllm_gpu_memory_utilization` 0.35 for GRPO, 0.25 for distillation).
³ The standard path loads a second full model as the DPO reference even with LoRA; the unsloth path uses the
policy with its fresh adapter disabled (the same weights), which is most of the saving.

Steady-state step time (first three steps excluded), peak allocated memory, single runs; `lora_r: 16` on all
linear layers (`lora_r: 8` for the MoE), `lora_dropout: 0.0`, gradient checkpointing on.

Why GRPO and distillation are slower (per-phase timing of the same two workloads; both sides generated the same
number of tokens): unsloth's vLLM engine shares the weights with the trained model and decodes with the LoRA
adapter applied, while the standard path merges the adapter into the weights before every generation. Decoding
runs 32% (GRPO) and 42% (distillation) fewer tokens/s, and that is all of distillation's slowdown. On GRPO it is
half of it (+1.4 s/step); the other half is the training phase, whose backward pass takes 1.5 s instead of 0.4 s.

So: turn it on for **LoRA / QLoRA fine-tuning** (SFT, DPO) and **MoE LoRA**, and when **memory** is what limits
you. Full fine-tuning gains nothing. For **GRPO and distillation** it saves memory but runs slower, so leave it
off when throughput matters.

## What it changes, and what it keeps

**Import order.** unsloth patches `trl`/`transformers`/`peft` when it is imported, and the Dnotitia trainers
subclass TRL's trainers when they are defined. So `import unsloth` must run before any of them: a small
bootstrap at the top of each entry script (`dna_factory/unsloth_bootstrap.py`) reads `use_unsloth` and the
model's `model_type` from the CLI/YAML and imports unsloth only when both say so. This is also why the switch
only works when the script is run directly.

**Model and LoRA.** The model is loaded by `FastModel.from_pretrained` with the requested repo, revision, dtype
and attention implementation (`dna_factory/unsloth_integration.py`). unsloth's patched trainers ignore
`peft_config`, so LoRA is applied up front through `FastModel.get_peft_model` from the usual `lora_*` options.
An unset `lora_target_modules` follows PEFT's per-architecture default as on the standard path (`q_proj`,
`v_proj` for Qwen3); architectures PEFT has no default for (Qwen3 MoE, Qwen3.5) get unsloth's: every linear
layer of the language model, plus the MoE expert parameters. The script's own tokenizer is used.

**Hyperparameters.** unsloth's patched TRL config classes rewrite a few values when they are built (for
example GRPO with `loss_type: dapo` gets `epsilon_high: 0.28`, and `top_k: -1`). The scripts re-parse the
arguments with TRL's own classes, restore every rewritten value and print it:

```
[use_unsloth] restored 2 config value(s) rewritten by unsloth: top_k: -1 -> 0, epsilon_high: 0.28 -> None
```

The one value kept is `gradient_checkpointing_kwargs: {use_reentrant: true}`, which unsloth's gradient
checkpointing needs. `use_liger_kernel` is turned off with a warning (unsloth ships its own fused kernels and
Liger would patch the model twice).

**SFT data.** unsloth's SFT trainer prepares datasets itself and has no support for conversational `messages`
datasets, so the integration tokenizes them first exactly as TRL does (same chat template, same assistant
masks). Checked token for token on `trl-lib/Capybara`: the same sequences, tokens and trained tokens as the
standard path. Two differences remain:

- Packing groups sequences slightly differently (unsloth packs the whole dataset at once rather than per
  `dataset_num_proc` shard): 0.25% fewer packed rows on Capybara, same tokens trained.
- unsloth does not pack Qwen3.5-architecture (hybrid linear-attention) models; the integration warns that the
  run trains on unpacked rows, so it has more steps per epoch.

**Checkpoints.** Saved checkpoints are ordinary PEFT adapters (LoRA) or Hugging Face checkpoints (full
fine-tuning) and load without unsloth. Qwen3.5-architecture checkpoints keep their
`*ForConditionalGeneration` architecture, as on the standard path.

## Not supported with unsloth

- **AsyncGRPO** (`grpo_execution: async`): rejected with an error. It uses TRL's experimental
  `AsyncGRPOTrainer` and a separate `vllm serve` process, neither of which unsloth patches; use
  `grpo_execution: sync` for unsloth. The error is raised even when the model type would make unsloth fall back
  to the standard path.
- **DeepSpeed / multi-process launches**: see above.

## Checkpoint evaluation

`eval_on_checkpoint: true` works unchanged. It only reads the checkpoints the trainer writes and serves them with
`vllm serve` on `eval_devices` in separate processes, so unsloth's engine and patches are not involved; the
checkpoints are ordinary PEFT/HF checkpoints (see above). Keep `eval_devices` off the GPU training uses, as always.
As on the standard path, a LoRA run saves adapter weights only; merge the adapter first if the eval server needs
full weights.

## Per-trainer notes

- **SFT**: see above. Resuming from a checkpoint works as usual.
- **DPO**: with LoRA the reference model is the policy with its adapter disabled at initialisation (TRL's
  "ref" adapter); with full fine-tuning a separate plain Hugging Face reference model is loaded, as TRL does.
- **GRPO**: with `use_vllm: true` unsloth's own vLLM engine is used (`fast_inference`): it is colocated, shares
  the weights with the trained model, and is sized by `vllm_gpu_memory_utilization`. It needs LoRA: full
  fine-tuning falls back to HF generate with a warning, as do model types without engine support (see the
  table). `vllm_mode: server` is not supported. `dynamic_sampling` (`mask` / `resample`) works.
- **Distillation**: only the student is loaded through unsloth; the teacher stays a plain Hugging Face model.
  Generation uses unsloth's engine as for GRPO; the student's LoRA weights reach it before every generation
  (checked against HF generate on the live student).

## Errors you may hit

These combinations are rejected before the model is loaded, rather than silently changed:

| Setting | Why | Fix |
|---|---|---|
| `lora_dropout > 0` on `qwen3_moe` / `qwen3_5_moe` | unsloth adapts the MoE experts through PEFT's `ParamWrapper`, which has no dropout | `lora_dropout: 0.0` |
| GRPO on a MoE model with a non-zero `router_aux_loss_coef` (by default taken from the model config: `0.001` for Qwen3 MoE) | unsloth's GRPO step does not compute the router auxiliary loss | `router_aux_loss_coef: 0.0` |
| `use_dora: true` | unsloth has no DoRA kernels | `use_dora: false` |
| `load_in_4bit` / `load_in_8bit` without `use_peft` | quantized weights can't be fully fine-tuned | `use_peft: true` |
| `model_revision` other than `main` with unsloth's vLLM engine | the engine always loads the default branch | drop the revision, or `use_vllm: false` |
| `vllm_mode: server` | unsloth manages the engine in-process | `vllm_mode: colocate` |

`lora_dropout` above 0 is otherwise allowed, but unsloth's fast LoRA path needs 0 (TRL's default is `0.05`),
so the example configs set `lora_dropout: 0.0`. unsloth also advises against 4-bit QLoRA on Qwen3.5-architecture
models (a warning is printed).

## Troubleshooting

- **`/tmp` mounted `noexec`** (as on our cluster): torch.compile caches kernels under
  `/tmp/torchinductor_<user>` by default, and the compiled `.so` files can't be loaded from there. unsloth then
  runs the affected modules eagerly (`Unsloth: torch.compile could not generate code ...; running it eagerly`)
  or fails with `failed to map segment from shared object`. The bootstrap warns about this. Point the cache at
  an exec-capable filesystem:

  ```bash
  $ export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor  # any directory on an exec-capable filesystem
  ```

  (unsloth clears this variable when it is imported; the bootstrap puts it back.)
- The first steps are slow while unsloth compiles kernels; the cache makes later runs start faster.
- unsloth writes `unsloth_compiled_cache/` and, with its vLLM engine, `vllm_gen_lora_<gpu>/` into the working
  directory. Both are git-ignored.
- Two runs started at the same time on one machine can collide on `MASTER_PORT` (29500); set a different
  `MASTER_PORT` per run.

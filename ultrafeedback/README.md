# SPO LLM experiments (UltraFeedback + Skywork-Reward-V2)

This codebase implements preference-optimization experiments for the SPO paper using UltraFeedback, Skywork-Reward-V2 as a ground-truth reward model, and KL-regularized policy evaluation. It supports DPO, PSPO, OSPO, and RSPO and evaluates reward/KL tradeoffs by scanning a beta grid.

Default model choices: Qwen3-0.6B for the base/reference model and Skywork-Reward-V2-Qwen3-1.7B for rewards.

## Setup
- Python 3.10+
- CUDA GPUs (tested on A100)
- Key deps: `torch`, `transformers`, `datasets`, `peft`, `pandas`, `matplotlib`

Install dependencies (example):
```bash
pip install torch transformers datasets peft pandas matplotlib
```

## Pipeline Overview
1. (Optional) SFT a reference model on UltraFeedback: `scripts/prepare_reference.py`
2. Generate preference data:
   - Sample prompts from UltraFeedback
   - Generate two responses from the reference model
   - Score with the reward model
   - Normalize reward differences to variance 1
   - Sample preferences using the shifted-logistic-mixture link
3. Train preference optimizers (DPO/PSPO/OSPO/RSPO) with shared preference data
4. Evaluate each trained policy by scanning beta and computing reward/KL
5. Analyze CSV results to produce Pareto and convergence plots

## Key Scripts
- `scripts/prepare_reference.py`: light SFT for the reference model
- `scripts/generate_preferences.py`: build preference JSONL
- `scripts/train_policy.py`: train a policy with a given optimizer
- `scripts/evaluate_policy.py`: beta-scan eval and CSV output
- `scripts/run_grid.py`: run the full grid with parallel GPU workers
- `scripts/analyze_results.py`: plots + summary tables

## Example: Small Smoke Test
```bash
python scripts/generate_preferences.py \
  --dataset_id openbmb/UltraFeedback \
  --num_samples 200 \
  --model_id Qwen/Qwen3-0.6B \
  --reward_model_id Skywork/Skywork-Reward-V2-Qwen3-1.7B \
  --output data/preferences/prefs_small.jsonl

python scripts/train_policy.py \
  --pref_path data/preferences/prefs_small.jsonl \
  --method dpo \
  --base_model_id Qwen/Qwen3-0.6B \
  --output_dir outputs/smoke_dpo \
  --max_steps 50

python scripts/evaluate_policy.py \
  --model_id outputs/smoke_dpo/dpo_final \
  --ref_model_id Qwen/Qwen3-0.6B \
  --reward_model_id Skywork/Skywork-Reward-V2-Qwen3-1.7B \
  --output outputs/eval/smoke_dpo.csv
```

## Parallel Grid Runs
```bash
python scripts/run_grid.py \
  --seeds 123,456,789,1011 \
  --link_shifts 0.0,0.5,1.0,2.0 \
  --n_list 5000 \
  --methods dpo,pspo,ospo,rspo \
  --gpus 0,1,2,3,4,5,6,7
```

Useful options:
- `--max_prompt_length` / `--max_response_length` for memory control
- `--gen_max_new_tokens` / `--eval_max_new_tokens` for faster runs
- `--eval_samples` / `--eval_batch_size` to trade off speed vs variance

## Output
Each evaluation CSV row records:
- base model id, reward model id
- link parameters (shift, temperature)
- random seed, optimizer, preference data size
- beta, average reward, KL divergence to reference

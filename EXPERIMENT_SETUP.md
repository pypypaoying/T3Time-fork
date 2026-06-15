# T3Time Baseline Setup

## Environment

```bash
conda env create -f environment_linux.yml
conda activate t3time
```

The reproducible environment definition is in `environment_linux.yml`.

## Datasets

The official repository already contains ETT, Weather, ILI, and Exchange.
Download the missing ECL and Traffic files from the THUML dataset mirror:

```bash
bash scripts/download_datasets.sh
```

All requested datasets then live under `dataset/`:

- `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`
- `Weather.csv`, `ECL.csv`, `Traffic.csv`
- `ILI.csv`, `exchange_rate.csv`

ETT loaders use the fixed 60/20/20 split from the original code. Other
datasets use 70/10/20.

## Prompt Embeddings

T3Time requires GPT-2 last-token embeddings before real training. Embeddings
are keyed by input length, so the following cache is for `seq_len=336` and can
be reused for all Weather horizons. Generate using the shortest horizon to
cover the largest number of windows:

```bash
for split in train val test; do
  python storage/store_emb.py \
    --data_path Weather \
    --root_path ./dataset \
    --embed_root ./Embeddings \
    --input_len 336 \
    --output_len 96 \
    --divide "$split" \
    --batch_size 1 \
    --prompt_batch_size 8 \
    --device cuda
done
```

For ILI, use `--output_len 24`. For the other datasets, use `96`.

## Timing Benchmark

`embedding_mode=zeros` is timing-only. It preserves the model tensor shapes
and forward/backward cost without waiting for GPT-2 preprocessing.

```bash
DEVICE=cuda BENCHMARK_STEPS=100 BENCHMARK_VAL_STEPS=30 \
  bash scripts/benchmark_weather_336.sh
```

Use `DEVICE=cpu` when no GPU is available. Set both benchmark step variables
to `0` to run complete training and validation loaders.

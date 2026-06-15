#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_dir="${DATASET_DIR:-${repo_root}/dataset}"
base_url="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"

mkdir -p "$dataset_dir"

download() {
  local remote_path="$1"
  local local_name="$2"
  local destination="${dataset_dir}/${local_name}"

  if [[ -s "$destination" ]]; then
    echo "Found ${local_name}; skipping."
    return
  fi

  echo "Downloading ${local_name}..."
  curl -L --fail --retry 3 \
    "${base_url}/${remote_path}?download=true" \
    -o "${destination}.part"
  mv "${destination}.part" "$destination"
}

# These two large benchmark files are not included in the official T3Time repo.
download "electricity/electricity.csv" "ECL.csv"
download "traffic/traffic.csv" "Traffic.csv"

echo "Datasets are ready under ${dataset_dir}."

import torch
import sys
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from gen_prompt_emb import GenPromptEmb


def embedding_file_ready(file_path, d_model, num_nodes):
    if not os.path.isfile(file_path):
        return False
    try:
        with h5py.File(file_path, "r") as hf:
            return hf["embeddings"].shape == (d_model, num_nodes)
    except (OSError, KeyError):
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prompt_batch_size", type=int, default=8)
    parser.add_argument("--root_path", type=str, default=os.path.join(PROJECT_ROOT, "dataset"))
    parser.add_argument("--embed_root", type=str, default=os.path.join(PROJECT_ROOT, "Embeddings"))
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def get_dataset(data_path, flag, input_len, output_len, root_path):
    datasets = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(
        root_path=root_path, flag=flag, size=[input_len, 0, output_len], data_path=data_path
    )

def save_embeddings(args):
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be greater than 0")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(
        args.data_path, args.divide, args.input_len, args.output_len, args.root_path
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    print(
        f"Preparing {args.data_path}/{args.divide}: {len(dataset)} samples, "
        f"device={device}, prompt_batch_size={args.prompt_batch_size}",
        flush=True,
    )

    gen_prompt_emb = GenPromptEmb(
        device=device, # type: ignore
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide,
        prompt_batch_size=args.prompt_batch_size,
    ).to(device)

    save_path = os.path.join(
        args.embed_root, dataset.data_path_file, f"seq{args.input_len}", args.divide
    )
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    print(f"Saving embeddings to {save_path}", flush=True)
    sample_offset = 0
    completed = 0
    generated = 0
    started_at = time.perf_counter()
    for x, y, x_mark, y_mark in data_loader:
        batch_indices = list(range(sample_offset, sample_offset + len(x)))
        output_paths = [
            os.path.join(save_path, f"{sample_index}.h5")
            for sample_index in batch_indices
        ]

        if not args.overwrite and all(
            embedding_file_ready(path, args.d_model, x.shape[2])
            for path in output_paths
        ):
            sample_offset += len(x)
            completed += len(x)
            continue

        # Prompt construction stays on CPU. Only tokenized GPT-2 inputs are
        # transferred to the selected device inside GenPromptEmb.
        embeddings = gen_prompt_emb.generate_embeddings(x, x_mark)

        for batch_index, embedding in enumerate(embeddings):
            file_path = output_paths[batch_index]
            if not args.overwrite and embedding_file_ready(
                file_path, args.d_model, x.shape[2]
            ):
                continue
            temporary_path = f"{file_path}.tmp"
            with h5py.File(temporary_path, 'w') as hf:
                hf.create_dataset('embeddings', data=embedding.numpy())
            os.replace(temporary_path, file_path)
        sample_offset += len(embeddings)
        completed += len(embeddings)
        generated += len(embeddings)

        if completed % args.log_interval < len(embeddings) or completed == len(dataset):
            elapsed = time.perf_counter() - started_at
            rate = generated / elapsed
            remaining = (len(dataset) - completed) / rate if rate else 0
            print(
                f"[{args.divide}] {completed}/{len(dataset)} "
                f"({100 * completed / len(dataset):.1f}%), "
                f"generated={generated}, {rate:.2f} new samples/s, "
                f"ETA {remaining / 60:.1f} min",
                flush=True,
            )

        # # Save and visualize the first sample
        # if i >= 0:
        #     break

    elapsed = time.perf_counter() - started_at
    print(
        f"Finished {args.data_path}/{args.divide}: "
        f"{completed}/{len(dataset)} files ready, "
        f"{generated} generated in this run, elapsed {elapsed / 60:.2f} min",
        flush=True,
    )
    
if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")

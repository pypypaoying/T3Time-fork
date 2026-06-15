import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.T3Time import TriModal
from utils.metrics import MSE, MAE, metric
import faulthandler
faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--root_path", type=str, default="./dataset", help="dataset directory")
    parser.add_argument("--embed_root", type=str, default="./Embeddings", help="embedding directory")
    parser.add_argument(
        "--embedding_mode",
        choices=["precomputed", "zeros"],
        default="precomputed",
        help="Use zeros only for timing benchmarks; real experiments require precomputed GPT-2 embeddings.",
    )
    parser.add_argument("--data_path", type=str, default="ETTm1", help="data path")
    parser.add_argument("--channel", type=int, default=32, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--epochs", type=int, default=150, help="")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument("--es_patience", type=int, default=25, help="quit if no improvement after this many iterations")
    parser.add_argument("--max_train_steps", type=int, default=0, help="0 runs the full training loader")
    parser.add_argument("--max_val_steps", type=int, default=0, help="0 runs the full validation loader")
    parser.add_argument("--skip_test", action="store_true", help="skip final test evaluation")
    parser.add_argument("--save", type=str, default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-", help="save path")
    return parser.parse_args()

class trainer:
    def __init__(
        self,
        scaler,
        channel,
        num_nodes,
        seq_len,
        pred_len,
        dropout_n,
        d_llm,
        e_layer,
        d_layer,
        head,
        lrate,
        wdecay,
        device,
        epochs
    ):
        self.model = TriModal(
            device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len, 
            dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, d_layer=d_layer, head=head
        )
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 50), eta_min=1e-6)
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))


    def train(self, input, mark, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input, mark, embeddings)
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.MAE(predict, real)
        return loss.item(), mae.item()
    
    def eval(self, input, mark, embeddings, real_val):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input,mark, embeddings)
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()

def load_data(args):
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    data_class = data_map.get(args.data_path, Dataset_Custom)
    dataset_args = dict(
        root_path=args.root_path,
        embed_root=args.embed_root,
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        embedding_mode=args.embedding_mode,
        d_llm=args.d_llm,
    )
    train_set = data_class(flag='train', **dataset_args)
    val_set = data_class(flag='val', **dataset_args)
    test_set = data_class(flag='test', **dataset_args)

    scaler = train_set.scaler

    loader_args = dict(
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_set, **loader_args)
    val_loader = DataLoader(val_set, **loader_args)
    test_loader = DataLoader(test_set, **loader_args)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    train_set, val_set, test_set, train_loader, val_loader, test_loader,scaler = load_data(args)

    print()
    seed_it(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    loss = 9999999
    test_log = 999999
    epochs_since_best_mse = 0
    bestid = 0

    path = os.path.join(args.save, args.data_path, 
                        f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.d_layer}_{args.learning_rate}_{args.dropout_n}_{args.seed}/")
    if not os.path.exists(path):
        os.makedirs(path)
     
    his_loss = []
    val_time = []
    train_time = []
    projected_val_time = []
    projected_train_time = []
    print(args)

    engine = trainer(
        scaler=scaler,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        epochs=args.epochs
    )

    print("Start training...", flush=True)

    for i in range(1, args.epochs + 1):

        synchronize(device)
        t1 = time.perf_counter()
        train_loss = []
        train_mae = []

        for iter, (x,y,x_mark,y_mark, embeddings) in enumerate(train_loader):
            trainx = x.to(device=device, dtype=torch.float32, non_blocking=True) # [B, L, N]
            trainy = y.to(device=device, dtype=torch.float32, non_blocking=True)
            trainx_mark = x_mark.to(device=device, dtype=torch.float32, non_blocking=True)
            train_embedding = embeddings.to(device=device, dtype=torch.float32, non_blocking=True)
            metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            if args.max_train_steps and iter + 1 >= args.max_train_steps:
                break

        synchronize(device)
        t2 = time.perf_counter()
        train_steps = len(train_loss)
        train_projection = (t2 - t1) * len(train_loader) / train_steps
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        if train_steps < len(train_loader):
            print(
                "Projected full training epoch: {:.4f} secs "
                "({:d}/{:d} batches measured)".format(
                    train_projection, train_steps, len(train_loader)
                )
            )
        train_time.append(t2 - t1)
        projected_train_time.append(train_projection)

        # validation
        val_loss = []
        val_mae = []
        synchronize(device)
        s1 = time.perf_counter()

        for iter, (x,y,x_mark,y_mark, embeddings) in enumerate(val_loader):
            valx = x.to(device=device, dtype=torch.float32, non_blocking=True)
            valy = y.to(device=device, dtype=torch.float32, non_blocking=True)
            valx_mark = x_mark.to(device=device, dtype=torch.float32, non_blocking=True)
            val_embedding = embeddings.to(device=device, dtype=torch.float32, non_blocking=True)
            metrics = engine.eval(valx, valx_mark, val_embedding, valy)
            val_loss.append(metrics[0])
            val_mae.append(metrics[1])
            if args.max_val_steps and iter + 1 >= args.max_val_steps:
                break

        synchronize(device)
        s2 = time.perf_counter()
        val_steps = len(val_loss)
        val_projection = (s2 - s1) * len(val_loader) / val_steps
        log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        if val_steps < len(val_loader):
            print(
                "Projected full validation: {:.4f} secs "
                "({:d}/{:d} batches measured)".format(
                    val_projection, val_steps, len(val_loader)
                )
            )
        val_time.append(s2 - s1)
        projected_val_time.append(val_projection)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mvalid_loss = np.mean(val_loss)
        mvalid_mae = np.mean(val_mae)

        his_loss.append(mvalid_loss)
        print("-----------------------")

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} "
        print(
            log.format(i, mtrain_loss, mtrain_mae),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_mae),
            flush=True,
        )

        if mvalid_loss < loss:
            print("###Update tasks appear###")
            if i <= 10:
                
                loss = mvalid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mse = 0
                print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
                print("epoch: ", i)
            else:
                test_outputs = []
                test_y = []

                for iter, (x,y,x_mark,y_mark, embeddings) in enumerate(test_loader):
                    testx = torch.Tensor(x).to(device)
                    testy = torch.Tensor(y).to(device)
                    testx_mark = torch.Tensor(x_mark).to(device)
                    test_embedding = torch.Tensor(embeddings).to(device)
                    with torch.no_grad():
                        preds = engine.model(testx, testx_mark, test_embedding)
                    test_outputs.append(preds)
                    test_y.append(testy)
                
                test_pre = torch.cat(test_outputs, dim=0)
                test_real = torch.cat(test_y, dim=0)

                amse = []
                amae = []
                
                for j in range(args.pred_len):
                    pred = test_pre[:, j,].to(device)
                    real = test_real[:, j, ].to(device)
                    metrics = metric(pred, real)
                    log = "Evaluate best model on test data for horizon {:d}, Test MSE: {:.4f}, Test MAE: {:.4f}"
                    amse.append(metrics[0])
                    amae.append(metrics[1])

                log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
                print(
                    log.format(
                        np.mean(amse), np.mean(amae)
                    )
                )

                if np.mean(amse) < test_log:
                    test_log = np.mean(amse)
                    loss = mvalid_loss
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mse = 0
                    print("Test low! Updating! Test Loss: {:.4f}".format(np.mean(amse)), end=", ")
                    print("Test low! Updating! Valid Loss: {:.4f}".format(mvalid_loss), end=", ")

                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mse += 1
                    print("No update")

        else:
            epochs_since_best_mse += 1
            print("No update")

        engine.scheduler.step()

        if epochs_since_best_mse >= args.es_patience and i >= args.epochs//2: # early stop
            break

    # Output consumption
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))
    if args.max_train_steps:
        print("Projected Average Training Time: {:.4f} secs/epoch".format(np.mean(projected_train_time)))
    if args.max_val_steps:
        print("Projected Average Validation Time: {:.4f} secs".format(np.mean(projected_val_time)))

    # Test
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    if args.skip_test:
        return

    engine.model.load_state_dict(torch.load(path + "best_model.pth", map_location=device))
    
    test_outputs = []
    test_y = []

    for iter, (x,y,x_mark,y_mark, embeddings) in enumerate(test_loader):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        testx_mark = torch.Tensor(x_mark).to(device)
        test_embedding = torch.Tensor(embeddings).to(device)
        with torch.no_grad():
            preds = engine.model(testx, testx_mark, test_embedding)
        test_outputs.append(preds)
        test_y.append(testy)

    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.cat(test_y, dim=0)

    amse = []
    amae = []
    
    for j in range(args.pred_len):
        pred = test_pre[:, j,].to(device)
        real = test_real[:, j, ].to(device)
        metrics = metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MSE: {:.4f}, Test MAE: {:.4f}"
        amse.append(metrics[0])
        amae.append(metrics[1])

    log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
    print(log.format(np.mean(amse), np.mean(amae)))

if __name__ == "__main__":
    t1 = time.perf_counter()
    main()
    t2 = time.perf_counter()
    print("Total time spent: {:.4f}".format(t2 - t1))

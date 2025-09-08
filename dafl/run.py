import os, json, argparse, numpy as np, torch, pandas as pd
from .utils import set_seed, make_out_dir, save_json
from .data import load_imagefolder_splits, make_client_loaders, make_test_loader
from .model import get_resnet18
from .train import train_local_full, eval_loss_acc
from .influence import make_logical_groups, keys_for_prefix, compute_group_influence
from .quality import compute_data_quality_scores
from .assignment import aggregate_by_influence_and_quality_blended
from . import plotting as viz

def parse_args():
    p = argparse.ArgumentParser(description="Data-Aware Layer Assignment (train-all, send-one)")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs_per_client", type=int, default=2)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max_influence_batches", type=int, default=10)
    p.add_argument("--sample_count_weight", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = make_out_dir(args.out_dir)
    print(f"[INFO] Output dir: {out_dir}")

    # Data
    full_train, test_set = load_imagefolder_splits(args.dataset_path, args.num_workers)
    num_classes = len(full_train.classes)
    client_train, client_val, client_counts = make_client_loaders(
        full_train, args.num_clients, args.batch_size, args.num_workers, seed=args.seed
    )
    test_loader = make_test_loader(test_set, args.batch_size, args.num_workers)

    # Model + logical groups
    device = torch.device(args.device)
    global_model = get_resnet18(num_classes).to(device)
    gkeys = list(global_model.state_dict().keys())
    prefixes = make_logical_groups(gkeys)
    groups   = [keys_for_prefix(p, gkeys) for p in prefixes]

    # Param sizes per prefix (for bytes accounting)
    param_sizes = {k: int(v.numel()) for k, v in global_model.state_dict().items()}
    def layer_param_size(prefix_keys): return sum(param_sizes[k] for k in prefix_keys)
    layer_sizes = {p: layer_param_size(ks) for p, ks in zip(prefixes, groups)}

    # History
    hist = {
        "round": [], "test_acc": [], "test_loss": [],
        "client_train_losses": [], "client_val_losses": [],
        "client_val_accs": [], "data_quality": [],
        "influences": [], "mapping": [],
        "bytes_round": [], "bytes_cum": [],
        "logical_prefixes": prefixes
    }
    bytes_cum = 0

    print("\n[INFO] Starting federated training")
    for rnd in range(1, args.rounds + 1):
        print(f"\n===== Round {rnd}/{args.rounds} =====")
        client_states, client_val_accs = [], []
        client_tr_losses, client_va_losses = [], []

        # Local training
        for cid in range(args.num_clients):
            tr = client_train[cid]; va = client_val[cid]
            if tr is None or len(tr.dataset) == 0:
                print(f" Client {cid}: no train data, skip.")
                client_states.append(global_model.state_dict().copy())
                client_val_accs.append(0.0)
                client_tr_losses.append(np.nan); client_va_losses.append(np.nan)
                continue
            local_state, tr_loss = train_local_full(
                global_model.state_dict(), tr, args.epochs_per_client, args.lr, device
            )
            client_states.append(local_state); client_tr_losses.append(float(tr_loss))
            if va is None or len(va.dataset)==0:
                va_loss, va_acc = np.nan, 0.0
            else:
                va_loss, va_acc = eval_loss_acc(local_state, va, device)
            client_va_losses.append(va_loss); client_val_accs.append(va_acc)
            print(f" Client {cid}: tr_n={len(tr.dataset)}, va_loss={va_loss if not np.isnan(va_loss) else float('nan'):.4f}, va_acc={va_acc*100:.2f}%")

        # Influence
        influence_model = get_resnet18(num_classes).to(device)
        influence_model.load_state_dict(global_model.state_dict())
        inf = compute_group_influence(influence_model, test_loader, groups, device, args.max_influence_batches)

        # Quality
        q_scores = compute_data_quality_scores(client_val_accs, client_counts, weight=args.sample_count_weight)

        # Assignment + aggregation
        new_state, mapping = aggregate_by_influence_and_quality_blended(
            global_model.state_dict(), client_states, prefixes, groups, inf, q_scores, alpha=args.alpha
        )
        global_model.load_state_dict(new_state)

        # Eval
        t_loss, t_acc = eval_loss_acc(global_model.state_dict(), test_loader, device)
        print(f" Server test: loss={t_loss:.4f}, acc={t_acc*100:.2f}%")

        # Bytes accounting
        bytes_round = sum(layer_sizes[p] * 4 for p in mapping.keys())  # float32
        bytes_cum += bytes_round

        # Log
        hist["round"].append(rnd)
        hist["test_acc"].append(float(t_acc))
        hist["test_loss"].append(float(t_loss))
        hist["client_train_losses"].append([float(x) for x in client_tr_losses])
        hist["client_val_losses"].append([float(x) if not np.isnan(x) else np.nan for x in client_va_losses])
        hist["client_val_accs"].append([float(x) for x in client_val_accs])
        hist["data_quality"].append([float(x) for x in q_scores])
        hist["influences"].append([float(x) for x in inf])
        hist["mapping"].append(mapping)
        hist["bytes_round"].append(int(bytes_round))
        hist["bytes_cum"].append(int(bytes_cum))

    # Save history + CSVs
    save_json(hist, os.path.join(out_dir, "history.json"))
    rounds = np.array(hist["round"]); test_acc = np.array(hist["test_acc"])
    test_loss = np.array(hist["test_loss"]); bytes_round = np.array(hist["bytes_round"])
    bytes_cum = np.array(hist["bytes_cum"])

    pd.DataFrame({
        "round": rounds, "test_acc": test_acc,
        "test_loss": test_loss, "bytes_round": bytes_round, "bytes_cum": bytes_cum
    }).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # client selection counts
    C = args.num_clients
    sel_counts = np.array([sum(1 for M in hist["mapping"] if cid in M.values()) for cid in range(C)])
    pd.DataFrame({"client_id": np.arange(C), "assigned_rounds": sel_counts}).to_csv(
        os.path.join(out_dir, "client_selection_counts.csv"), index=False
    )

    # Figures
    viz.plot_accuracy(rounds, test_acc, out_dir)
    viz.plot_loss(rounds, test_loss, out_dir)
    viz.plot_comm(rounds, bytes_cum, test_acc, out_dir)
    viz.plot_assignment_heatmap(prefixes, hist["mapping"], out_dir)
    viz.plot_influence_traces(rounds, prefixes, hist["influences"], out_dir, topk=8)
    viz.plot_update_frequency(prefixes, hist["mapping"], out_dir)
    viz.plot_client_parity(args.num_clients, hist["mapping"], out_dir)
    viz.plot_quality_iqr(rounds, hist["data_quality"], out_dir)

    print(f"\n[INFO] Finished. Artifacts in: {out_dir}")

if __name__ == "__main__":
    main()

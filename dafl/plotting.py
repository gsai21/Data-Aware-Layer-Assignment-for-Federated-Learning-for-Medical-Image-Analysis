import os, numpy as np, matplotlib.pyplot as plt

def plot_accuracy(rounds, test_acc, out_dir):
    plt.figure(); plt.plot(rounds, test_acc)
    plt.xlabel("Round"); plt.ylabel("Test accuracy"); plt.title("Accuracy vs. Rounds"); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"figures/acc_vs_rounds.pdf")); plt.close()

def plot_loss(rounds, test_loss, out_dir):
    plt.figure(); plt.plot(rounds, test_loss, marker="o")
    plt.xlabel("Round"); plt.ylabel("Test loss"); plt.title("Test Loss vs. Rounds"); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"figures/test_loss_vs_rounds.pdf")); plt.close()

def plot_comm(rounds, bytes_cum, test_acc, out_dir):
    plt.figure(); plt.plot(bytes_cum/1e6, test_acc, marker="o")
    plt.xlabel("Cumulative uplink (MB)"); plt.ylabel("Test accuracy"); plt.title("Accuracy vs. Communication"); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"figures/mb_vs_accuracy.pdf")); plt.close()

def plot_assignment_heatmap(logical_prefixes, mappings, out_dir):
    L = len(logical_prefixes); T = len(mappings)
    A = -1 * np.ones((L, T), dtype=int)
    for t, m in enumerate(mappings):
        for li, pref in enumerate(logical_prefixes):
            if pref in m: A[li, t] = m[pref]
    plt.figure(); plt.imshow(A, aspect="auto", interpolation="nearest")
    plt.yticks(range(L), logical_prefixes, fontsize=6)
    plt.xlabel("Round"); plt.ylabel("Layer (prefix)"); plt.title("Layer–Client Assignment Heatmap")
    plt.colorbar(label="Client id"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"figures/assignment_heatmap.pdf")); plt.close()

def plot_influence_traces(rounds, logical_prefixes, influences, out_dir, topk=8):
    L = len(logical_prefixes)
    inf = np.array(influences)        # T x L
    mean_inf = np.nanmean(inf, axis=0)
    k = min(topk, L); idx = np.argsort(-mean_inf)[:k]
    plt.figure()
    for i in idx:
        plt.plot(rounds, inf[:, i], label=logical_prefixes[i])
    plt.xlabel("Round"); plt.ylabel("Influence (avg grad-norm)")
    plt.title(f"Top-{k} Layer Influence Trajectories"); plt.legend(fontsize=6); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"figures/influence_traces.pdf")); plt.close()

def plot_update_frequency(logical_prefixes, mappings, out_dir):
    L = len(logical_prefixes)
    counts = np.zeros(L, dtype=int)
    for m in mappings:
        for li, pref in enumerate(logical_prefixes):
            if pref in m: counts[li] += 1
    order = np.argsort(-counts)
    plt.figure(); plt.bar(range(L), counts[order])
    labels = [logical_prefixes[i] for i in order]
    plt.xticks(range(L), labels, rotation=90, fontsize=6)
    plt.xlabel("Layer (sorted)"); plt.ylabel("#Rounds updated"); plt.title("Per-Layer Update Frequency")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"figures/layer_update_frequency.pdf")); plt.close()

def plot_client_parity(num_clients: int, mappings, out_dir):
    sel = np.zeros(num_clients, dtype=int)
    for m in mappings:
        for cid in m.values():
            if cid is not None: sel[cid] += 1
    plt.figure(); plt.bar(range(num_clients), sel)
    plt.xlabel("Client id"); plt.ylabel("#Assignments"); plt.title("Client Selection Parity")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"figures/client_selection_parity.pdf")); plt.close()

def plot_quality_iqr(rounds, data_quality, out_dir):
    C = max(len(x) for x in data_quality) if data_quality else 0
    Q = np.full((len(rounds), C), np.nan)
    for t, q in enumerate(data_quality):
        arr = np.asarray(q, dtype=float)
        Q[t, :len(arr)] = arr
    q_med = np.nanmedian(Q, axis=1)
    q_p25 = np.nanpercentile(Q, 25, axis=1)
    q_p75 = np.nanpercentile(Q, 75, axis=1)
    plt.figure()
    plt.plot(rounds, q_med, label="median")
    plt.fill_between(rounds, q_p25, q_p75, alpha=0.2)
    plt.xlabel("Round"); plt.ylabel("Quality score"); plt.title("Client Data-Quality Dynamics (median ± IQR)")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(out_dir,"figures/quality_dynamics.pdf")); plt.close()
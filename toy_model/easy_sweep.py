# The code I used for the easy model sweep (only difference is the produce_example)
# "Easy" here means that in a context, each entity and each relation only appears once - so we assume that the model does not learn an entity binding circuit, but rather a "look-X-Steps-Forward-Circuit"
# Need to confirm that

import os, math, time, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import itertools

### ALL HPARAM SETTINGS ARE HERE ### 

N_TRIALS = 50                     
MAX_EPOCHS = 100
PATIENCE = 10                                          
D_MLP_FACTOR = 4                
N_CTX = 64

D_MODEL_CHOICES = [128, 256, 512]
LAYER_CHOICES = [2, 3, 4]
HEAD_CHOICES  = [1, 2, 4]
ATTN_ONLY_CHOICES = [True, False]
WD_CHOICES = [0.01]
LR = [5e-4]

### Dataset ###

E = 100  # num entities
T = 10   # num types/relations

SEP = E + T
Q = E + T + 1
PAD = E + T + 2
D_VOCAB = E + T + 3

ENTITIES = np.arange(0, E)
TYPES    = np.arange(E, E + T)

N_WORLDS = 80_000
MIN_FACTS, MAX_FACTS = 4, 8
SEED = 0

rng = np.random.default_rng(SEED)

def produce_example(num_relations: int, *, allow_self_loops: bool = False):
    facts = []
    seen_head_rel = set()
    seen_e = set()
    seen_t = set()

    while len(facts) < num_relations:
        e = int(rng.integers(0, E))
        t = int(TYPES[rng.integers(0, T)])

        # enforce uniqueness of e and t (not just the tuple)
        if e in seen_e or t in seen_t:
            continue
        if (e, t) in seen_head_rel:
            continue

        # forbid self-loop 
        e2 = int(rng.integers(0, E))
        while not allow_self_loops and e2 == e:
            e2 = int(rng.integers(0, E))

        seen_head_rel.add((e, t))
        seen_e.add(e)
        seen_t.add(t)
        facts.append((e, t, e2))

    q_idx = int(rng.integers(0, num_relations))
    Eq, Tq, E2q = facts[q_idx]

    seq = []
    for (e, t, e2) in facts:
        seq.extend([e, t, e2, SEP])
    seq.extend([Tq, Eq, Q])

    return seq, E2q, facts

# --- Build dataset ---
rows = []
for _ in tqdm(range(N_WORLDS)):
    k = int(rng.integers(MIN_FACTS, MAX_FACTS + 1))
    seq, label, _ = produce_example(k, allow_self_loops=False)
    rows.append({"tokens": seq, "label": label})

df = pd.DataFrame(rows)

IGNORE_INDEX = -100

class EntityBindingDataset(Dataset):
    def __init__(self, dataframe, parse_tokens_if_str=True):
        self.df = dataframe.reset_index(drop=True)
        self.parse_tokens_if_str = parse_tokens_if_str

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["tokens"]
        tokens = torch.tensor(seq, dtype=torch.long)
        label  = torch.tensor(int(row["label"]), dtype=torch.long)
        return tokens, label


def collate_fn(batch):
    max_len = max(len(seq) for seq,_ in batch)
    B = len(batch)
    toks   = torch.full((B, max_len), PAD, dtype=torch.long)
    target = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)

    for i, (seq, label) in enumerate(batch):
        x = torch.tensor(seq if isinstance(seq, list) else seq.tolist(), dtype=torch.long)
        L = len(x)
        toks[i, :L] = x
        q_pos = (x == Q).nonzero(as_tuple=False).squeeze()
        assert q_pos.numel() == 1, "Each example must have exactly one Q"
        target[i, q_pos.item()] = int(label)
    return toks, target


train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = EntityBindingDataset(train_df)
val_dataset = EntityBindingDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = IGNORE_INDEX):
    """
    Accuracy over positions where targets != ignore_index.
    Returns correct_count and total_count
    """
    with torch.no_grad():
        mask = targets.ne(ignore_index)
        total = mask.sum().item()
        preds = logits.argmax(dim=-1)
        correct = preds.masked_select(mask).eq(targets.masked_select(mask)).sum().item()
        return correct, total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(d_model, n_layers, n_heads, attn_only):
    if d_model % n_heads != 0:
        return None
    d_head = d_model // n_heads
    d_mlp  = d_model * D_MLP_FACTOR
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_head,
        d_mlp=d_mlp,
        n_ctx=N_CTX,
        d_vocab=D_VOCAB,
        d_vocab_out=E,          # predict entities only
        act_fn="gelu",
        attn_only=attn_only,
        normalization_type="LN",
        use_attn_result=True,
        use_hook_tokens=True,
        device=device,
    )
    return HookedTransformer(cfg)

def run_trial(trial_id, cfg_dict, *, log_tb=True):

    seed = 1000
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    model = build_model(cfg_dict["d_model"], cfg_dict["n_layers"], cfg_dict["n_heads"], cfg_dict["attn_only"])
    if model is None:
        return {**cfg_dict, "status":"skipped_invalid_heads", "best_val_acc":None, "best_epoch":None, "epochs_run":0}

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg_dict["lr"],
                                  betas=(0.9, 0.98),
                                  weight_decay=cfg_dict["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    run_name = (f"D{cfg_dict['d_model']}_L{cfg_dict['n_layers']}_H{cfg_dict['n_heads']}"
                f"_attnOnly{int(cfg_dict['attn_only'])}"
                f"_lr{cfg_dict['lr']:.1e}_wd{cfg_dict['weight_decay']}")
    writer = SummaryWriter(f"runs/entity_binding/{run_name}") if log_tb else None

    best_val = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    global_step = 0
    ckpt_path = f"ckpts_determ/{run_name}.pt"
    os.makedirs("ckpts_determ", exist_ok=True)

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        tot_loss = 0.0; tot_c = 0; tot_n = 0
        for input_tokens, targets in train_loader:
            input_tokens, targets = input_tokens.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_tokens)  # [B, T, E]
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot_loss += loss.item()
            c, n = compute_accuracy(logits, targets)
            tot_c += c; tot_n += n
            if writer:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Acc/train_step", (c/n) if n else 0.0, global_step)
            global_step += 1

        train_loss = tot_loss / max(1, len(train_loader))
        train_acc = (tot_c / tot_n) if tot_n else 0.0

        model.eval()
        val_loss_accum = 0.0; val_c = 0; val_n = 0
        with torch.no_grad():
            for input_tokens, targets in val_loader:
                input_tokens, targets = input_tokens.to(device), targets.to(device)
                logits = model(input_tokens)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 targets.reshape(-1))
                val_loss_accum += loss.item()
                c, n = compute_accuracy(logits, targets)
                val_c += c; val_n += n
        val_loss = val_loss_accum / max(1, len(val_loader))
        val_acc = (val_c / val_n) if val_n else 0.0

        if writer:
            writer.add_scalar("Loss/val", val_loss, global_step)
            writer.add_scalar("Acc/val_epoch", val_acc, global_step)

        improved = val_acc > best_val + 1e-6
        if improved:
            best_val = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(),
                        "cfg": cfg_dict,
                        "epoch": epoch,
                        "val_acc": best_val}, ckpt_path)
        else:
            epochs_no_improve += 1

        # Optional: print concise trace
        print(f"[{run_name}] Ep {epoch:03d} | "
              f"trL {train_loss:.3f} trA {train_acc:.3f} | "
              f"vaL {val_loss:.3f} vaA {val_acc:.3f} | "
              f"best {best_val:.3f}@{best_epoch}")

        if epochs_no_improve >= PATIENCE:
            break

    if writer: writer.close()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {
        **cfg_dict,
        "status": "ok" if best_epoch > 0 else "no_improve",
        "best_val_acc": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "ckpt_path": ckpt_path if best_epoch > 0 else None,
        "run_name": run_name,
    }

def random_search(n_trials=N_TRIALS, log_tb=True):
    rng = np.random.default_rng(124)
    all_cfgs = list(itertools.product(D_MODEL_CHOICES, LAYER_CHOICES, HEAD_CHOICES,
                                  ATTN_ONLY_CHOICES, WD_CHOICES, LR))
    results = []
    print(all_cfgs)
    for t, (d_model, n_layers, n_heads, attn_only, weight_decay, lr) in enumerate(all_cfgs):
        cfg_dict = {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads":  n_heads,
            "attn_only": bool(attn_only),
            "weight_decay": float(weight_decay),
            "lr": float(lr),
        }
        res = run_trial(t, cfg_dict, log_tb=log_tb)
        results.append(res)

    df = pd.DataFrame(results)
    df.sort_values(by=["best_val_acc", "best_epoch"], ascending=[False, True], inplace=True)
    os.makedirs("sweeps", exist_ok=True)
    out_csv = "sweeps/sweep_results_determ.csv"
    df.to_csv(out_csv, index=False)
    print("\nTop configs:")
    print(df.head(10)[["best_val_acc","best_epoch","n_layers","n_heads","attn_only","lr","weight_decay","run_name"]])
    print(f"\nSaved: {out_csv}")
    return df

df_results = random_search(n_trials=N_TRIALS, log_tb=True)

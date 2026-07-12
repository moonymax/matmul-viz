"""Training for TinyGPT (TinyStories) with AdamW and selectable FFN variant.

Changes vs tiny-lm/main_v3.py:
  * Even (uniform) head schedule  -> see model.UNIFORM_HEADS.
  * Muon removed entirely; ALL params trained with torch.optim.AdamW
    (reason: the "sparse" FFN has 1-D SparseLinear params that Muon's
    dense orthogonalization cannot handle). Param-group split: weight decay on
    >=2-D weights only; no decay on 1-D params, norms, embeddings, ar_proj.
    LR ~3e-4 with the v3 warmup + cosine schedule.
  * FFN_TYPE selects one of {"ple","sparse","moe"}; the checkpoint stores the
    full config (incl. ffn_type) so inference.py can rebuild the exact model.

`datasets` / `tokenizers` are imported lazily inside functions so the model can
be constructed/tested without them (see smoke_test.py).
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .model import TinyGPT, GPTConfig
except ImportError:
    from model import TinyGPT, GPTConfig

# --- Device ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# --- Which FFN to train ---
FFN_TYPE = os.environ.get("FFN_TYPE", "ple")   # {"ple", "sparse", "moe"}

# --- Training hyperparameters ---
BATCH_SIZE = 64
TRAIN_EPOCHS = 3
ADAM_LR = 3e-4
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0

TOKENIZER_PATH = f"tokenizer_{FFN_TYPE}.json"
RESUME_PATH = f"checkpoint_{FFN_TYPE}_resume.pt"


def make_config(vocab_size: int) -> GPTConfig:
    return GPTConfig(vocab_size=vocab_size, ffn_type=FFN_TYPE).validate()


# --- BPE Tokenizer (lazy import) ---
def build_or_load_tokenizer(texts=None, vocab_target=4096):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading tokenizer from {TOKENIZER_PATH}")
        return Tokenizer.from_file(TOKENIZER_PATH)

    assert texts is not None, "Need texts to build tokenizer"
    print(f"Training BPE tokenizer (target vocab={vocab_target})...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<unk>", "<eos>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_target,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(TOKENIZER_PATH)

    unk_id = tokenizer.token_to_id("<unk>")
    n_unk = n_total = 0
    for t in texts[:1000]:
        enc = tokenizer.encode(t)
        n_total += len(enc.ids)
        n_unk += sum(1 for i in enc.ids if i == unk_id)
    print(f"Saved {TOKENIZER_PATH} (vocab={tokenizer.get_vocab_size()}, "
          f"coverage={100 * (1 - n_unk / max(n_total, 1)):.1f}%)")
    return tokenizer


# --- Dataset ---
class StoryDataset(Dataset):
    def __init__(self, chunks):
        self.data = chunks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c = self.data[idx]
        return c[:-1], c[1:]


def tokenize_stories(stories, tokenizer, eos_id, block_size):
    all_tokens = []
    for text in stories:
        all_tokens.extend(tokenizer.encode(text).ids)
        all_tokens.append(eos_id)

    chunks = []
    stride = block_size + 1
    for i in range(0, len(all_tokens) - block_size, stride):
        chunk = all_tokens[i:i + stride]
        if len(chunk) == stride:
            chunks.append(torch.tensor(chunk, dtype=torch.long))

    avg_toks = len(all_tokens) / max(len(stories), 1)
    print(f"  {len(all_tokens):,} tokens ({avg_toks:.0f} avg/story), {len(chunks):,} chunks")
    return chunks


# --- Optimizer: AdamW for ALL params with a decay / no-decay split ---
def build_optimizer(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # no weight decay on 1-D params (biases, norms), embeddings, ar_proj
        if p.ndim < 2 or "emb" in name or "ar_proj" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=ADAM_LR, betas=ADAM_BETAS)


# --- Checkpointing (config carries ffn_type so inference can rebuild) ---
def save_checkpoint(model, optimizer, scheduler, global_step, epoch, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "tokenizer_path": TOKENIZER_PATH,
        "config": model.cfg.to_dict(),
        "global_step": global_step,
        "epoch": epoch,
    }, path)


def make_scheduler(optimizer, total_steps):
    def lr_factor(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


# --- Training ---
def main():
    from datasets import load_dataset

    print(f"Device: {DEVICE}  |  FFN_TYPE={FFN_TYPE}")

    print("\nLoading TinyStories...")
    ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")
    stories = [ex["text"] for ex in ds]
    print(f"Loaded {len(stories):,} stories")

    tokenizer = build_or_load_tokenizer(stories)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")
    print(f"Vocab: {vocab_size}, pad={pad_id}, eos={eos_id}")

    cfg = make_config(vocab_size)
    print(f"Config: dim={cfg.embed_dim}, layers={cfg.n_layers}, head_dim={cfg.head_dim}, "
          f"heads/layer={cfg.uniform_heads} (attn_dim={cfg.attn_dim}), "
          f"block_size={cfg.block_size}, batch={BATCH_SIZE}")

    print("\nTokenizing stories...")
    chunks = tokenize_stories(stories, tokenizer, eos_id, cfg.block_size)
    dataset = StoryDataset(chunks)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=(DEVICE == "cuda"), drop_last=True)

    model = TinyGPT(cfg).to(DEVICE)
    print(f"\nModel parameters: {model.count_parameters():,}")

    total_steps = TRAIN_EPOCHS * len(loader)
    optimizer = build_optimizer(model)
    scheduler = make_scheduler(optimizer, total_steps)

    start_epoch = 0
    global_step = 0
    resumed = False
    if os.path.exists(RESUME_PATH):
        print(f"\nResuming from {RESUME_PATH}...")
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        resumed = True
        print(f"  Resumed at epoch {start_epoch + 1}, step {global_step}")

    print(f"\n=== Training ({TRAIN_EPOCHS} epochs, {total_steps:,} steps) ===")
    t0 = time.time()
    total_tokens = 0

    for epoch in range(start_epoch, TRAIN_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        steps_into_epoch = global_step - epoch * len(loader) if epoch == start_epoch and resumed else 0

        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx < steps_into_epoch:
                continue

            x, y = x.to(DEVICE), y.to(DEVICE)

            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                   ignore_index=pad_id)
            aux = model.collect_aux_loss()   # 0 for ple/sparse; MoE load-balance term
            total = loss + aux

            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_count += 1
            global_step += 1
            total_tokens += x.numel()

            if global_step % 100 == 0:
                elapsed = time.time() - t0
                tps = total_tokens / elapsed if elapsed > 0 else 0
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step}/{total_steps} — loss: {loss.item():.4f} — "
                      f"aux: {float(aux.detach()):.4f} — {tps:,.0f} tok/s — lr: {lr:.6f}")

            if global_step % 5000 == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, epoch, RESUME_PATH)
                print(f"  Saved {RESUME_PATH} (step {global_step})")

        avg_loss = epoch_loss / max(epoch_count, 1)
        print(f"  Epoch {epoch + 1}/{TRAIN_EPOCHS} — avg loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, scheduler, global_step, epoch + 1,
                        f"checkpoint_{FFN_TYPE}_epoch{epoch + 1}.pt")
        save_checkpoint(model, optimizer, scheduler, global_step, epoch + 1, RESUME_PATH)
        print(f"  Saved checkpoint_{FFN_TYPE}_epoch{epoch + 1}.pt")

    save_checkpoint(model, optimizer, scheduler, global_step, TRAIN_EPOCHS,
                    f"checkpoint_{FFN_TYPE}.pt")
    if os.path.exists(RESUME_PATH):
        os.remove(RESUME_PATH)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 3600:.1f}h")
    print(f"Saved checkpoint_{FFN_TYPE}.pt")


if __name__ == "__main__":
    main()

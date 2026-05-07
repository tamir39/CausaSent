"""Fine-tune mT5-base for action generation."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import Adafactor, get_linear_schedule_with_warmup

from ..data.dataset import load_action_dataset
from ..models.mt5_action import load_mt5
from ..utils.repro import dump_run_metadata, log_metrics, make_run_dir, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = make_run_dir(cfg["output"].get("log_dir", "runs"), tag="mt5")
    dump_run_metadata(run_dir, cfg)

    model, tokenizer = load_mt5(cfg["model"]["pretrained"])
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required when gradient checkpointing is on

    train_ds = load_action_dataset(
        cfg["data"]["train_path"], tokenizer,
        cfg["data"]["max_input_len"], cfg["data"]["max_output_len"],
    )
    val_ds = load_action_dataset(
        cfg["data"]["val_path"], tokenizer,
        cfg["data"]["max_input_len"], cfg["data"]["max_output_len"],
    )

    bs = cfg["train"]["batch_size"]
    accum = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    epochs = cfg["train"]["epochs"]
    total_steps = max(1, math.ceil(len(train_loader) / accum) * epochs)
    # Adafactor (T5/mT5's native optimizer) keeps factored second-moment estimates
    # instead of AdamW's full per-parameter state — roughly halves optimizer memory.
    optim = Adafactor(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=cfg["train"]["weight_decay"],
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * cfg["train"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    ckpt_dir = Path(cfg["output"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        optim.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / accum).backward()
            running += out.loss.item()
            if step % accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                optim.step()
                sched.step()
                optim.zero_grad()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={running / step:.4f}")

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                v_loss += model(**batch).loss.item()
        v_loss /= max(1, len(val_loader))
        print(f"[epoch {epoch}] val_loss={v_loss:.4f}")
        log_metrics(run_dir, {"epoch": epoch, "val_loss": v_loss})

        if v_loss < best_val:
            best_val = v_loss
            torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "best.pt")
            print(f"  ↳ saved best checkpoint (val_loss={v_loss:.4f})")

    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "last.pt")


if __name__ == "__main__":
    main()

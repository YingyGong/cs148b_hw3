"""§3 — CLIP-style pretraining on EuroSAT.

You implement the training loop. This script provides the CLI scaffolding,
config loading, and logging hooks.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from vlm.data import build_eurosat_loaders, EUROSAT_CLASSES
from basics.vit import ViT
from basics.text_encoder import FrozenTextEncoder
from vlm.clip import ProjectionHeads, init_logit_scale, clip_loss
from vlm.eval import zeroshot_classification_accuracy

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: students fill in the training loop.
    # Sketch:
    #   1. Build train/val/test loaders via vlm.data.build_eurosat_loaders.
    #   2. Build the ViT (basics.vit.ViT) and FrozenTextEncoder.
    #   3. Build ProjectionHeads + logit_scale.
    #   4. AdamW optimizer, cosine LR schedule.
    #   5. For each epoch:
    #         - Train one epoch with vlm.clip.clip_loss.
    #         - Clamp logit_scale.data to <= ln(100).
    #         - Compute zero-shot val accuracy via vlm.eval.zeroshot_classification_accuracy.
    #         - Log to stdout (and W&B if args.wandb).
    #   6. Save the best checkpoint to args.output_dir / "best.pt".
    
    device = torch.device(args.device)
    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    class_indices = list(range(len(EUROSAT_CLASSES)))

    train_loader, val_loader, test_loader = build_eurosat_loaders(
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    vit_config = cfg["vit"]
    vit = ViT(
        image_size=vit_config["img_size"], patch_size=vit_config["patch_size"],
        d_model=vit_config["d_model"], num_heads=vit_config["num_heads"],
        num_blocks=vit_config["num_blocks"], dropout=vit_config["dropout"],
    ).to(device)

    text_encoder = FrozenTextEncoder(model_name=cfg["text_encoder"]["model_name"]).to(device)

    projection_heads = ProjectionHeads(
        d_image=vit_config["d_model"],
        d_text=cfg["text_encoder"]["d_text"],
        d_proj=cfg["projection"]["d_proj"],
    ).to(device)
    logit_scale = init_logit_scale().to(device)

    optimizer = torch.optim.AdamW(
        list(vit.parameters()) + list(projection_heads.parameters()) + [logit_scale],
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )

    num_epochs = cfg["train"]["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    if args.wandb:
        import wandb
        wandb.init(project="clip-eurosat", config=cfg)

    best_val_acc = -1.0
    for epoch in range(num_epochs):
        vit.train()
        projection_heads.train()
        for imgs, captions in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            image_emb = vit(imgs)
            text_emb = text_encoder(captions)
            image_proj, text_proj = projection_heads(image_emb, text_emb)
            loss = clip_loss(image_proj, text_proj, logit_scale)
            loss.backward()
            optimizer.step()
            logit_scale.data.clamp_(max=torch.log(torch.tensor(100.0)))

        val_acc = zeroshot_classification_accuracy(
            vit, projection_heads, text_encoder, val_loader,
            class_prompts, class_indices, device,
        )
        print(f"epoch {epoch+1}/{num_epochs}  val_acc={val_acc:.4f}")
        if args.wandb:
            wandb.log({"val_acc": val_acc, "epoch": epoch + 1})
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "vit": vit.state_dict(),
                    "projection_heads": projection_heads.state_dict(),
                    "logit_scale": logit_scale.data,
                },
                args.output_dir / "best.pt",
            )



if __name__ == "__main__":
    main()

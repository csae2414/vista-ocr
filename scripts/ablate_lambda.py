"""Ablation: lambda balance between text and location losses.

Paper formula: L = lambda * L_text + (1 - lambda) * L_loc.

The paper does NOT report a value. PLAN_VM lists {0.3, 0.5, 0.7} as the
sweep. This script runs the same number of steps for each lambda and
reports loss components on a small held-out PDFA slice so we can pick
the best value before launching long pretraining.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from vista_ocr.ablation import Ablation, AblationVariant
from vista_ocr.data.dataloader import DataLoaderConfig, make_pdfa_dataloader
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.logging_config import setup_logging
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.training.train_loop import TrainConfig

LOG = logging.getLogger("ablate_lambda")


@dataclass
class _SharedCtx:
    shards: list[Path]
    spm: Path
    page_h: int
    page_w: int
    steps: int


class LambdaAblation(Ablation):
    def __init__(self, ctx: _SharedCtx, lambdas: list[float]) -> None:
        self.ctx = ctx
        self._lambdas = lambdas

    def variants(self) -> list[AblationVariant]:
        return [
            AblationVariant(name=f"lambda={lam:.2f}", overrides={"lambda_text": lam},
                            extra={"lambda": lam})
            for lam in self._lambdas
        ]

    def build(self, variant):
        lam = variant.overrides["lambda_text"]
        grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
        tokenizer = VistaTokenizer(spm_model_path=str(self.ctx.spm), grid=grid)
        encoder = FCNEncoderWidther(input_channels=1, dropout=0.5, gradient_checkpointing=True)
        decoder = small_random_decoder(
            vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=4, n_heads=16,
            ffn_dim=4096, max_position_embeddings=4096,
        )
        model = VistaOCR(encoder=encoder, decoder=decoder)

        pre_cfg = PreprocessConfig(
            target_h=self.ctx.page_h, target_w=self.ctx.page_w, pad_multiple=32,
        )
        loader = make_pdfa_dataloader(
            shards=[str(s) for s in self.ctx.shards],
            tokenizer=tokenizer, pre_cfg=pre_cfg,
            dl_cfg=DataLoaderConfig(micro_batch_size=1, num_workers=4, prefetch_factor=4),
        )
        cfg = TrainConfig(
            base_lr=3e-4, warmup_steps=50, total_steps=self.ctx.steps,
            micro_batch_size=1, grad_accum_steps=1, log_every=100,
            lambda_text=lam,
            target_h=self.ctx.page_h, target_w=self.ctx.page_w, pad_multiple=32,
            device="cuda", autocast_dtype=torch.bfloat16, gradient_checkpointing=True,
            freeze_decoder=False,
            adam_betas=(0.9, 0.98), adam_eps=1e-6, label_smoothing=0.1,
        )
        return model, tokenizer, cfg, loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--page-h", type=int, default=1100)
    ap.add_argument("--page-w", type=int, default=850)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    args = ap.parse_args()

    setup_logging(level="INFO")
    ctx = _SharedCtx(
        shards=args.shards, spm=args.spm, page_h=args.page_h,
        page_w=args.page_w, steps=args.steps,
    )
    abl = LambdaAblation(ctx, args.lambdas)
    results = abl.run(max_steps=args.steps)
    LOG.info("LAMBDA ABLATION SUMMARY (steps=%d each)", args.steps)
    Ablation.report(
        results,
        columns=["name", "lambda", "loss_first", "loss_last", "delta", "elapsed"],
    )


if __name__ == "__main__":
    main()

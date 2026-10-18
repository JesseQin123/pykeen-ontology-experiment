"""Thin wrapper around PyKEEN: train a model, persist it, and expose a numpy score function."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from pykeen.models import model_resolver
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop

from .data import KG

torch.set_num_threads(16)

# Moderate, literature-informed configs; not exhaustively tuned (see REPORT.md caveats).
CONFIGS = {
    "TransE": dict(
        model_kwargs=dict(embedding_dim=256, scoring_fct_norm=1),
        loss="NSSALoss", loss_kwargs=dict(margin=6.0, adversarial_temperature=0.5),
        loop="sLCWA", num_negs=128, lr=1e-3, batch_size=1024, epochs=150,
    ),
    "RotatE": dict(
        model_kwargs=dict(embedding_dim=256),
        loss="NSSALoss", loss_kwargs=dict(margin=6.0, adversarial_temperature=0.5),
        loop="sLCWA", num_negs=128, lr=1e-3, batch_size=1024, epochs=150,
    ),
    "ComplEx": dict(
        model_kwargs=dict(embedding_dim=256, regularizer="lp", regularizer_kwargs=dict(p=3.0, weight=5e-3, normalize=True)),
        loss="CrossEntropyLoss", loss_kwargs={},
        loop="LCWA", lr=1e-3, batch_size=1024, epochs=150,
    ),
}


DEFAULT = dict(
    model_kwargs=dict(embedding_dim=256),
    loss="NSSALoss", loss_kwargs=dict(margin=6.0, adversarial_temperature=0.5),
    loop="sLCWA", num_negs=128, lr=1e-2, batch_size=1024, epochs=40,
)


def train(kg: KG, name: str, tf=None, seed: int = 0, device: str = "cpu", epochs: int | None = None,
          cache: Path | None = None, **overrides):
    cfg = {**CONFIGS.get(name, DEFAULT), **overrides}
    tf = tf if tf is not None else kg.factory
    if cache is not None and cache.exists():
        model = torch.load(cache, weights_only=False, map_location=device)
        return model, {"cached": True}
    torch.manual_seed(seed)
    model = model_resolver.make(
        name, triples_factory=tf, loss=cfg["loss"], loss_kwargs=cfg["loss_kwargs"],
        random_seed=seed, **cfg["model_kwargs"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    if cfg["loop"] == "sLCWA":
        loop = SLCWATrainingLoop(model=model, triples_factory=tf, optimizer=opt,
                                 negative_sampler="basic", negative_sampler_kwargs=dict(num_negs_per_pos=cfg["num_negs"]))
    else:
        loop = LCWATrainingLoop(model=model, triples_factory=tf, optimizer=opt)
    t0 = time.time()
    losses = loop.train(triples_factory=tf, num_epochs=epochs or cfg["epochs"], batch_size=cfg["batch_size"],
                        use_tqdm=False)
    info = {"train_seconds": time.time() - t0, "final_loss": float(losses[-1]), "epochs": epochs or cfg["epochs"]}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, cache)
    return model, info


def score_fn(model, num_relations: int):
    device = next(model.parameters()).device

    chunk = 32  # MPS cannot index > 2^31 elements; RotatE materialises (B, n, d) during 1-vs-all scoring

    @torch.inference_mode()
    def f(heads: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        model.eval()
        out = np.empty((len(heads), model.num_entities), dtype=np.float32)
        for s in range(0, len(heads), chunk):
            h, d = heads[s : s + chunk], dirs[s : s + chunk]
            o = out[s : s + chunk]
            fwd = d < num_relations
            if fwd.any():
                hr = torch.as_tensor(np.stack([h[fwd], d[fwd]], 1), device=device)
                o[fwd] = model.score_t(hr).cpu().numpy()
            if (~fwd).any():
                rt = torch.as_tensor(np.stack([d[~fwd] - num_relations, h[~fwd]], 1), device=device)
                o[~fwd] = model.score_h(rt).cpu().numpy()
        return out

    return f


def entity_embeddings(model) -> np.ndarray:
    """Real-valued view of entity embeddings (complex models are split into [re, im])."""
    w = model.entity_representations[0](indices=None).detach().cpu()
    if torch.is_complex(w):
        w = torch.cat([w.real, w.imag], dim=-1)
    return w.numpy().astype(np.float32)


def relation_embeddings(model) -> np.ndarray:
    w = model.relation_representations[0](indices=None).detach().cpu()
    if torch.is_complex(w):
        w = torch.cat([w.real, w.imag], dim=-1)
    return w.numpy().astype(np.float32)

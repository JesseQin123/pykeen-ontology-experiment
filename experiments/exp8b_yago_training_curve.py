"""Experiment 8b - How much training does RotatE need on the full YAGO3-10 to recover its accuracy?

Exp 8 trained 25 epochs on the full graph (MRR 0.147 on the fixed 1,000 test triples), far below the
published ~0.49. Here: keep training the same model and evaluate on the same 1,000 triples at checkpoints.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pykeen.models import model_resolver
from pykeen.training import SLCWATrainingLoop

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp8_scale_accuracy import load_yago, sub_kg  # noqa: E402
from kgv import evaluate, kge  # noqa: E402

LR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
CHECKPOINTS = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (10, 25, 50, 75, 100, 150)
OUT = ROOT / "results" / ("exp8b_yago_training_curve.json" if LR == 1e-2 else f"exp8b_yago_training_curve_lr{LR:g}.json")

train, valid, test, ent, rel = load_yago()
rng = np.random.default_rng(0)
fixed = test[rng.permutation(len(test))[:1000]]
kg = sub_kg(train, valid, fixed, ent, rel, np.ones(len(ent), bool))
torch.manual_seed(0)
model = model_resolver.make("RotatE", triples_factory=kg.factory, loss="NSSALoss",
                            loss_kwargs=dict(margin=6.0, adversarial_temperature=0.5), random_seed=0,
                            embedding_dim=256).to("mps")
opt = torch.optim.Adam(model.parameters(), lr=LR)
loop = SLCWATrainingLoop(model=model, triples_factory=kg.factory, optimizer=opt, negative_sampler="basic",
                         negative_sampler_kwargs=dict(num_negs_per_pos=128))
res, done, t0 = [], 0, time.time()
for target in CHECKPOINTS:
    # num_epochs is the cumulative total; continue_training keeps the weights (otherwise PyKEEN re-initialises)
    loop.train(triples_factory=kg.factory, num_epochs=target, batch_size=1024, use_tqdm=False,
               continue_training=done > 0)
    done = target
    m = evaluate.summarize(evaluate.ranks(kg, kge.score_fn(model, kg.num_relations), kg.test))
    res.append({"epochs": done, "hours": (time.time() - t0) / 3600, **m})
    print(time.strftime("%H:%M:%S"), res[-1], flush=True)
    OUT.write_text(json.dumps(res, indent=2))
print("done", flush=True)

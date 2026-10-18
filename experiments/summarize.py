"""Collect the headline numbers from every results/*.json into one compact summary (results/SUMMARY.json).

Every figure quoted in REPORT.md comes from this file, so each claim traces back to a raw result.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def pick(d, *keys):
    return {k: round(d[k], 4) if isinstance(d.get(k), float) else d.get(k) for k in keys if d and k in d}


S = {}
e1 = load("exp1_wn18rr.json")
if e1:
    S["wn18rr_random_holdout"] = {k: pick(v, "MRR", "Hits@1", "Hits@10") for k, v in e1["methods"].items()}
    S["wn18rr_random_holdout"]["RotatE_3seed_mean_std"] = e1["RotatE_seed_mean_std"]
    S["wn18rr_reachability"] = {k: (v if isinstance(v, float) else {kk: pick(vv, "MRR", "Hits@10", "n") for kk, vv in v.items()})
                                for k, v in e1["reachability"].items()}
    S["wn18rr_suggestions_open_world"] = e1["suggestions"]
e11 = load("exp11_anyburl.json")
if e11:
    S["anyburl"] = e11
e13 = load("exp13_ensembles.json")
if e13:
    S["ensembles"] = e13
e12 = load("exp12_seed_stability.json")
if e12:
    S["seed_stability"] = e12
e7 = load("exp7_model_zoo_wn18rr.json")
if e7:
    S["model_zoo_wn18rr"] = {k: (pick(v["test"], "MRR", "Hits@1", "Hits@10") if "test" in v else v) for k, v in e7.items()}
for tag in ("2020-01-01_2026-08-05", "2018-01-01_2020-01-01"):
    e4 = load(f"exp4_go_{tag}.json")
    if e4:
        S[f"go_temporal_{tag}"] = {
            "setup": e4["setup"],
            "methods_tail_only": {k: pick(v["tail_only(child->parent)"], "MRR", "Hits@1", "Hits@10") for k, v in e4["methods"].items()},
            "methods_novel_tail": {k: pick(v["novel_tail_only"], "MRR", "Hits@1", "Hits@10") for k, v in e4["methods"].items()},
            "methods_entailed_tail": {k: pick(v["entailed_tail_only"], "MRR", "Hits@1", "Hits@10") for k, v in e4["methods"].items()},
            "removed_edge_AUC": e4.get("removed_edge_AUC(model flags later-removed edges)"),
            "realized_suggestions": e4.get("realized_suggestions_is_a"),
            "llm_rerank_qwen": {k: v for k, v in (e4.get("llm_rerank_novel_is_a") or {}).items() if k != "examples"},
        }
e9 = load("exp9_poincare_go.json")
if e9:
    S["poincare_go"] = {a: pick(v["all_tail_only"], "MRR", "Hits@1", "Hits@10") for a, v in e9["by_alpha"].items()}
e10 = load("exp10_new_terms.json")
if e10:
    S["go_new_terms"] = {"n_queries": e10["n_queries"], "methods": e10["methods"],
                         "llm_rerank_qwen": {k: v for k, v in e10.get("llm_rerank", {}).items() if k != "examples"}}
e3w, e3f = load("exp3_alignment_wn18rr.json"), load("exp3_alignment_fb15k237.json")
S["alignment_renamed_views"] = {"wn18rr": e3w and e3w["results"], "fb15k237": e3f and e3f["results"],
                                "wn18rr_text": e3w and {k: {kk: vv for kk, vv in v.items() if kk != "examples_label_pairs"} for k, v in e3w["text"].items()}}
S["alignment_unsupervised_GW"] = load("exp6_unsupervised_alignment.json")
e5 = load("exp5_oaei_anatomy.json")
if e5:
    S["oaei_anatomy"] = {"anchors": e5["auto_anchors(exact label)"], "nontrivial_ranking": e5["nontrivial_ranking"],
                         "qwen_rerank": {k: v for k, v in e5["llm_rerank_nontrivial"].items() if k != "examples"},
                         "pipeline_qwen": e5.get("alignment_PRF")}
e15 = load("exp15_frontier_llm.json")
if e15:
    S["frontier_llm"] = e15
e14 = load("exp14_llm.json")
if e14:
    S["local_llm_extra"] = e14
e2, e2b = load("exp2_scaling.json"), load("exp2b_ann.json")
S["scaling"] = {"latency": e2 and e2.get("a_query_cost_vs_size"), "training": e2 and e2.get("b_training_cost"),
                "real_graph": e2 and e2.get("c_real_graph_latency"), "ann_realistic": e2b}
S["scale_accuracy_yago"] = load("exp8_scale_accuracy_yago.json")
S["yago_training_curve"] = {"lr=1e-2": load("exp8b_yago_training_curve.json"), "lr=2e-3": load("exp8b_yago_training_curve_lr0.002.json")}
e16 = load("exp16_llm_blind_rating.json")
if e16:
    S["llm_blind_rating"] = {k: v["summary"] for k, v in e16.items()}
e17 = load("exp17_llm_on_structure.json")
if e17:
    S["llm_on_structure"] = {k: v for k, v in e17.items() if not k.startswith("B_examples")}
(R / "SUMMARY.json").write_text(json.dumps(S, indent=1, default=str))
print("wrote", R / "SUMMARY.json", len(json.dumps(S)) // 1024, "KB")

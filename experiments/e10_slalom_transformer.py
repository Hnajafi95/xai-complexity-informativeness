#!/usr/bin/env python3
"""
E10 (HPC / GPU) -- SLALOM on a real transformer: the "Architecture (attention)"
cell of the Cross-Method Failure Map, run empirically.

Goal: show on a fine-tuned transformer that (a) SLALOM produces a high-fidelity
token-level explanation that separates a token's standalone *value* from its
*importance* (interaction weight), and (b) raw attention is NOT a faithful
importance signal -- the central claim of Leemann et al. (2025), TMLR. We
quantify faithfulness with a token-deletion test: rank tokens by each method,
mask the top-k, and measure how fast the predicted-class probability drops
(faster drop = more faithful). SLALOM-importance should drop faster than
attention and random.

Written against SLALOM's documented API (its test_minimal.py). The paper's E10
numbers were produced on a single GPU.

------------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------------
  # lightweight install is enough for SLALOM + this script:
  pip install slalom-explanations transformers torch datasets matplotlib scikit-learn

  # then:
  python e10_slalom_transformer.py

  (Model lvwerra/distilbert-imdb downloads from the HF hub on first run.
   One GPU is enough; CPU also works but is slower.)

For the *full* quantitative fidelity tables from the paper (SLALOM vs LIME/SHAP/
attention, insertion-deletion), their own tested code is in the repo:
  git clone https://github.com/tleemann/slalom_explanations
  conda env create -f environment.yml && conda activate slalom
  conda install -c conda-forge shap
  # then run notebooks/G_FidelityMetrics.ipynb
------------------------------------------------------------------------------
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from slalom_explanations import (SLALOMLocalExplanantions, slalom_scatter_plot)

MODEL = "lvwerra/distilbert-imdb"           # fine-tuned binary sentiment transformer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_REVIEWS = 50                              # for the faithfulness test
MAXLEN = 256
rng = np.random.default_rng(0)

print(f"device={DEVICE}; loading {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL)
# eager attention so output_attentions=True actually returns the weights
# (the default SDPA kernel does not support returning attentions).
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, attn_implementation="eager").to(DEVICE).eval()

# --------------------------------------------------------------------------
# Part 1 (low risk, verified API): qualitative SLALOM explanation + scatter
# --------------------------------------------------------------------------
explainer = SLALOMLocalExplanantions(model, tok, modes=["value", "imp", "lin"],
                                     device=DEVICE)
demo = ("This movie was absolutely wonderful: the acting was superb and the "
        "story was gripping, even if the ending felt a little rushed.")
res = explainer.tokenize_and_explain(demo)
print("\n[Part 1] SLALOM (value, importance, linear) for a demo review:")
for token, scores in res:
    print(f"  {token:>14s}  value={scores[0]:+.3f}  imp={scores[1]:+.3f}")
try:
    slalom_scatter_plot(res, sizex=6, sizey=6)
    plt.savefig("e10_slalom_scatter.pdf", bbox_inches="tight")
    plt.close("all")
    print("  wrote e10_slalom_scatter.pdf")
except Exception as e:
    print(f"  [scatter plot skipped: {e}]")


# --------------------------------------------------------------------------
# Part 2 (the paper-relevant result): MoRF deletion-faithfulness.
# Rank tokens by each method, mask the MOST-relevant first, and watch the
# predicted-class probability fall (faster drop = more faithful). SLALOM is
# ranked by its LINEAR attribution exp(importance)*value (linearize_explanation),
# directed toward the predicted class -- the faithfulness-relevant SLALOM score,
# not the raw interaction axis. Alignment is exact: tokenize_and_explain returns
# one score per content word-piece IN ORDER (it is tokenizer(text)["input_ids"]
# [1:-1]), so we rebuild the input ids straight from SLALOM's own token list.
# --------------------------------------------------------------------------
from slalom_explanations import linearize_explanation


@torch.no_grad()
def predicted_class_prob(input_ids, attn_mask, cls):
    out = model(input_ids=input_ids, attention_mask=attn_mask).logits
    return torch.softmax(out, -1)[0, cls].item()


@torch.no_grad()
def attention_received(input_ids, attn_mask):
    """Mean last-layer attention received by each position (over heads & queries)."""
    out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
    att = out.attentions[-1][0]                        # (heads, q, k)
    return att.mean(0).mean(0).detach().cpu().numpy()  # (k,)


def build_ids_from_tokens(token_strings):
    """[CLS] + content + [SEP] id tensor rebuilt from SLALOM's token list, so
    positions line up 1:1 with SLALOM's per-token scores."""
    content = tok.convert_tokens_to_ids(token_strings)
    ids = [tok.cls_token_id] + content + [tok.sep_token_id]
    t = torch.tensor([ids], device=DEVICE)
    return t, torch.ones_like(t)


def deletion_curve(ids, attn_mask, cls, content_scores, steps=10):
    """MoRF deletion over CONTENT positions (offset +1 for [CLS]); prob trace."""
    n = len(content_scores)
    order = np.argsort(-content_scores)                # most-relevant first
    mask_id = tok.mask_token_id if tok.mask_token_id is not None else tok.pad_token_id
    cur = ids.clone()
    trace = [predicted_class_prob(cur, attn_mask, cls)]
    k = max(1, n // steps)
    for s in range(steps):
        for ci in order[s*k:(s+1)*k]:
            cur[0, ci + 1] = mask_id                    # +1 skips [CLS]
        trace.append(predicted_class_prob(cur, attn_mask, cls))
    return np.array(trace)


# load some IMDB test reviews
try:
    from datasets import load_dataset
    ds = load_dataset("imdb", split="test")
    idx = rng.choice(len(ds), size=N_REVIEWS, replace=False)
    texts = [ds[int(i)]["text"] for i in idx]
except Exception as e:
    print(f"[datasets load failed: {e}; falling back to the demo review only]")
    texts = [demo]

curves = {"SLALOM (linear)": [], "Attention": [], "Random": []}
done = 0
for nrev, t in enumerate(texts):
    try:
        res = explainer.tokenize_and_explain(t)            # ordered per-wordpiece
        lin = linearize_explanation(res)                   # [(tok, exp(imp)*value)]
        toks = [tk for tk, _ in lin][:MAXLEN - 2]
        sl_lin = np.array([float(s) for _, s in lin])[:MAXLEN - 2]
        if len(toks) < 4:
            continue
        ids, am = build_ids_from_tokens(toks)
        with torch.no_grad():
            cls = int(model(input_ids=ids, attention_mask=am).logits.argmax(-1).item())
        sign = 1.0 if cls == 1 else -1.0                   # rank toward predicted class
        sl_pos = sign * sl_lin
        att = attention_received(ids, am)[1:1 + len(toks)]
        rnd = rng.random(len(toks))
        curves["SLALOM (linear)"].append(deletion_curve(ids, am, cls, sl_pos))
        curves["Attention"].append(deletion_curve(ids, am, cls, att))
        curves["Random"].append(deletion_curve(ids, am, cls, rnd))
        done += 1
    except Exception as e:
        print(f"  [review {nrev} skipped: {type(e).__name__}: {e}]")
print(f"faithfulness test completed on {done}/{len(texts)} reviews")

fig, ax = plt.subplots(figsize=(6.4, 4.4))
xs = np.linspace(0, 100, 11)
style = {"SLALOM (linear)": ("#0072B2", "o", "-"),
         "Attention": ("#D55E00", "s", "--"),
         "Random": ("#999999", "^", ":")}
summary = {}
for name, arr in curves.items():
    m = np.mean(arr, axis=0)
    summary[name] = float(np.trapz(m, xs) / 100.0)     # deletion AUC (lower=more faithful)
    c, mk, ls = style[name]
    ax.plot(xs, m, color=c, marker=mk, ls=ls, lw=2, label=f"{name} (AUC={summary[name]:.3f})")
ax.set_xlabel("% of tokens masked (most-relevant first)")
ax.set_ylabel("Predicted-class probability")
ax.set_title("E10 — MoRF deletion faithfulness on DistilBERT-IMDB\n"
             "(faster drop = more faithful)")
ax.legend(); ax.grid(alpha=0.3, ls="--")
plt.tight_layout()
plt.savefig("e10_slalom_faithfulness.pdf", bbox_inches="tight")
plt.savefig("e10_slalom_faithfulness.png", dpi=170, bbox_inches="tight")
np.save("e10_results.npy", {"deletion_auc": summary, "n_reviews": done})
print("\n[Part 2] mean MoRF-deletion AUC (lower = more faithful):")
for k, v in summary.items():
    print(f"  {k:>18s}: {v:.4f}")
print("wrote e10_slalom_faithfulness.pdf/.png and e10_results.npy")

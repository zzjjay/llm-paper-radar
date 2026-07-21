"""Generate the OAS+MBS algorithm-overview diagram for the interpretation README.

Draws a 3-stage flow (quantisation preprocessing -> unchanged MXFP4 GEMM ->
epilogue correction) plus a core-insight banner, showing HOW OAS+MBS works and WHY
it reuses the stock MXFP4 kernel unchanged. Output: images/algorithm_overview.png
(referenced by the "解读导航" table in README.md).

Generation rules follow paper-reprise's algorithm-overview convention: a committed,
re-runnable matplotlib script (not a throwaway), mathtext (fontset='cm') for correct
formula glyphs, FancyBboxPatch rounded blocks colour-coded by execution unit, a UNIFORM
vertical gap between every pair of boxes (incl. the bottom insight banner), and arrows
whose endpoints sit outside the boxes.

Run:  python images/gen_algorithm_overview.py   (needs matplotlib; e.g.
      /home/zhaolin/miniconda3/envs/quark_cuda/bin/python)
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "algorithm_overview.png")

fig, ax = plt.subplots(figsize=(9.0, 8.2))
ax.set_xlim(0, 9.0)
ax.set_ylim(0, 10.0)
ax.axis("off")
fig.patch.set_facecolor("white")

CX, W = 4.5, 8.4          # shared column centre + box width
GAP = 0.62                # uniform vertical gap between every pair of boxes


def rbox(cy, h, fc, ec, lw=1.8, z=2, pad=0.15):
    ax.add_patch(FancyBboxPatch((CX - W / 2, cy - h / 2), W, h,
                                boxstyle=f"round,pad={pad}", facecolor=fc,
                                edgecolor=ec, linewidth=lw, zorder=z))


def t(y, s, x=CX, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("zorder", 3)
    ax.text(x, y, s, **kw)


def arr(y_from, y_to):
    ax.annotate("", xy=(CX, y_to), xytext=(CX, y_from),
                arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", lw=2.2,
                                shrinkA=0, shrinkB=0, mutation_scale=20), zorder=4)


# ── Box geometry (top-down); centres/heights chosen for a uniform GAP ──────────
s0_top, s0_h = 9.60, 2.55
s0_cy = s0_top - s0_h / 2
s1_top = s0_top - s0_h - GAP
s1_h = 1.55
s1_cy = s1_top - s1_h / 2
s2_top = s1_top - s1_h - GAP
s2_h = 1.35
s2_cy = s2_top - s2_h / 2
ib_top = s2_top - s2_h - GAP
ib_h = 1.35
ib_cy = ib_top - ib_h / 2

# ── Stage 0: quantisation preprocessing (pure software) ───────────────────────
rbox(s0_cy, s0_h, "#EBF5FB", "#2980B9")
t(s0_top - 0.42, "Stage 0 · Quantisation preprocessing  (pure software)",
  fontsize=12, weight="bold", color="#1a5276")
t(s0_top - 1.02,
  r"① MBS  $\;c = 1 + m^{8}_{\rm MBS}\;$  ($8$-bit mantissa of $\,6/\alpha^{128}_{\max}$)"
  r"$\;\;\rightarrow\;\;$ pre-scale  $X' = X \times c$",
  fontsize=10.5, color="#1a3a55")
t(s0_top - 1.52,
  r"② OAS  $\;\mathrm{SF} = 2^{k}\;$  (map block $\alpha_{\max}$ to $(3.5,\,7]$: reference $7$, not $8$)",
  fontsize=10.5, color="#1a3a55")
t(s0_cy - s0_h / 2 + 0.42,
  r"Output:  E2M1 data $\,+\,$ E8M0 scale  $\;\leftarrow\;$  standard MXFP4 format",
  fontsize=10.5, weight="bold", color="#196f3d")

arr(s0_cy - s0_h / 2 - 0.06, s1_top + 0.06)

# ── Stage 1: MXFP4 GEMM on the Tensor Core (unchanged) ────────────────────────
rbox(s1_cy, s1_h, "#EAFAF1", "#27AE60", lw=2.5)
t(s1_top - 0.36, "Stage 1 · MXFP4 GEMM  (Tensor Core)",
  fontsize=12, weight="bold", color="#1e8449")
t(s1_cy - 0.02, r"FP4 $\times$ FP4 $\rightarrow$ FP32  accumulator",
  fontsize=10.5, color="#1a3a2a")
t(s1_cy - 0.44, "★  KERNEL UNCHANGED  ★", color="white", fontsize=10, weight="bold",
  zorder=5, bbox=dict(boxstyle="round,pad=0.25", facecolor="#27AE60", edgecolor="none"))

arr(s1_cy - s1_h / 2 - 0.06, s2_top + 0.06)

# ── Stage 2: epilogue correction on the Vector Core ───────────────────────────
rbox(s2_cy, s2_h, "#FEF9E7", "#E0A800")
t(s2_top - 0.38, "Stage 2 · Epilogue correction  (Vector Core)",
  fontsize=12, weight="bold", color="#7d6608")
t(s2_cy - 0.28,
  r"$C_{ij} \leftarrow C_{ij}\,\times\,\sigma_{A,i}\,\times\,\sigma_{B,j}\;$"
  r"  where $\;\sigma = 1/c\;$   (FMUL, $<1.6\%$ overhead)",
  fontsize=10.5, color="#5d4037")

# ── Core insight banner ───────────────────────────────────────────────────────
rbox(ib_cy, ib_h, "#EEF3FF", "#3a6cf4", lw=1.6, z=1, pad=0.12)
t(ib_top - 0.42,
  "Core insight:  OAS & MBS change only what happens BEFORE and AFTER the kernel.",
  fontsize=9.6, weight="bold", color="#1a2a7a")
t(ib_cy - 0.30,
  r"The kernel always receives standard E2M1 $+$ E8M0  —  no hardware change needed.",
  fontsize=9.3, color="#333")

plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")

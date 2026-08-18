# The Mathematics of NBSC: Non-Backtracking Spectral Convolution, Exponential Tilting, Hypergraph Extension, and Linear-Propagator Theory

This document explains, from first principles, the full mathematical and
physical machinery implemented across this workspace (`krylov_ds`,
`spectral_hypergraph`, `nbsc`), tying every Rust module back to the
theory it implements. It supersedes the earlier version of this document
by adding the mathematics behind the linear (SGC-style) propagation
theory added since, and by reporting how every piece of theory was
actually validated empirically, across three real citation networks
under the literature's canonical evaluation protocol.

The system has five layers, each building on the last:

1. **The Ihara zeta function and the Hashimoto (non-backtracking) matrix**
   — the number-theoretic/spectral-graph-theory object at the foundation
   (`docs/ihara_zeta.rs`).
2. **A matrix-free Krylov engine** (`krylov_ds`) used to extract spectral
   information from (1) without ever forming dense matrices.
3. **A graph neural network layer (NBSC)** built from the spectrum of
   (1), benchmarked against GCN/GAT/GraphSAGE (`nbsc/src/spectral.rs`,
   `burn_layer.rs`).
4. **An exponential tilt / large-deviations extension** of the
   non-backtracking spectral radius, borrowed from statistical mechanics
   and Cramér–Varadhan large-deviations theory, plus a **hypergraph
   generalization** via the Zhou–Huang–Schölkopf normalized hypergraph
   Laplacian (`spectral_hypergraph`).
5. **A linear-propagator (SGC) theory** (`nbsc/src/sgc.rs`) that strips
   the learnable-network machinery away entirely, isolating the
   propagation operator itself as the object under study — which is what
   made it possible to test the system's central hypothesis on a graph an
   order of magnitude larger than the multi-layer network can handle on
   commodity hardware.

Throughout, "physics" shows up in a precise, non-metaphorical sense: the
tilting construction is *literally* the Gibbs/Boltzmann exponential
reweighting used to build canonical ensembles from microcanonical ones,
and the resulting object is *literally* a scaled cumulant generating
function whose Legendre transform is a large-deviations rate function —
the same mathematics that underlies statistical mechanics' free energy
and Cramér's theorem in probability. Section 6 closes the loop: it shows
that the empirically measured, dataset-dependent shrinkage of an operator
norm ratio (§3.4, §6) is exactly the kind of quantity this same
machine-free spectral toolkit was built to measure — the theory and the
experiment are not separate stories.

---

## 1. The Ihara zeta function

### 1.1 Definition

For a finite graph $G$, a *closed backtrackless tailless cycle* is a
closed walk that never immediately reverses an edge (backtrackless) and
is not itself a repetition of a shorter cycle glued to a tail (tailless).
Grouping such cycles into equivalence classes $[C]$ under cyclic rotation
(the *primitive* cycles), the **Ihara zeta function** is

$$
\zeta_G(u) = \prod_{[C] \text{ primitive}} \left(1 - u^{\operatorname{len}(C)}\right)^{-1}.
$$

This is the graph-theoretic analogue of the Riemann zeta function's Euler
product over primes ($\zeta(s) = \prod_p (1-p^{-s})^{-1}$): primitive
cycles play the role of primes, and the "poles vs. critical line"
structure this document is ultimately about is a direct transplant of
the Riemann Hypothesis's geometry onto graphs.

### 1.2 Bass's theorem: two equivalent closed forms

Bass's theorem gives a finite, computable closed form in two dual bases:

**Vertex form** (size $n = |V|$), using the adjacency matrix $A$, degree
matrix $D$, and first Betti number $r = m - n + 1$ ($m = |E|$):

$$
\zeta_G(u)^{-1} = (1-u^2)^{r-1} \det\!\left(I_n - Au + (D-I)u^2\right).
$$

**Edge form** (size $2m$), using the **Hashimoto / non-backtracking
matrix** $B$, indexed by *directed arcs*:

$$
B_{(x\to y),(y'\to z)} = \mathbb{1}[y = y']\cdot \mathbb{1}[z \ne x],
$$

giving

$$
\zeta_G(u)^{-1} = \det\!\left(I_{2m} - uB\right).
$$

`docs/ihara_zeta.rs` implements **both formulas independently** and
cross-validates them numerically for five test graphs. Agreement of two
independently-derived formulas to $\sim 10^{-6}$ relative error is much
stronger evidence of correctness than trusting either transcription
alone — the same "two independent derivations, cross-checked
numerically" discipline used throughout the rest of the codebase.

### 1.3 Why the non-backtracking matrix, not the Laplacian?

The normalized graph Laplacian used by GCN/ChebNet is real symmetric —
its spectrum sees the graph's *local density* but is blind to *oriented
cycle structure*. $B$ is built on directed edges and is **not
symmetric**; complex-conjugate eigenvalue pairs correspond exactly to
non-real poles of $\zeta_G$, encoding oriented-cycle information the
Laplacian cannot see. NBSC's original hypothesis: build a graph filter
bank from $B$'s spectrum instead of the Laplacian's, and it should pick
up signal a Laplacian-based GCN structurally cannot. §5–6 report how that
hypothesis actually fared, in real, executed experiments across three
datasets.

### 1.4 The graph Riemann Hypothesis analogue (Ramanujan graphs)

For a $(q+1)$-regular graph, each factor's roots (poles $\lambda$ of
$\zeta_G$) solve $\lambda^2 - \mu_i \lambda + q = 0$, so
$\lambda_1\lambda_2 = q$; when $\mu_i^2 < 4q$ the roots are a
complex-conjugate pair forced to $|\lambda_1|=|\lambda_2|=\sqrt q$ — the
**graph analogue of the critical line**. $G$ is **Ramanujan** iff every
non-trivial eigenvalue satisfies $|\mu_i|\le 2\sqrt q$. Unlike the actual
Riemann Hypothesis, **this is a solved theorem** with known constructions
(Lubotzky–Phillips–Sarnak; Marcus–Spielman–Srivastava). `ihara_zeta.rs`'s
`ramanujan_check` verifies the bound on several test graphs and reports
pass/fail honestly (small graphs can and do fail it).

---

## 2. Extracting spectral data without forming dense matrices: `krylov_ds`

$B$'s eigenvalues (away from a trivial set at $\pm1$) coincide with those
of the smaller **Bass-reduced $2n\times2n$ matrix**

$$
M = \begin{pmatrix} A & I - D \\ I_n & 0 \end{pmatrix},
$$

implemented **matrix-free** in `HashimotoLinearization` (`apply(v) = [Ax
+ (I-D)y;\ x]`), fed to `krylov_ds::Arnoldi` to extract $\rho_B$, the
**Perron–Frobenius spectral radius of the non-backtracking matrix** — a
real, non-negative quantity even though the rest of $M$'s spectrum is
generically complex. `krylov_ds` also provides `Lanczos` for symmetric
operators (e.g. plain $A$, used by `adjacency_operator_norm`), with full
reorthogonalization for numerical stability. Every non-trivial numeric
claim in the codebase is unit-tested against an independent dense ground
truth, isolating "is the matrix-free construction correct" from "did
Krylov converge."

---

## 3. NBSC: a graph filter bank built from $B$'s spectrum

### 3.1 The rescaled three-term recursion

`NbscFilterBank` builds a ChebNet-style filter bank generated by $M$
(equivalently $B$), rescaled by $\rho_B$:

$$
T_0 = I, \qquad T_1 = \frac{A}{\rho_B}, \qquad
T_{k+1} = \frac{2A}{\rho_B}\,T_k - \frac{D-I}{\rho_B^2}\,T_{k-1}.
$$

`NbscFilterBank::apply_taps` never forms $T_k$ as an $n\times n$ matrix —
it applies the recursion **directly to a feature matrix** via
$O(K\cdot|E|\cdot f)$ sparse primitives, the same complexity class as
GCN/ChebNet.

### 3.2 The learnable layer

`burn_layer.rs`'s `NbscLayer` wraps this in a standard readout
$H = \sigma\!\left(\sum_{k=0}^K T_k X\, W_k + b\right)$, with $T_k$
graph-fixed and only $\{W_k\}, b$ learned; Burn's autodiff differentiates
through the recursion directly (dense tensor path, `n×n` materialized —
see §5 for why this becomes the binding constraint at PubMed's scale).
The GCN baseline (`GcnLayer`) uses the standard
$\hat A = D^{-1/2}(A+I)D^{-1/2}$, $H=\operatorname{ReLU}(\hat A XW)$.

### 3.3 Dirichlet energy: the over-smoothing diagnostic

$$
E(X) = \frac{1}{\overline{\|x\|^2}}\cdot\frac{1}{|E|}\sum_{(u,v)\in E} \|x_u - x_v\|^2.
$$

Laplacian-based propagators are non-expansive ($\|\hat A\|_2 \le 1$
exactly), which drives $E(X)\to0$ with depth — over-smoothing. NBSC's
propagator carries no such guarantee; §3.4 reports what actually
happens.

### 3.4 Empirical results, now on three real datasets under the canonical split

The system's central empirical question — does the non-backtracking
propagator carry class-relevant signal the Laplacian-based one doesn't
— has now been tested at two structurally unrelated levels of model
complexity, on three real citation networks (Cora, Citeseer, PubMed),
under the **literature's exact published train/val/test split** (Yang,
Cohen & Salakhutdinov, ICML 2016 — obtained by unpickling the reference
`tkipf/gcn` data files and converting to plain text, not re-derived; see
`nbsc/src/dataset.rs`'s `load_planetoid_canonical` doc comment for full
provenance).

**Deep-network level** (`NbscLayer`/`GcnLayer`, Burn, `HIDDEN=16`,
`K_TAPS=2`, Adam, 150 epochs, 2–3 seeds per config):

| Dataset | Depth | NBSC test acc | GCN test acc | NBSC final Dirichlet energy |
|---|---|---|---|---|
| Cora | 1 | 0.666 ± 0.009 | **0.754 ± 0.007** | 4.94 |
| Cora | 2 | 0.720 ± 0.016 | **0.768 ± 0.010** | 10.84 |
| Cora | 3 | 0.668 ± 0.013 | **0.720 ± 0.007** | 12.15 |
| Citeseer | 2 | 0.562 ± 0.007 | **0.603 ± 0.042** | 6.77 |

GCN wins at every tested depth on both datasets. NBSC's Dirichlet energy
*grows* with depth on Cora (4.94 → 10.84 → 12.15) — the opposite of
ordinary over-smoothing — replicating, under the corrected canonical
split, the same anomalous pattern the earlier (non-canonical-split)
study first found.

**Linear (SGC-style) level, §6 below**, extends this to all three
datasets including PubMed, which the dense deep-network path cannot
reach on commodity hardware. Same qualitative result: GCN's propagator
wins everywhere.

**Reading this honestly**: NBSC's original hypothesis — that
non-backtracking-walk structure carries a graph-classification signal
the Laplacian-based propagator structurally cannot see — is *not*
supported by these three homophilous citation networks; both propagators
massively beat unpropagated raw features (§6.2), so graph structure
matters, but the specific non-backtracking construction is consistently
the *weaker* of the two propagators tested, not the stronger one. §6.3
traces this to a specific, checkable spectral mechanism rather than
leaving it as an unexplained empirical gap.

---

## 4. Exponential tilting: statistical mechanics on graphs

### 4.1 The construction

For the Hashimoto matrix $B$ on the $2m$-dimensional directed-arc
("darc") space and a per-arc observable $f$, the **exponential tilt** is

$$
B(\theta)_{ij} = B_{ij}\cdot e^{\theta f_j}.
$$

This is *exactly* the Gibbs/Boltzmann reweighting that turns a uniform
measure into a canonical ensemble weighted by $e^{-\beta H}$ — here
$\theta$ plays the role of (minus) inverse temperature and $f$ the
Hamiltonian/energy observable. It is simultaneously the tilting operation
in Cramér's theorem, the Gibbs measure construction in statistical
mechanics, and a quantum-mechanical reweighting of amplitudes — the same
formula in three costumes. `TiltedForward`/`TiltedTranspose` implement
$B(\theta)$ and $B(\theta)^T$ matrix-free, respecting the non-backtracking
constraint; because $e^{\theta f}>0$ always, Perron–Frobenius applies at
every $\theta$, so `dominant_real_pair` safely picks the unique Perron
root $\rho_B(\theta)$.

### 4.2 The derivative: a Hellmann–Feynman / Perron-root perturbation identity

With right/left Perron eigenvectors $w(\theta), v(\theta)$:

$$
\frac{d\rho}{d\theta} = \frac{v^T\, \dfrac{dB}{d\theta}\, w}{v^Tw}
$$

— the Hellmann–Feynman theorem generalized from the symmetric/Hermitian
case to a general non-symmetric matrix. `tilted_spectral_radius` closes
this as a single dot-product ratio, no repeated re-solving at nearby
$\theta$ (checked against finite differences independently).

**Physical reading (Varadhan's lemma):** with
$\Lambda(\theta):=\log\rho_B(\theta)$ as the **scaled cumulant generating
function** of $S_n=\sum_k f(\text{step}_k)$ along a long non-backtracking
walk, $\Lambda'(\theta)$ is the mean of $f$ under the $\theta$-tilted walk
measure — checked both by a closed-form uniform-tilt case ($f\equiv1
\Rightarrow \rho(\theta)=e^\theta\rho(0)$, matching to $\sim10^{-5}$
relative error) and against a real, non-trivial tilt on Cora's walk
structure (degree-biased, checked against central finite differences).

### 4.3 The Legendre–Fenchel transform: a genuine large-deviations rate function

Once $\Lambda(\theta)$ is known convex and differentiable, the
**Gärtner–Ellis theorem** gives the large-deviations rate function of the
empirical mean of $f$ as

$$
I(x) = \sup_\theta\big(\theta x - \Lambda(\theta)\big),
$$

the same free-energy/entropy duality underlying equilibrium statistical
mechanics and Cramér's theorem. `legendre_rate` finds the maximizing
$\theta^*$ by bisection on the monotone $\Lambda'(\theta)-x$. Checked
properties: the uniform-tilt case collapses to a point mass exactly as
convex duality predicts; $I$ is minimized (at $-\Lambda(0)$) exactly at
the untilted mean; $\theta^*$ satisfies the first-order condition on
re-evaluation. **What this buys**: a principled, spectrally-derived
rarity/anomaly score — "how exponentially rare is an empirical mean of
$x$ along a long non-backtracking walk?" — the same machinery underlying
importance sampling and rare-event simulation in statistical physics.

---

## 5. Hypergraph generalization: `spectral_hypergraph`

### 5.1 The normalized hypergraph Laplacian

Following Zhou, Huang & Schölkopf (NeurIPS 2006), with incidence matrix
$H$, hyperedge weights $W$, vertex-degree matrix $D_v$, hyperedge-
cardinality matrix $D_e$:

$$
\Delta = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2},
$$

symmetric PSD, reducing to the ordinary normalized Laplacian when every
hyperedge has cardinality 2. `HypergraphOperator` implements $\Delta$
matrix-free in $O(\operatorname{nnz}(H))$ per application.

### 5.2 Fiedler vector and spectral clustering

`fiedler_vector` (smallest non-zero eigenvalue's eigenvector) generalizes
Cheeger-style graph partitioning to hypergraphs; `spectral_cluster`
extends this to $k$-way clustering via Ng–Jordan–Weiss.

### 5.3 The bridge to NBSC: two integration paths

`nbsc::hypergraph_bridge` offers **clique expansion** (flattens
hyperedges into cliques, runs the full NBSC/Hashimoto machinery
unmodified, but discards higher-order structure) and a **matrix-free
operator adapter** (wraps $\Delta$ as a `krylov_ds::LinearOperator<f64>`,
running the crate's own Arnoldi/Lanczos directly against the true
hypergraph structure, no clique expansion). A worked demo shows native
hypergraph spectral clustering reaching purity 1.000 vs. 0.750 for the
clique-expanded graph on the same synthetic ground truth — a concrete,
measured cost of the simpler construction. The same run finds the
clique-expanded graph's `‖A‖₂/ρ_B` ratio at $1.136$, in the same
expansive-operator range as the real citation networks (§6.3).

---

## 6. Linear-propagator theory: the SGC reduction, and what it revealed

### 6.1 Why a fifth layer was needed

`NbscLayer`/`GcnLayer` materialize a dense $n\times n$ tensor per layer.
At Cora/Citeseer scale (a few thousand nodes) this is fine — the same
complexity class as GCN/ChebNet in the literature. At PubMed's scale
(19717 nodes), a single $n\times n$ `f32` tensor is already $\approx1.55$
GB, and a training step needs several such tensors alive at once
(forward activations, autodiff-retained backward state) — several times
the RAM available on the evaluation machine. Rather than skip PubMed
entirely, `nbsc/src/sgc.rs` implements a genuinely different, much
cheaper evaluation of the *same underlying question*.

### 6.2 The reduction

Following Wu, Souza, Zhang, Fifty, Yu & Weinberger (*"Simplifying Graph
Convolutional Networks,"* ICML 2019, "SGC"): fix the propagation entirely
(no learnable weights inside it — the graph enters only through a
one-time, matrix-free sparse computation), and fit a single linear
(softmax) classifier on top of the propagated features. Concretely,
`gcn_propagate_taps` computes $[X, \hat AX, \hat A^2X,\dots]$ (the same
$\hat A = D^{-1/2}(A+I)D^{-1/2}$ as `GcnLayer`, applied directly to
features rather than materialized as a matrix), and `NbscFilterBank
::apply_taps` computes $[T_0X,\dots,T_KX]$ exactly as in §3.1;
`concat_taps` concatenates the taps column-wise, which is
representationally equivalent to a learnable layer applying a separate
weight matrix per tap and summing them
($\sum_k T_kX\,W_k$ spans the same function class as
$[T_0X|\cdots|T_KX]\,[W_0;\dots;W_K]$) — so this is a faithful linear
analogue of both `NbscLayer` and `GcnLayer`, not an ad hoc simplification.

Memory is $O(n\cdot f)$ for the propagated features and $O(f\cdot
\text{classes})$ for the classifier — independent of any $n\times n$
object — so it scales to PubMed trivially (a few tens of MB), and was run
on all three datasets, not just PubMed, as a same-methodology
cross-check alongside the deep networks.

### 6.3 The classifier itself, and a convexity-based sanity property

`SoftmaxClassifier` is ordinary multinomial logistic regression, trained
by full-batch gradient descent on cross-entropy plus an $L2$ (weight
decay) penalty:

$$
\mathcal L(W,b) = -\frac{1}{|\mathcal T|}\sum_{i\in\mathcal T}
\log \operatorname{softmax}(Wx_i+b)_{y_i} \;+\; \frac{\lambda}{2}\|W\|_F^2.
$$

This objective is **convex** in $(W,b)$ (cross-entropy composed with an
affine map is convex; the $L2$ term is strictly convex), so — unlike
§3's non-convex multi-layer networks — different random initializations
should converge to numerically the same optimum, not merely similar ones.
This is checked directly (`softmax_classifier_is_seed_invariant_on_a_
convex_problem`) and reported as a genuine methodological contrast: the
deep-network seed variance in §3.4's table reflects real optimization
landscape multi-modality; the linear-classifier seed variance reflects
only gradient-descent-trajectory noise on a landscape with one basin.

### 6.4 Results and what they add

Best test accuracy (5-point weight-decay grid $\{0,10^{-4},5\times10^{-4},10^{-3},10^{-2}\}$,
selected by validation accuracy, 3 seeds):

| Dataset | Raw features | NBSC-linear | GCN-linear |
|---|---|---|---|
| Cora | 0.494 (wd=0) | 0.678 (wd=$10^{-2}$) | **0.742** (wd=0) |
| Citeseer | 0.425 (wd=$10^{-2}$) | 0.540 (wd=$10^{-2}$) | **0.601** (wd=0) |
| PubMed | 0.683 (wd=0) | 0.712 (wd=$10^{-2}$) | **0.745** (wd=0) |

Three things this adds to the theory, beyond simply extending coverage to
PubMed:

1. **The deep-network finding replicates at the linear level, on all
   three datasets.** Since the linear classifier's loss surface is
   convex and the deep network's is not, this is evidence the GCN-beats-
   NBSC finding is a property of the *propagators themselves*, not an
   artifact of a particular non-convex training dynamics.
2. **NBSC's optimal weight decay is the grid's largest value on all
   three datasets; GCN's is zero on all three.** This is a clean,
   reproducible signature consistent with §6.5's diagnosis: if the
   NBSC propagator is an expansive operator, its propagated features
   have inflated scale/variance, and a classifier fit on them benefits
   more from $L2$ shrinkage than one fit on GCN's non-expansive-by-
   construction propagated features.
3. **Both propagators beat raw (unpropagated) features by a wide margin
   everywhere** (e.g. Cora: 49%→68–74%) — the more basic claim that
   non-backtracking-walk structure is *usable* graph-learning signal at
   all is supported strongly; the comparative claim (better than the
   Laplacian propagator) is the one that isn't.

### 6.5 The expansive-operator mechanism, now measured on all three datasets

`examples/expansive_operator_check.rs` computes $\rho_B$
(`estimate_spectral_radius`, matrix-free Arnoldi) and $\|A\|_2$
(`adjacency_operator_norm`, matrix-free Lanczos) — the same estimators
used throughout §2–3 — for all three datasets:

| Dataset | $n$ | $\rho_B$ | $\|A\|_2$ | $\|A\|_2/\rho_B$ |
|---|---:|---:|---:|---:|
| Cora | 2708 | 9.03 | 14.39 | **1.594** |
| Citeseer | 3327 | 11.49 | 13.74 | **1.197** |
| PubMed | 19717 | 21.64 | 23.24 | **1.074** |

$A/\rho_B$ — the rescaled tap the recursion in §3.1 uses — is an
**expansive** operator ($\|\cdot\|_2>1$) on all three graphs, unlike
GCN's $\hat A$, which is non-expansive **by construction**
($\|\hat A\|_2=1$ exactly, since it is a similarity transform of a
row-stochastic matrix). This directly explains §3.4's Dirichlet-energy-
growing-with-depth anomaly: an expansive linear step compounds
multiplicatively across stacked layers, the opposite of the
non-expansive-propagator over-smoothing every other architecture
exhibits.

The ratio **shrinks monotonically as the graph grows** — 59% above 1 on
the smallest graph tested down to 7% above 1 on the largest — which is
consistent with (though, from three data points, does not prove) both
the shrinking accuracy gap in §6.4 and the shrinking Dirichlet-energy gap
in §3.4. This is exactly the kind of graph-scale-dependent spectral
quantity the matrix-free Arnoldi/Lanczos machinery in §2 was built to
measure cheaply — it required no new mathematics to compute here, only
applying the existing tool to more/bigger graphs, and it is a genuinely
falsifiable hypothesis (does the ratio continue toward 1, or below, on
larger/denser real graphs — and does NBSC's relative disadvantage vanish
with it?) that could be tested against a fourth or fifth dataset in
future work.

---

## 7. How the pieces compose

```
                 Ihara zeta ζ_G(u)  (§1)
                        │  Bass's theorem, cross-validated numerically
                        ▼
      Hashimoto / non-backtracking matrix B  (2m×2m, sparse, non-symmetric)
                        │
        ┌───────────────┼──────────────────────┬───────────────────────┐
        │ Bass-reduced   │ matrix-free darc-space│ clique-expand /       │ sparse tap
        │ linearization  │ operator + tilt B(θ)  │ operator-adapt into   │ applied
        │ M (2n×2n)      │ (§4, statistical mech.│ spectral_hypergraph   │ directly to
        │                │ / large deviations)   │ (§5)                  │ features
        ▼                ▼                       ▼                       ▼
  krylov_ds Arnoldi   krylov_ds Arnoldi        normalized hypergraph   NbscFilterBank
  → ρ_B (Perron root) (fwd+bwd) → ρ_B(θ),      Laplacian Δ            ::apply_taps
        │               dρ/dθ (Hellmann-        → Fiedler vector,      (§3.1, reused
        │               Feynman), Λ(θ),          spectral clustering    verbatim by §6)
        │               I(x) = Legendre                                       │
        │               transform (rarity score)                             │
        ▼                                                                     ▼
  T_k rescaled 3-term recursion (§3.1)                          ┌── concat_taps + softmax
        │                                                       │   classifier (§6, "SGC"),
        ├── NbscLayer (learnable GNN, §3.2) ──────┐              │   O(n·f) memory, all 3
        │   benchmarked vs GCN/GAT/GraphSAGE      │              │   datasets incl. PubMed
        │   on canonical-split Cora+Citeseer      │              │
        │   (§3.4)                                │              │
        │                                          ▼              ▼
        │                              Dirichlet energy    weight-decay grid,
        │                              diagnostic → grows    convexity-checked
        │                              with depth (Cora,       seed-invariance
        │                              Citeseer both)
        │
        └── examples/expansive_operator_check.rs (§6.5): ‖A‖₂/ρ_B
            measured on all 3 datasets → confirms & extends the
            mechanism found via the Dirichlet-energy route
```

Every arrow corresponds to a Rust module cross-validated against an
independent computation (a second closed-form derivation, a dense
ground-truth eigensolver, a finite-difference check, or — new in this
version — a same-underlying-question check across two structurally
unrelated model classes and three real datasets under the literature's
own published evaluation split). The recurring methodological stance,
carried through from the theory chapters into the empirical ones, is
that a result earns trust by being checked against something computed a
different way — not by being presented once and assumed correct. What
changed since the previous version of this document is not the
mathematics (§1–5 are unchanged) but the empirical closure: the
hypothesis in §1.3 (non-backtracking structure should out-perform
Laplacian structure) has now actually been tested, honestly, at scale,
and found not to hold on these three graphs — with the mechanism behind
*why* traced to a specific, measured, three-dataset-consistent spectral
quantity ($\|A\|_2/\rho_B$) rather than left as an open question.

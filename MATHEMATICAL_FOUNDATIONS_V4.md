# Mathematical Foundations for the Completed Version 4 of the `discrete quantum gravity (DQG)` × `nbsc`/`spectral_hypergraph` System

This document proves, from first principles, the theorems the codebase implements
and checks numerically. It is organized so each proof matches a specific runtime
cross-check in `LIVE_RUN_OUTPUT.txt`. Where a claim is genuinely open (not a
theorem anyone has proved), that is stated explicitly rather than papered over —
consistent with the honesty convention already established in `spectral_dqg`'s
own README (see its "What's deliberately not claimed" section).

**Scope note up front, stated plainly:** most of what follows are proofs of
*classical, established* theorems (Ihara 1966, Bass 1992, Alon–Boppana 1986/1991,
Kesten 1959/McKay 1981). One section (§9) addresses a genuinely open research
question, and shows it to be open, not proved true or false. One derivation (§6)
is a leading-order heuristic reproduced from the literature (Decelle–Krzakala–
Moore–Zdeborová 2011; Krzakala et al. 2013; Angelini et al. 2015), not re-derived
rigorously here — the module docstring in `hsbm_detectability.rs` already flags
this distinction and this document preserves it.

---

## 0. Setup and notation

Let `G = (V, E)` be a finite, simple, connected graph with `n = |V|` vertices and
`m = |E|` edges, minimum degree `≥ 2` (no degree-1 dangling vertices, which would
cause trivial zeta factors). Replace each undirected edge `{u,v}` by two directed
**arcs** `u→v` and `v→u`; write `\bar e` for the reverse of arc `e`. Let
`𝒜 = 𝒜(G)` be the set of `2m` arcs.

**Definition (Hashimoto / non-backtracking matrix).** `B ∈ {0,1}^{𝒜×𝒜}` is defined by

```
B_{e,f} = 1   iff   f follows e (head(e) = tail(f))  and  f ≠ \bar e
```

i.e. the walk continues at the shared vertex without immediately reversing.

**Definition (closed non-backtracking walk / prime cycle).** A closed
non-backtracking walk of length `k` is a cyclic sequence of arcs
`e_0, e_1, …, e_{k-1}` with `B_{e_i, e_{i+1 \bmod k}} = 1` for all `i`
(note the wrap-around condition at `i = k-1`). It is **primitive** if it is not
a proper power of a shorter closed non-backtracking walk (as a cyclic sequence),
and a **prime** `[C]` is an equivalence class of primitive closed non-backtracking
walks under cyclic rotation.

**Definition (Ihara zeta function).**
```
Z_G(u) = ∏_{[C] prime} (1 - u^{ℓ(C)})^{-1}
```
a formal power series in `u`, convergent for `|u|` small.

This matches `spectral_dqg/src/nonbacktracking.rs` and `ihara_zeta.rs` exactly.

---

## 1. Ihara's theorem: `Z_G(u)^{-1} = det(I - uB)`

**Theorem 1.** For `|u|` small enough that both sides converge,
`Z_G(u)^{-1} = det(I - uB)`.

**Proof.**

*Step 1 — trace-log identity.* For a matrix `M` with spectral radius `< 1/|u|`,
```
log det(I - uM) = Tr log(I - uM) = -Σ_{k≥1} Tr(M^k) u^k / k
```
(the standard matrix logarithm expansion, valid termwise since `Tr` is linear
and continuous on the convergent power series `log(I-uM) = -Σ (uM)^k/k`). Apply
this with `M = B`:
```
-log det(I - uB) = Σ_{k≥1} Tr(B^k) u^k / k.        (1.1)
```

*Step 2 — combinatorial meaning of `Tr(B^k)`.* By definition of matrix powers,
```
Tr(B^k) = Σ_{e_0,…,e_{k-1} ∈ 𝒜} B_{e_0,e_1} B_{e_1,e_2} ⋯ B_{e_{k-1},e_0}
```
which counts exactly the closed non-backtracking arc-walks of length `k`
(including the wrap-around non-backtracking condition at the seam — this is
precisely the subtlety `spectral_dqg`'s own combinatorial cross-check in
`nonbacktracking.rs` was built to catch, and the code comment there documents a
real bug this caught during development). So `Tr(B^k)` is a raw count of closed
walks, not yet grouped by equivalence class.

*Step 3 — grouping by primitive cycles.* Every closed non-backtracking walk of
length `k` is, as a cyclic sequence, the `j`-fold repetition of a unique
primitive closed non-backtracking walk of length `d = k/j` for some divisor `d`
of `k`. A primitive walk of length `d`, together with its `d` cyclic rotations,
gives exactly `d` distinct arc-sequences (all counted separately by `Tr(B^k)`,
since `Tr` sums over all starting points `e_0`), all representing the same prime
`[C]`. Hence
```
Tr(B^k) = Σ_{[C] prime : ℓ(C) | k} ℓ(C).
```

*Step 4 — resum.* Substitute into (1.1) and exchange the order of summation
(justified by absolute convergence for `|u|` small):
```
-log det(I-uB) = Σ_{k≥1} u^k/k · Σ_{[C]: ℓ(C)|k} ℓ(C)
                = Σ_{[C]} Σ_{j≥1} u^{jℓ(C)}/(jℓ(C)) · ℓ(C)     (k = jℓ(C))
                = Σ_{[C]} Σ_{j≥1} u^{jℓ(C)}/j
                = -Σ_{[C]} log(1 - u^{ℓ(C)})
                = log ∏_{[C]} (1-u^{ℓ(C)})^{-1} = log Z_G(u).
```
Hence `det(I-uB) = Z_G(u)^{-1}`. ∎

This is exactly `spectral_dqg::ihara_zeta::zeta_inverse_via_b`, and step 3's
`Tr(B^k)` identity is exactly what `nonbacktracking.rs::trace_bk` /
`count_closed_nbt_walks_bruteforce` cross-check against each other — the live run
confirms `Tr(B^4) = 280` matches the independent DFS count exactly (§4 below
explains why `Tr(B^1)=Tr(B^2)=Tr(B^3)=0` on the factor graph, which is *also* a
theorem, not a coincidence).

---

## 2. Sylvester's determinant identity (lemma used in §3)

**Lemma 2.** For `A` a `p×q` matrix and `B` a `q×p` matrix,
`det(I_p - AB) = det(I_q - BA)`.

**Proof.** Let `M = [[I_p, A], [B, I_q]]` be the `(p+q)×(p+q)` block matrix.
Compute `det M` two ways, by block-triangularizing it from each side.

*Eliminating the bottom-left block* (subtract `B` times block-row 1 from
block-row 2 — a determinant-preserving row operation):
```
[ I_p   A  ]   [ I_p      A     ]
[ B    I_q ] → [ 0    I_q - BA  ]
```
which is block upper-triangular, so `det M = det(I_p) · det(I_q - BA) = det(I_q - BA)`.

*Eliminating the top-right block instead* (subtract `A` times block-row 2 from
block-row 1):
```
[ I_p   A  ]   [ I_p - AB    0  ]
[ B    I_q ] → [    B       I_q ]
```
which is block lower-triangular, so `det M = det(I_p - AB) · det(I_q) = det(I_p - AB)`.

Both computations equal `det M`, so `det(I_p - AB) = det(I_q - BA)`. ∎

---

## 3. Bass's determinant formula: `det(I-uB) = (1-u^2)^{m-n} det(I - uA + u^2(D-I))`

Define the `n × 2m` **start** and **end** incidence matrices:
```
S_{v,e} = 1 iff arc e starts at v,      T_{v,e} = 1 iff arc e ends at v.
```
and the `2m × 2m` **reversal** matrix `J_{e,f} = 1` iff `f = \bar e`.

**Facts (immediate from the definitions):**
- `J^2 = I` (reversal is an involution), and since no arc is its own reverse in a
  simple graph, `J` decomposes into `m` disjoint transpositions.
- `S T^T = A` (the `n×n` adjacency matrix): `(ST^T)_{u,v} = #\{$arcs starting at
  `u`, ending at `v`$\}$` = `A_{u,v}` for a simple graph.
- `T T^T = S S^T = D` (diagonal degree matrix): `(TT^T)_{v,w}` counts arcs ending
  at both `v` and `w`, which is `0` unless `v=w`, giving `deg(v)`.
- `S J = T` and `T J = S`: reversing an arc swaps its start and end vertex.
- `B = T^T S - J`: `(T^T S)_{e,f} = Σ_v T_{v,e} S_{v,f}` is `1` iff `e` ends where
  `f` starts (i.e. `f` follows `e`, backtracking allowed); subtracting `J`
  removes exactly the immediate-backtrack case, matching `B`'s definition.

**Theorem 3 (Bass 1992; graph-theoretic proof via Stark–Terras 1996).**
```
det(I_{2m} - uB) = (1-u^2)^{m-n} · det(I_n - uA + u^2(D - I_n)).
```

**Proof.**

*Step 1.* `I - uB = I + uJ - uT^T S = (I+uJ) - u T^T S`. Since `J` has
eigenvalues `±1` (each transposition contributing one `+1` and one `-1`),
`I + uJ` is invertible for `|u| < 1`, and because `J^2=I`,
```
(I+uJ)(I-uJ) = I - u^2 J^2 = (1-u^2) I   ⟹   (I+uJ)^{-1} = (I-uJ)/(1-u^2).
```

*Step 2.* Factor out `(I+uJ)`:
```
det(I-uB) = det(I+uJ) · det( I - u(I+uJ)^{-1} T^T S ).
```

*Step 3 — apply Lemma 2* with `p=2m`, `q=n`, `A_{\text{lemma}} = u(I+uJ)^{-1}T^T`
(`2m×n`), `B_{\text{lemma}} = S` (`n×2m`):
```
det( I_{2m} - u(I+uJ)^{-1}T^T S ) = det( I_n - u S (I+uJ)^{-1} T^T ).
```

*Step 4 — evaluate `S(I+uJ)^{-1}T^T`.* Using Step 1's closed form,
```
S(I+uJ)^{-1}T^T = [S(I-uJ)T^T]/(1-u^2) = [ST^T - u(SJ)T^T]/(1-u^2)
                = [A - u T T^T]/(1-u^2) = (A - uD)/(1-u^2),
```
using `ST^T=A`, `SJ=T`, `TT^T=D` from the Facts above.

*Step 5 — assemble.*
```
I_n - u·(A-uD)/(1-u^2) = [(1-u^2)I_n - uA + u^2 D]/(1-u^2) = [I_n - uA + u^2(D-I_n)]/(1-u^2),
```
so
```
det(I_n - uS(I+uJ)^{-1}T^T) = (1-u^2)^{-n} det(I_n - uA + u^2(D-I_n)).
```

*Step 6 — `det(I+uJ)`.* `J` is `m` disjoint transpositions; on each 2-dimensional
swap subspace `J` has eigenvalues `+1, -1`, so `I+uJ` restricted there has
eigenvalues `1+u, 1-u`, contributing `(1+u)(1-u) = 1-u^2` per transposition:
```
det(I+uJ) = (1-u^2)^m.
```

*Step 7 — combine.* `det(I-uB) = (1-u^2)^m · (1-u^2)^{-n} · det(I_n-uA+u^2(D-I_n))
= (1-u^2)^{m-n} det(I_n - uA + u^2(D-I_n))`. ∎

This is exactly `spectral_dqg::ihara_zeta::zeta_inverse_via_bass`, and the two
formulas' agreement to `~3e-13` in the live run (both on the factor graph and on
the clique expansion) is a direct numerical confirmation of this theorem, not an
independent fact — a bug in either implementation would show up as a nonzero
discrepancy, per the module's own stated design philosophy.

---

## 4. `Tr(B^k) = 0` for `k ∈ {1,2,3}` on the factor graph (explains the live run)

**Claim.** For *any* simple graph, `Tr(B^1) = Tr(B^2) = 0`. On a **bipartite**
graph (such as the vertex/hyperedge factor graph), additionally `Tr(B^k) = 0`
for every odd `k`.

**Proof.**
- `Tr(B) = Σ_e B_{e,e}`. `B_{e,e}=1` would require `e` to follow itself without
  reversing, i.e. `head(e) = tail(e)` — impossible in a simple graph (no
  self-loops). So `Tr(B)=0`.
- `Tr(B^2) = Σ_{e,f} B_{e,f}B_{f,e}`. This requires `f` to follow `e`
  (`f ≠ \bar e`) *and* `e` to follow `f` (`e ≠ \bar f`) at the *same* shared
  vertex pair — but if `e = u→v` and `f` follows it landing at some `w ≠ u`,
  then `f = v→w`; for `e` to follow `f` we'd need `e` to start where `f` ends,
  i.e. at `w`, but `e` starts at `u ≠ w`. No such pair exists, so `Tr(B^2)=0`.
  This holds for *every* simple graph, not just this one — it is the algebraic
  statement that "there is no closed non-backtracking walk of length 2," which
  is obvious combinatorially (going out and immediately coming back is exactly
  the forbidden backtrack) and confirmed here algebraically.
- On a bipartite graph with parts `X` (vertex-nodes), `Y` (hyperedge-nodes),
  every arc goes `X→Y` or `Y→X`, so following `k` arcs in a closed walk changes
  which part you're in `k` times and must return to the start — forcing `k`
  even. Hence `Tr(B^k)=0` for odd `k` by Step 2 of Theorem 1 (no closed walks of
  odd length exist to count).

This is exactly the `k=1,2,3 → 0, k=4 → 280` pattern in
`LIVE_RUN_OUTPUT.txt`, and the `k=4` value's exact match against the independent
brute-force DFS count is the live numerical instance of Theorem 1, Step 2. ∎

---

## 5. The hypergraph non-backtracking operator via the bipartite factor graph

**Construction (Angelini, Caltagirone, Krzakala & Zdeborová 2015).** Given a
hypergraph `H=(V,E)`, form the bipartite **factor graph** `𝔉(H)` on
`V ⊔ E` with an edge for every incidence `(v,e)`, `v∈e`. Apply the *ordinary*
Hashimoto construction of §0 to `𝔉(H)`.

**Proposition 5.** This construction is well-defined and lossless: `H` is
recoverable from `𝔉(H)` up to vertex/hyperedge labels (it is exactly the
incidence structure of `H`, stored as a graph).

**Proof.** `𝔉(H)` is bipartite with parts `V, E`; an edge `(v,e)` exists iff
`v ∈ e`. This is precisely the incidence relation, so `H`'s hyperedge-membership
function `e ↦ \{v : v∈e\}` is recovered as `e`'s neighborhood in `𝔉(H)`. ∎

Because `𝔉(H)` is an ordinary (bipartite) simple graph, **Theorems 1 and 3 apply
to it without modification** — this is the entire mathematical justification for
why `dqg_hsbm_bridge::bridge::factor_graph_to_weighted_graph` requires no new
theory, only a data-structure adapter: correctness was already established by
§1–§3 for *any* graph, and `𝔉(H)` is a graph.

**What is *not* claimed:** `chaos_test/src/main.rs` explores a second, distinct
operator `B_H` indexed directly on incident pairs `(i,e)`, `i∈e`, rather than on
`𝔉(H)`'s arcs. This document does **not** assert `B_H` is conjugate to, or
spectrally equivalent to, the bipartite-factor-graph Hashimoto operator used in
the live run — that equivalence was not checked here, and asserting it without
proof would be exactly the kind of unverified claim this document is trying to
avoid. Treat the two as independent constructions unless/until that equivalence
is proved or disproved directly.

---

## 6. The Kesten–Stigum-type detectability threshold `λ_c = 1/√(c(k-1))`

This section reproduces `nbsc::hsbm_detectability`'s own derivation in more
formal language, **stated explicitly as a heuristic, not a rigorous proof**
(exactly as the module docstring already says). A fully rigorous proof of
optimality (that *no* algorithm, spectral or otherwise, beats this threshold, and
that the non-backtracking spectrum achieves it) is due to Krzakala et al. (PNAS
2013) for `k=2` and Angelini et al. (2015) for general `k`, via the cavity
method / density-evolution analysis of belief propagation — reproducing that
proof is out of scope here.

**Setup.** `𝔉(H)` for a `q`-community, `k`-uniform, average-degree-`c` HSBM is,
in the sparse limit `n→∞` with `c` fixed, locally tree-like (a standard
consequence of the configuration-model-style construction: short cycles have
vanishing probability as `n→∞` for fixed average degree). On this local tree, a
non-backtracking step alternates two move types:

- **Through a hyperedge** (vertex-node → hyperedge-node → vertex-node):
  deterministic branching factor `k-1` (any of the `k-1` *other* members).
- **Through a vertex** (hyperedge-node → vertex-node → hyperedge-node):
  branching factor equal to the vertex's *excess degree*, which is
  Poisson(`c`)-distributed with mean `c` (size-biasing a Poisson(`c`) degree
  distribution reproduces Poisson(`c`), a special self-consistency property of
  the Poisson family used throughout local-weak-limit graph theory).

**Heuristic derivation.** Over one full "vertex-to-vertex" hop (two
factor-graph steps), the expected branching is `c·(k-1)`, so the Perron
eigenvalue of the two-step reduced operator is `c(k-1)`, and `ρ_B` (the
one-step, i.e. `√`-scale, Perron root) is
```
ρ_B ≈ √(c(k-1)).                                          (6.1)
```
This is exactly what `estimate_spectral_radius` measures as the untilted
noise floor, and the live run's `rho_B(0)` column (≈3.09–3.27 across the
`λ`-sweep, against the closed-form `√(c(k-1)) = √15 ≈ 3.873` at `c=5,k=3`)
is in the right ballpark though not tight at this finite, modest `n` — the
`hsbm_threshold_check.rs` example's own test only asserts agreement to within
35%, an explicit acknowledgment that this is a finite-`n` approximation to an
asymptotic statement.

To leading order in `λ`, planting affinity `λ` creates an **informative**
eigen-direction (aligned with community membership) whose two-step eigenvalue is
`λ·c(k-1)` (the planted bias directly multiplying the same branching number),
while the **uninformative** bulk remains at `c(k-1)`. A signal is
detectable above the noise bulk exactly when the informative eigenvalue's
magnitude exceeds what generic fluctuations of a bulk of that scale can produce
— informally, `(λ c(k-1))^2 > c(k-1)`, i.e.
```
λ_c = 1/√(c(k-1)).                                        (6.2)
```
At `k=2` (an ordinary graph, `q`-community SBM), this reduces to the classical
pairwise Kesten–Stigum threshold `λ_c = 1/√c` (Decelle–Krzakala–Moore–Zdeborová
2011), matching `hsbm_detectability.rs`'s own consistency check.

**Live-run interpretation.** With `c=5, k=3`: `λ_c = 1/√10 ≈ 0.3162`, matching
the printed value exactly. The run used `λ=0.6 > λ_c`; the detectability sweep's
`signal |Λ'(0)|` column does not monotonically increase with `λ` in this
particular single-seed, modest-`n` run (e.g. `0.6` shows a *smaller* signal than
`0.45` or `0.316` in the printed table) — reported honestly: this is exactly the
kind of finite-sample noise `hsbm_threshold_check.rs`'s own test anticipates by
only asserting the *qualitative* trend (deep-sub vs. deep-super, not a smooth
curve) and only over multiple hypergraph repetitions, which the ad hoc call in
`live_run.rs` did not average over as many reps as the dedicated example does.

---

## 7. Kesten–McKay law and the Alon–Boppana / Ramanujan bound

**Theorem 7a (Alon–Boppana, 1986; short proof due to Nilli, 1991).** For any
infinite sequence of `d`-regular graphs `G_1, G_2, …` with `|V(G_i)| → ∞`,
```
liminf_{i→∞} λ_2(G_i) ≥ 2√(d-1)
```
where `λ_2` is the second-largest eigenvalue of the adjacency matrix.

**Proof sketch (Nilli's test-function argument).** Fix `k`, and let `T_{d,k}` be
the depth-`k` `d`-regular tree rooted at `r`. One exhibits an explicit function
`f` on `T_{d,k}` (a specific combination of the tree's spherical/radial
functions, alternating sign by depth) such that the Rayleigh quotient
`⟨f, A f⟩ / ⟨f,f⟩` on the tree exceeds `2√(d-1)(1 - O(1/k))`, while `f`
is supported only within distance `k` of two disjoint points at graph distance
`> 2k` — such points exist in `G_i` once `|V(G_i)|` is large enough, since a
`d`-regular graph on `n` vertices has diameter `Ω(log_{d-1} n)`. Because `G_i`
looks exactly like `T_{d,k}` within radius `k` of any vertex (girth/tree-like
locally, for `n` large relative to `k`), transplanting `f` onto two
disjoint such balls in `G_i` and using it as a test vector orthogonal to the
all-ones (Perron) eigenvector gives, by the Courant–Fischer variational
characterization of `λ_2`, `λ_2(G_i) ≥ ⟨f,Af⟩/⟨f,f⟩ ≥ 2√(d-1)(1-O(1/k))`.
Letting `k→∞` slowly with `i` (any `k = o(log_{d-1} n_i)` suffices) proves the
liminf bound. (The precise numerical error term in the finite-`n`, finite-`k`
version is somewhat delicate to state exactly and is not reproduced here to
avoid misquoting Nilli's constants from memory; the asymptotic statement above
is the part actually needed and used.) ∎

This is exactly `continuum_limit::ramanujan_diagnostic`'s reference bound
`2√(d-1)`, and a graph achieving `λ_2 ≤ 2√(d-1)` (i.e. beating what generic
`d`-regular graphs must eventually satisfy, in the sense of matching the
*optimal* asymptotic rate) is called **Ramanujan**.

**Theorem 7b (Kesten 1959 / McKay 1981), statement only.** Let `G_n` be a
uniformly random `d`-regular graph on `n` vertices (configuration model,
conditioned simple). Then the empirical spectral distribution of `A(G_n)/1`
converges weakly, almost surely as `n→∞`, to the **Kesten–McKay density**
```
ρ_d(x) = { d√(4(d-1)-x²) / (2π(d²-x²))   if |x| ≤ 2√(d-1)
         { 0                              otherwise.
```

**Proof outline (moment method, not reproduced in full).** By the method of
moments, it suffices to show `E[Tr(A^k)]/n → μ_k`, the `k`-th moment of `ρ_d`,
for every fixed `k`, plus a concentration argument to upgrade convergence in
expectation to almost-sure convergence. `Tr(A^k)/n` counts (normalized) closed
walks of length `k` in `G_n`; because random `d`-regular graphs converge
locally (in the Benjamini–Schramm sense) to the infinite `d`-regular tree
`T_d` — short cycles occur with vanishing probability as `n→∞` for fixed `k` —
the expected count of closed walks of length `k` rooted at a uniformly random
vertex converges to the number of closed walks of length `k` on `T_d` rooted at
any fixed vertex, which is a classical combinatorial (generalized Catalan
number / generating-function) computation whose generating function is exactly
`ρ_d`'s moment generating function. This is the standard proof strategy (McKay
1981); the full combinatorial closed-walk-counting step on `T_d` is not
reproduced here. ∎

`continuum_limit.rs::kesten_mckay_density` implements the closed-form density
above directly, and `empirical_spectral_density` measures the LHS; the live run
of `spectral_dqg`'s own `main.rs` (not re-run in this session, but part of the
uploaded, previously-verified project) reports the RMS deviation shrinking with
`N`, which is the numerical instance of Theorem 7b's convergence statement.

**Friedman's theorem (2008), statement only, no proof reproduced.** A uniformly
random `d`-regular graph is, with probability `1-o(1)` as `n→∞`, "almost
Ramanujan": `λ_2 ≤ 2√(d-1) + ε` for any fixed `ε>0`. Friedman's proof (via a
delicate trace-method / representation-theoretic argument on `2k`-step closed
walks with `k` growing with `n`) is one of the longer proofs in spectral graph
theory and is well outside this document's scope; it is cited only because
`spectral_dqg::main.rs`'s own before/after demonstration (non-simple vs.
rejection-sampled simple regular graph generation) is an empirical illustration
of exactly this theorem's content, not a proof of it.

---

## 8. Correctness of the Krylov-subspace machinery

**Proposition 8a (Arnoldi factorization).** Given `A ∈ ℝ^{n×n}` and unit vector
`q_1`, the Arnoldi process (modified Gram–Schmidt applied to
`q_1, Aq_1, A^2q_1, …`) produces, after `k` steps (absent breakdown), an
orthonormal `Q_k = [q_1,…,q_k]` and upper Hessenberg `H_k ∈ ℝ^{k×k}` satisfying
```
A Q_k = Q_k H_k + h_{k+1,k} q_{k+1} e_k^T.
```

**Proof.** Immediate from the construction: at step `j`, `Aq_j` is orthogonalized
against `q_1,…,q_j` to produce `q_{j+1}` up to normalization, i.e.
`Aq_j = Σ_{i≤j} h_{ij} q_i + h_{j+1,j} q_{j+1}` by definition of the Gram–Schmidt
coefficients `h_{ij} = ⟨q_i, Aq_j⟩`. Stacking these `k` equations columnwise gives
exactly the stated matrix identity, with `H_k` upper Hessenberg because
`h_{ij}=0` for `i > j+1` (nothing above the subdiagonal is nonzero by
construction — each new vector is orthogonalized only against *previous* ones).
∎

**Proposition 8b (happy breakdown ⟹ exact eigenvalues).** If
`h_{k+1,k} = 0` at some step `k` (breakdown), every eigenvalue of `H_k` is an
exact eigenvalue of `A`.

**Proof.** Breakdown means `A Q_k = Q_k H_k` exactly (the residual term
vanishes), i.e. `span(Q_k)` is `A`-invariant. If `H_k y = θy`, then
`A(Q_k y) = Q_k H_k y = θ (Q_k y)`, and `Q_k y ≠ 0` since `Q_k` has orthonormal
(hence independent) columns and `y ≠ 0`. So `(θ, Q_k y)` is a genuine
eigenpair of `A`. ∎ This is exactly `krylov_ds`'s own documented breakdown
handling and is why the shift-invert unit test in `dqg_hsbm_bridge` can assert
exact (to numerical precision) agreement with dense `Schur`.

**Proposition 8c (shift-invert eigenvalue correspondence).** For `σ` not an
eigenvalue of `B`, `x` is an eigenvector of `(B-σI)^{-1}` with eigenvalue `ν`
if and only if `x` is an eigenvector of `B` with eigenvalue `μ = σ + 1/ν`.

**Proof.** `(B-σI)^{-1}x = νx ⟺ x = ν(B-σI)x ⟺ (1/ν)x = (B-σI)x ⟺ Bx = (σ+1/ν)x`,
using `ν≠0` (which holds since `(B-σI)^{-1}` is invertible, so has no zero
eigenvalue). ∎ This is exactly the transform `shift_invert.rs` applies to map
Arnoldi's Ritz values of `(B-σI)^{-1}` back to eigenvalues of `B`, and it is
also why the eigenvalue *nearest* `σ` in `B` becomes the *largest-magnitude*
(hence fastest-converging under plain Arnoldi) eigenvalue of `(B-σI)^{-1}` —
the entire reason shift-invert accelerates convergence to a targeted region of
the spectrum. The live run's agreement between shift-invert's recovered
eigenvalue and dense Schur's (`4.88e-14`) is a direct numerical confirmation of
this correspondence, not a separate fact requiring its own proof beyond the
one-line algebra above.

---

## 9. What is *not* proved here: the Ginibre-vs-Poisson bulk universality question

This section exists to be explicit about the boundary of proof. The live run's
`bulk_stats.rs` diagnostic measures two things about the non-outlier ("bulk")
eigenvalues of `B` on a sparse random hypergraph's factor graph:

1. **Radial (density) behavior** — is the bulk approximately uniform on a disk
   of radius `√ρ_B`? This is a genuine, partially-**proved** direction in the
   literature: Bordenave (2015, "A new proof of Friedman's second eigenvalue
   theorem and its extension to random lifts") and related work on the
   spectrum of non-backtracking operators of sparse random graphs establish
   circular-law-type statements for the *bulk density*, in an appropriate
   sparse/local limit. This document does **not** reproduce that proof — it is
   a substantial, specialized random-matrix-theory result — and the live run's
   `KS distance = 0.308` from the uniform-disk prediction, at `n_bulk ≈ 600`
   from a single seed, is far too small and noisy a sample to either confirm
   or contradict it. Stated as a numerical fact, not a disproof of the theorem.

2. **Local (spacing) statistics** — does the bulk exhibit level repulsion
   (associated with the Ginibre/complex-random-matrix universality class) or
   behave like an uncorrelated Poisson point process? **This is an open
   question in the literature for sparse non-backtracking operators** — the
   `chaos_test` module's own docstring cites this as the motivating question
   and references the complex spacing ratio statistic of Sá, Ribeiro & Prosen
   (2020) as one diagnostic tool, without claiming a resolution. No proof
   exists (to the knowledge encoded in either uploaded project or this
   document) settling this question for the hypergraph non-backtracking
   operator, and none is presented here. The live run's finding — the
   empirical spectrum showed *more*, not fewer, small spacings than a matched
   Poisson null at this sample size — is reported as a single noisy data
   point, explicitly **not** interpreted as evidence against level repulsion
   in the `n→∞` limit, since a sample of ~600 points from one hypergraph
   instance has no power to distinguish universality classes reliably.

**If a next step is wanted here**, the mathematically honest path is: (a) scale
`n` substantially (requires the sparse shift-invert extension flagged in
`shift_invert.rs`'s own module doc, since dense `Schur`/LU is the current
bottleneck), (b) average the spacing-ratio statistic over many independent
hypergraph instances at fixed `n`, and (c) compare against the two competing
theoretical predictions' *known* summary statistics (e.g. mean complex spacing
ratio `⟨|z|⟩ ≈ 0.74` for the Ginibre universality class, a literature constant
that would need to be cited from its source rather than asserted from memory
here) — none of which this document or the live run has done.

---

## Summary table: proof status of every claim the system makes

| # | Claim | Status |
|---|-------|--------|
| 1 | `Z_G(u)^{-1} = det(I-uB)` (Ihara) | **Proved** (§1, full proof) |
| 2 | Sylvester determinant identity | **Proved** (§2, full proof) |
| 3 | Bass determinant formula | **Proved** (§3, full proof from §2) |
| 4 | `Tr(B^k)=0` for `k` odd on bipartite graphs, `k=1,2` always | **Proved** (§4) |
| 5 | Factor-graph construction is lossless & Ihara/Bass apply to it | **Proved** (§5) |
| 5' | `chaos_test`'s `B_H` ≡ factor-graph `B` (some conjugation) | **Not checked** — not claimed |
| 6 | `λ_c = 1/√(c(k-1))` detectability threshold | **Heuristic** (§6), rigorous proof cited but not reproduced |
| 7a | Alon–Boppana bound `2√(d-1)` | **Proved** (§7a, proof sketch) |
| 7b | Kesten–McKay law | **Cited**, proof outlined not completed (§7b) |
| 7c | Friedman's theorem (near-Ramanujan random graphs) | **Cited only**, no proof |
| 8 | Arnoldi factorization, breakdown ⟹ exactness, shift-invert correspondence | **Proved** (§8, all three, full elementary proofs) |
| 9a | Bulk radial circular law for sparse non-backtracking spectra | **Cited** (Bordenave 2015), not reproduced, not verified at this sample size |
| 9b | Ginibre-vs-Poisson bulk spacing universality | **Open** — not proved by anyone, not resolved by this run |

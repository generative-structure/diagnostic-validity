# CANONICAL_DEFINITIONS.md

Single source of truth for every metric in the Synthetic Construction Engine
(Work Package 3). Every computation in `src/` implements the formulas below
exactly. Cross-reference: `theory_kernel_v02.tex` (definitions of mechanism,
projection, signature, baseline) and `signature_matrix_rationale.csv` (the
58 `experiment_needed=YES` cells this engine must resolve).

---

## 0. Conventions and engine-level decisions

These are decisions made where the WP3 prompt was underspecified. They are
flagged here so they can be reviewed and changed before execution.

- **Factorization backend.** `sympy` is not installed in this environment.
  Per the prompt ("`sympy.factorint` *or* a custom trial-division function")
  the engine uses a custom **smallest-prime-factor (SPF) sieve** plus a
  trial-division fallback for values above the sieve limit. The $p$-adic
  valuations $\nu_p(n)$ it returns are exact, so all formulas below are
  unchanged; only the algorithm differs.
- **Unit convention (Proposition 5, unitization is active).** All monetary
  generators operate in **integer cents**. `mag_range = (low, high)` bounds
  the *final recorded integer* (in cents), i.e. values lie in
  $[10^{\text{low}}, 10^{\text{high}}]$. This keeps `cents = value mod 100`
  meaningful and keeps magnitudes tractable for factorization. Changing the
  unit changes the arithmetic signature — by design.
- **Primary factorization projections cover all $n>1$.** $c_p$, L-profile and
  tail mass are computed over **every** $n>1$, including primes. A prime $q$
  contributes $L_1=1$ (all other $L_j=0$, tail $=0$) and $c_p=0$ for $p\ne q$,
  $c_q=1$. Excluding primes would condition on an arithmetic *outcome* of the
  mechanism (e.g. a divisibility constraint removes nearly all primes) and so
  would discard part of the signature. **Composite-only** variants (suffix
  `_composite`) are retained as a *sensitivity analysis* and never drive
  verdicts. `prime_fraction` (share of values that are prime) is reported per
  cell so a change in prime/composite composition is itself visible. All of
  this is applied identically to the mechanism and its magnitude-matched null.
- **Projection-column reduction.** The signature matrix has single columns
  (`DecimalResidue`, `cp_spectrum`, `L_profile`) that map to several engine
  metrics. The column verdict is taken from the **sub-metric with the largest
  $|z|$**, and `signature_matrix_resolved.csv` records which sub-metric was
  selected. (Mechanisms target specific primes/moduli, so max-$|z|$ is the
  natural reduction. Caveat: max over $k$ sub-metrics inflates $|z|$ under the
  null; the chosen sub-metric is reported so the multiple-comparison can be
  audited.)
- **Unresolvable cells.** A projection that requires a dimension the
  mechanism does not generate yields `NaN` and verdict `?` with the note
  *"not resolvable by engine (exogenous dimension)"*. Specifically:
  `TemporalConcentration` for any mechanism without timestamps (everything
  except `fiscal`), and `ThresholdDensity` for any mechanism without a defined
  threshold (everything except `threshold`). This honestly marks WP2 cells
  such as *Repeated exact × TemporalConcentration* and *Sum of … ×
  ThresholdDensity*, whose rationale was already "exogenous to the mechanism."

---

## 1. Projection metrics

All take a 1-D array of positive integers (`values`), filtered to `value > 1`.

### Leading-digit MAD
First significant digit $d\in\{1,\dots,9\}$. Mean absolute deviation of
observed proportions $\hat p_d$ from Benford:
$$\text{MAD} = \frac{1}{9}\sum_{d=1}^{9}\big|\hat p_d - p_d\big|,\qquad
p_d=\log_{10}\!\left(1+\tfrac{1}{d}\right).$$

### Terminal-digit chi-square
Terminal digit $t = \text{value} \bmod 10$, counts $O_t$, expected
$E = N/10$ (uniform):
$$\chi^2 = \sum_{t=0}^{9}\frac{(O_t-E)^2}{E}.$$

### Cents entropy
Cents $c = \text{value} \bmod 100$. Shannon entropy in **bits** of the
distribution over $c\in\{0,\dots,99\}$:
$$H = -\sum_{c} \hat p_c \log_2 \hat p_c, \qquad H_{\max}=\log_2 100\approx 6.644.$$

### Decimal-residue total variation
For modulus $m$ with observed residue frequencies $f_r$:
$$\text{TV}_m = \tfrac{1}{2}\sum_{r=0}^{m-1}\big|f_r-\tfrac1m\big|,
\qquad m\in\{5,10,25,100\}.$$
This projection tests focal decimal moduli $\{5,10,25,100\}$. Non-decimal
divisibility (e.g.\ mod 3, mod 7) is captured by the prime-identity spectrum
$c_p$, not by this projection. (Column name: `DecimalResidue`; sub-metric keys
remain `mod5_tv`, `mod10_tv`, `mod25_tv`, `mod100_tv`.)

### Vocabulary Gini
Gini coefficient of the **count distribution** of distinct values
(counts $x_1\le\dots\le x_K$):
$$G = \frac{2\sum_{i=1}^{K} i\,x_i}{K\sum_i x_i} - \frac{K+1}{K}.$$
$G=0$: all distinct values equally frequent. $G\to1$: one value dominates.

### Threshold density ratio
For threshold $T$ and bandwidth $\delta = b\cdot T$ (default $b=0.05$):
$$\text{ratio} = \frac{\#\{x\in[T-\delta,\,T)\}}{\#\{x\in[T,\,T+\delta]\}}.$$
(Only defined for mechanisms with a known $T$; see §0.)

### Temporal concentration index
For a calendar variable over $P=12$ periods, counts $O_t$, $E=N/P$:
$$\chi^2_{\text{temporal}} = \sum_{t=1}^{P}\frac{(O_t-E)^2}{E}.$$

### Prime-identity spectrum $c_p$
For integer $n>1$ and prime $p$, with $p$-adic valuation $\nu_p(n)$:
$$c_p(n) = \frac{\nu_p(n)\,\log p}{\log n}.$$
Population means $\bar c_p$ for $p\in\{2,3,5,7\}$. Plus two derived metrics:

**$c_{10}$ — base-10 / decimal-unit concentration** (use for round-number and
base-10 mechanisms; MSA convention $c_{25}=c_2+c_5$ — here named $c_{10}$ to
avoid collision with the $5^2$ metric below):
$$c_{10}(n) = c_2(n) + c_5(n).$$

**$c_{5^2}$ — pairs of factors of 5** (use for quarter-dollar / 25-cent focal
mechanisms; in code this is `c5sq`, and it must **not** be called $c_{25}$):
$$c_{5^2}(n) = \Big\lfloor \tfrac{\nu_5(n)}{2}\Big\rfloor\cdot\frac{\log 25}{\log n}
\quad(\log 25 = 2\log 5).$$

The `cp_spectrum` column reduces over $\{c_2,c_3,c_5,c_7,c_{10},c_{5^2}\}$.
Primary means are over all $n>1$ (a prime $q$ gives $c_q=1$, $c_p=0$ for
$p\ne q$); `_composite` variants restrict to composites (sensitivity only).

### L-profile
For $n=\prod_i p_i^{a_i}$, unsorted log-shares $\ell_i = a_i\log p_i/\log n$
(these sum to 1). Sort descending to get $L=(\ell_{(1)},\ell_{(2)},\dots)$.
Population means $\bar L_1,\bar L_2,\bar L_3,\bar L_4$ over **all** $n>1$ — a
prime contributes $L_1=1$ and $L_2=L_3=L_4=0$. `_composite` variants restrict
to composites (sensitivity only).

### Tail mass
$$\text{Tail} = \sum_{j\ge 3} L_j = 1 - L_1 - L_2 \quad(\text{per } n,\text{ then averaged, over all } n>1).$$

### CBAD parameters
Encode the **ordered** integers as strings joined by a single space; for each
prefix length $N$ compress with `zlib` level 9 and form the ratio
$$R(N) = \frac{\text{len}(\text{zlib9}(\text{prefix bytes}))}{\text{len}(\text{prefix bytes})}.$$
Use $\ge 10$ prefix lengths geometrically spaced from $N=100$ to $N_{\max}$
(default 12 points). Fit by nonlinear least squares (`scipy.optimize.curve_fit`,
bounds $a\in[0,1]$, $b\in[-2,2]$, $c\in[0,5]$):
$$R(N) = a + bN^{-c}.$$
Report $a$ (asymptotic compression ratio), $c$ (convergence exponent), and
$$R^2 = 1 - \frac{\sum (R_i-\hat R_i)^2}{\sum (R_i-\bar R)^2}.$$

---

## 2. Signature test

Per mechanism $k$ and metric $j$, with seed-level means $\bar\Phi$, seed-level
variances $s^2$, and seed counts $n$:
$$z_{j,k} = \frac{\bar\Phi_{j,k} - \bar\Phi_{j,\text{null}}}
{\sqrt{s_{j,k}^2/n_k + s_{j,\text{null}}^2/n_{\text{null}}}}.$$
The null is the **magnitude-matched** baseline $B_j(P_k)$ (§3), not a global null.

**Sign consistency.** With per-seed differences $\Delta_s=\Phi_{k,s}-\Phi_{\text{null},s}$
and overall direction $\sigma=\operatorname{sign}(\overline{\Delta})$:
$$\text{sign\_consistency} = \frac1{n}\sum_s \mathbf{1}[\operatorname{sign}(\Delta_s)=\sigma].$$

**Effect size.** Cohen's $d$ on seed-level values:
$$d = \frac{\bar\Phi_{j,k}-\bar\Phi_{j,\text{null}}}{\sqrt{(s_{j,k}^2+s_{j,\text{null}}^2)/2}}.$$

**Verdict rule.**
- $|z|>3$ **and** sign_consistency $>0.8$ → **`+`** (detected)
- $|z|<1.5$ across all magnitude strata → **`0`** (null confirmed)
- otherwise → **`?`** (uncertain; note recorded)

The main run assigns a provisional verdict from the pooled $z$; a `0` is
**confirmed** only if `sensitivity_magnitude.csv` (Step 5b) shows $|z|<1.5$ in
every digit stratum. Cells reduced from multiple sub-metrics use the
max-$|z|$ sub-metric (§0).

**Multiple-comparison audit.** Because max-$|z|$ over $k$ sub-metrics inflates
$|z|$ under the null, `signature_matrix_resolved.csv` records, per cell:
`selected_submetric`, `num_submetrics` ($k$), `raw_z` (the selected sub-metric's
$z$), `bonferroni_z_threshold` $=\Phi^{-1}\!\big(1-\alpha/(2k)\big)$ with
$\alpha=0.002$, and `passes_bonferroni`. For single-metric columns $k=1$ and
`passes_bonferroni` is simply $|z|>3$.

**Materiality.** `effect_size_d` is Cohen's $d$; `material` is $|d|>0.2$ (a soft
flag, not a gate). The detection `verdict` is unchanged (flat $|z|>3$). A separate
`verdict_material` reads `+` only when the detection rule fires **and** $|d|>0.2$
**and** `passes_bonferroni` is true. The last clause matters only for
max-$|z|$-reduced columns ($c_p$, modular residue, L-profile): without it a
selection-inflated null cell can read as a material positive. For single-metric
columns the Bonferroni threshold ($\approx 3.09$) is below the detection cutoff,
so the clause is automatically satisfied.

---

## 3. Magnitude-matched null baseline $B_j(P_k)$

1. Bin the mechanism values by digit length $d=\lfloor\log_{10}x\rfloor+1$ to get
   the target digit-length histogram.
2. Draw uniform integers over the mechanism's single range
   $[\min x,\ \max x]$ and accept them to fill each digit-length quota exactly
   (rejection matching). Sampling from one range — exactly as the generators do
   via `_uniform_int` — keeps numpy's range-dependent terminal-digit sampling
   artifact identical between mechanism and null, so the null control carries
   ~zero spurious signature. (Per-decade sampling sits systematically below
   single-range sampling on terminal-digit $\chi^2$ and produced a false positive
   on the null row.)
3. Compute all projections on this null. For temporal projections the null
   uses uniform months; threshold uses the mechanism's $T$.
4. Repeat per seed. RNG for the null uses `seed + 1_000_000` so it is
   independent of the mechanism stream.

---

## 4. Reproducibility

- All randomness via `numpy.random.Generator` (`np.random.default_rng(seed)`);
  every generator is deterministic given its seed.
- Defaults: `N = 100_000` observations, `SEEDS = range(20)`.
- `ProcessPoolExecutor` with 13 workers; the SPF sieve is built once and
  memory-mapped by workers (see `src/projections.py`).
- Output: CSV tables and PDF figures only.

# Gaussian-kernel topographic regularizers have no attractor at non-zero distance

**Date:** 2026-04-19
**Branch:** `autoresearch/trajectory-topography`
**Context:** Four Exp 2 pilots, three distinct pathologies, one unifying property.
**Status:** theoretical finding worth keeping regardless of whether Exp 2 continues.

## Claim

Let `K(d) = exp(−d² / 2σ²)` be a Gaussian kernel of bandwidth σ, applied to pairwise grid distances `d_ij = ‖pos_i − pos_j‖` between learned positions. Let `L = f(K(d_ij), target_ij)` be any topographic regularization loss that factors through `K(d_ij)` — i.e. any loss where the dependence on `d_ij` enters only via `K(d_ij)`.

Under gradient descent on `L` with respect to `pos_i`, the gradient magnitude vanishes in both the d→0 and d→∞ limits, for *every* pair (i, j), regardless of the choice of f or target matrix. Consequently: any non-zero equilibrium distance implied by the loss is a stationary point of the gradient field but is **not an attractor** — it has a finite basin of attraction surrounded on both sides by gradient-vanishing regions from which positions cannot be pulled back.

In practice, this means Gaussian-kernel-based topographic regularizers admit pathologies where pairs either *collapse to coincidence* (d → 0) or *escape to large separation* (d → ∞), and neither failure mode can be fixed by reshaping the target matrix.

## Derivation

For any loss `L = f(K(d_ij), T_ij)`, the chain rule gives

```
∂L/∂pos_i = Σ_j  [∂f/∂K]·[∂K/∂d]·[∂d/∂pos_i]
          = Σ_j  [∂f/∂K] · (−K(d_ij) / σ²) · (pos_i − pos_j)
```

where `∂K/∂d = −d · K(d) / σ²` and `∂d/∂pos_i = (pos_i − pos_j) / d`, so `(∂K/∂d)(∂d/∂pos_i) = −K(d)(pos_i − pos_j)/σ²`. The gradient magnitude for each term in the sum is

```
|∇_{pos_i} L|_term ∝ |∂f/∂K| · K(d_ij) · ‖pos_i − pos_j‖
```

regardless of f. Consider the two limits:

- **d → 0.** `K(d) → 1` (bounded) and `‖pos_i − pos_j‖ → 0`, so the gradient magnitude vanishes linearly with distance. **No force pushes coincident positions apart**, even if the loss at coincidence is non-zero.

- **d → ∞.** `K(d) → 0` exponentially, dominating the polynomial factor `‖pos_i − pos_j‖`. So `|∇ L|_term → 0` exponentially. **No force pulls widely-separated positions closer**.

The `∂f/∂K` factor can only reshape the gradient within the "active" distance range (roughly d ~ σ) where `K(d)` is non-trivial. It cannot extend forces into the d→0 or d→∞ regions where `K(d)` or `(pos_i − pos_j)` is already near-zero.

## Consequence: stationary points are not attractors

An equilibrium distance `d*` for pair (i,j) is defined as the d at which `∂f/∂K · K(d) = 0` — typically where `actual_sim(d*) = target_ij`. At `d*`, the gradient is zero by construction.

But gradient flow in the neighborhood of `d*` is defined by whether small perturbations from `d*` produce a restoring gradient. Under a Gaussian kernel, the gradient has a bell-shaped magnitude envelope that peaks near `d = σ` and decays in both directions. Any stationary point `d*` for which this envelope has already decayed significantly lacks a restoring gradient, even though the loss itself at `d*` might be locally concave.

Specifically: if `d* ≪ σ`, a perturbation to `d* − ε` lies in the `d → 0` region where the envelope is already small. No pullback. If `d* ≫ σ`, a perturbation to `d* + ε` lies in the `d → ∞` region. Same problem, other end.

## Empirical confirmation: three Exp 2 pilots, three pathologies, one property

| formulation | loss | observed pathology | pathology mechanism |
|---|---|---|---|
| Pure attractive Gaussian | `L = − Σ w_ij · K(d_ij)` | Collapse to coincidence, saturated at floor (pilots 1 / 2A / 2B all converge to spread 0.56 regardless of LR) | No repulsive term; d → 0 is the only attractor. Grid equilibrium spread of 0.56 is a frozen-mid-descent state, not a stable attractor |
| MSE-simple with target 0 for non-cooccur | `L = mean((K − T)²), T = cap·cooccur_norm` | Escape: positions expand to d where K≈0, gradient vanishes, system freezes at mean d ≈ 13 (pilot 2C-MSE) | d → ∞ stationary region reached by non-cooccur pairs; once there, gradient is zero in both directions |
| Equilibrium MSE with `T = max(er(σ), cap·cooccur_norm)` | Same loss, corrected target matrix | Bimodal failure: max-cooccur pairs collapse past d = 1.38 to d ≈ 0.2; non-cooccur pairs escape past d = 5.73 to d ≈ 16 (pilot 2D, mid-pilot) | Neither the near nor far equilibrium is an attractor |

All three failures are instances of the same mechanism. The targets differ, the pathologies differ in which direction they manifest, but the underlying property is the same: no basin of attraction for the intended equilibrium.

## What *does* avoid this

Losses parameterized on distance `d` directly, not on `K(d)`. For example:

```
L = mean((d_ij − target_d_ij)²)
∇_{pos_i} L ∝ (d_ij − target_d_ij) / d_ij · (pos_i − pos_j)
```

The gradient magnitude is proportional to the distance *error* `|d_ij − target_d_ij|`, which is unbounded above and only zero at the equilibrium itself. The direction is always toward the target distance. Globally well-behaved gradient dynamics; equilibria are true attractors, not stationary points.

The important distinction: losses on `K(d)` couple the "how far off are we from target" signal to the "how responsive is the kernel here" envelope. Losses on `d` decouple these.

## Why SOM-style topography doesn't have this problem

Self-organizing maps (Kohonen 1982) use a Gaussian kernel but are *not* gradient-based with respect to grid positions. The grid positions are fixed; the weights associated with each grid position learn. The Gaussian kernel in SOM controls which *weights* get updated based on proximity of their fixed grid location to the best-matching-unit's fixed grid location. The kernel does not determine a force on positions, so no gradient-vanishing issue arises.

Translating SOM-style dynamics into gradient-based learning of grid positions — which is what this project attempted — imports the Gaussian kernel without importing the non-gradient competitive-learning structure that keeps SOM well-behaved. The kernel ends up doing a different job than it was designed for, and the job it does (shaping gradient through `K(d)`) has the vanishing-gradient pathology.

## Implications for topographic regularization on transformer embeddings

1. **Don't use Gaussian-kernel-based topographic losses for gradient-based learning of grid positions.** The pathology is structural, not tunable.

2. **Distance-based losses (MSE on `d_ij`, rank-matching on distances, graph-Laplacian formulations) are the safer default.** They have globally convex gradient dynamics with respect to position error.

3. **Initialization-based topography is a path that avoids the regularization problem entirely.** Compute a grid layout offline (e.g., MDS on co-occurrence distances), use it to initialize token representations, let training proceed without any ongoing topographic loss. The research question shifts from "does topographic constraint affect learning" to "does topographic starting point affect learning," which is a cleaner question to test anyway.

4. **Published topographic-regularization approaches should be checked for this pathology.** If a reported method uses Gaussian-kernel attraction (or any kernel with the same limiting behavior), the method's stability may depend on training-horizon artifacts and the reported layouts may not be stable equilibria. The rate of collapse is a function of LR, not the loss's equilibrium structure.

## Relation to existing literature

The gradient-vanishing property of Gaussian kernels is well-known in neural-network optimization more broadly (e.g., in vanishing gradients through saturating nonlinearities, in kernel-induced ridge regression). Its application to topographic regularization doesn't appear to have been called out explicitly in the small-model representation-learning literature that motivated this project. Plausibly because most prior work on topographic regularization either (a) uses SOM-style non-gradient dynamics, (b) reports results at short training horizons where the formal equilibrium is less distinguishable from frozen-descent states, or (c) uses explicit repulsion terms that mask the underlying pathology.

Worth searching the mech-interp / superposition literature to see whether anyone has formulated the same observation. If not, this is a methodological contribution that belongs in an appendix of the eventual writeup.

## Bottom line

Three pilots over two days, each with its own apparent-pathology, were actually three manifestations of the same structural property: *Gaussian kernels induce vanishing gradients at both tails of the distance distribution, so any loss that factors through the kernel cannot produce stable equilibria at distances far from σ.* The formulations differed in which tail was reached; the property that made equilibria unreachable was shared.

Recording this here so the next time someone on this project considers a topographic regularizer, they start with a distance-based formulation and skip the kernel-based detour.

# QEC-QEM (`qec_qem`)

Quantum error correction (QEC) with quantum error mitigation (QEM) for small
surface codes, written in pure Python. It simulates a low-distance rotated
surface code with circuit-level noise, decodes it with a minimum-weight matching
decoder, and uses zero-noise extrapolation (ZNE) to estimate the logical result
at lower noise.

**It needs only `numpy` and `scipy`.** There is no Stim or PyMatching dependency
and nothing to compile, so it installs wherever those two install (including
Android Python apps such as Pydroid).

> **Status:** early release (0.2.x). The API may change.

## Contents
1. [The idea](#1-the-idea)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [How a run works](#4-how-a-run-works)
5. [The simulator](#5-the-simulator)
6. [Configuration reference](#6-configuration-reference)
7. [Results reference](#7-results-reference)
8. [The extrapolation (ZNE)](#8-the-extrapolation-zne)
9. [Recipes](#9-recipes)
10. [Reproducibility and speed](#10-reproducibility-and-speed)
11. [Validation](#11-validation)
12. [Troubleshooting](#12-troubleshooting)
13. [Limitations](#13-limitations)
14. [Development](#14-development)

---

## 1. The idea

A surface-code logical qubit is built from many physical qubits. In the rotated
layout one logical qubit needs **2d² − 1** physical qubits for code distance `d`:

| distance d | 3 | 5 | 7 | 9 |
|---|---|---|---|---|
| physical qubits per logical qubit | 17 | 49 | 97 | 161 |

A larger `d` suppresses logical errors better, but its qubit cost grows with d².
**QEC-QEM keeps `d` small to lower that resource cost.** A small code leaves
*residual* logical errors, and QEC-QEM handles them in software with ZNE:

1. Simulate the small code at several **amplified** noise levels. Errors are
   frequent there, so they are cheap to measure.
2. Decode each run and record the logical result.
3. Fit how the result changes with the noise scale, and **extrapolate back down**
   to lower noise: to your real operating noise, or toward zero noise for logical
   expectation values.

You get an estimate of what the logical qubit would deliver with the residual
noise removed, from a code that needs few physical qubits.

**Example (one run, `verify_low_distance_zne.py` defaults, p = 0.1%):** the raw
d=3 code has a residual logical error of 1.6e-3 (17 qubits). After zero-noise
extrapolation the estimate is within about 5e-4 of the ideal value. For
comparison, d=5 measured directly has 2.6e-4 (49 qubits). Your numbers will
differ with the noise level, shots and fit settings; the estimate carries a
statistical error and a small bias from the fit model.

## 2. Installation

```bash
pip install qec-qem          # the import name is qec_qem
```

Requires Python 3.9+, `numpy` and `scipy`.

## 3. Quick start

```python
from qec_qem import Config, CircuitArgs, SamplerArgs, ZNEArgs, run

p = 1e-3  # your real physical error rate (this is noise scale lambda = 1)

cfg = Config(
    circuit=CircuitArgs(
        distance=3,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    ),
    sampler=SamplerArgs(shots=500_000, seed=1),
    zne=ZNEArgs(target=1.0, models=("powerlaw", "polynomial")),
    lambdas=(2.0, 3.0, 4.0, 5.0),   # amplified noise levels to simulate
)

result = run(cfg)

for pt in result.points:
    print(pt.lam, pt.p_L, pt.stderr, pt.stats["num_qubits"])

z = result.zne
print("extrapolated p_L:", z.value, "CI:", z.ci_low, z.ci_high)
print("model spread:", z.model_spread)
print("warnings:", z.warnings)

result.to_json("result.json")   # full record of the run
```

This simulates noise rates 2p, 3p, 4p, 5p (500,000 shots each), fits the logical
error rate against the noise scale, and extrapolates to `lambda = 1`, i.e. the
rate `p` you care about.

## 4. How a run works

`run(cfg)` does the following:

| Step | What happens | Module |
|---|---|---|
| 1. Build circuit | Generate the noisy rotated surface-code memory circuit. | `circuit.py` |
| 2. Enumerate faults | Find, for every possible single fault, which detectors it flips and whether it flips the logical observable. Done once per code and cached; it does not depend on the noise rates. | `frame.py` |
| 3. Scale noise | For each noise scale `lambda`, multiply the four base rates by `lambda` (or use your `noise_scale`). | `config.py` |
| 4. Build graph | Turn the fault list and the scaled rates into a weighted matching graph. | `graph.py` |
| 5. Sample | Simulate many shots with random faults; record which detectors fire and whether the logical observable flipped. | `frame.py` |
| 6. Decode | Predict the observable flip from the detection events. | `decoder.py` |
| 7. Count | A shot **fails** if the prediction differs from the true flip. `p_L = failures / shots`. | `simulate.py` |
| 8. Extrapolate | Fit `p_L` (or the expectation) versus `lambda` and evaluate at `zne.target`. | `zne.py` |

`p_L` is the probability that the logical observable is decoded wrongly over the
whole experiment of `rounds` rounds of syndrome extraction. Total shots simulated
is `len(lambdas) × sampler.shots`.

## 5. The simulator

### Code and circuit
- **Code:** rotated surface code, odd distance `d >= 3`, one logical qubit,
  `d²` data qubits and `d² − 1` check qubits.
- **Tasks:** `rotated_memory_z` (the logical Z is preserved) and
  `rotated_memory_x`. The prefix `surface_code:` is accepted.
- **Each round:** data-qubit depolarizing noise; Hadamards on X-checks; four layers
  of CNOT gates; Hadamards again; measure and reset every check qubit.
- **Gate order:** X-checks and Z-checks use different four-step orders (a "Z" and
  an "N" shape) so that hook errors do not shorten the effective distance. The test
  suite checks that no qubit is used twice in a layer and that X and Z checks
  are ordered consistently.
- **Whole experiment:** reset, `rounds` rounds, then measure all data qubits.
  Detectors compare consecutive rounds; the observable is the logical operator
  read from the final data measurement.

### Noise model
Four independent channels, each with its own rate:

| Channel (config field) | Applied |
|---|---|
| `before_round_data_depolarization` | single-qubit depolarizing on data qubits at the start of every round |
| `after_clifford_depolarization` | single-qubit depolarizing after each H, two-qubit depolarizing after each CNOT |
| `before_measure_flip_probability` | basis-flip error right before each measurement |
| `after_reset_flip_probability` | basis-flip error right after each reset |

With rate `p`, depolarizing noise applies one of the 3 (or 15) non-identity Paulis,
each with probability `p/3` (or `p/15`). Idle qubits get no noise.

These are the same four channels that Stim's generated surface-code circuits use,
but this package is **not** Stim and its circuits are not guaranteed to match
Stim's gate for gate. See [Validation](#11-validation) for what was checked.

### Sampling
Errors in stabiliser circuits propagate deterministically, so each shot is
simulated by tracking only the Pauli error on every qubit (a "Pauli frame"),
vectorised over shots with numpy. Faults are drawn efficiently: the number of
faults is drawn from a binomial and their positions chosen without replacement.

### Decoder
A minimum-weight perfect matching decoder. Detection events are paired with each
other, or with the boundary, at minimum total weight (edge weight
`ln((1 − p)/p)`), and the logical flip is read from the chosen paths. Errors that
flip more than two detectors are split into graph-like pieces, and correlations
between pieces are ignored (as in standard matching decoders).

To stay fast, events are split into independent clusters (two events are only
paired if that is cheaper than sending both to the boundary). Each cluster is
solved **exactly** by dynamic programming. A cluster too large for the exact solver
(`max_exact_component` events or `max_states` DP states) falls back to a greedy
matching, which is counted in `decoder_info["approximate_clusters"]` and reported
with a warning. At `d = 3` and `d = 5` this did not occur in testing up to
`p = 1%`; it can occur for larger distances near threshold.

## 6. Configuration reference

Everything is set through `Config`.

### `Config`
| Field | Default | Description |
|---|---|---|
| `circuit` | `CircuitArgs()` | Circuit settings. |
| `sampler` | `SamplerArgs()` | Sampling settings. |
| `decoder` | `DecoderArgs()` | Decoder settings. |
| `zne` | `ZNEArgs()` | Extrapolation settings. |
| `lambdas` | `(2.0, 3.0, 4.0, 5.0)` | Noise scales to simulate. Positive and distinct. |
| `store_raw` | `False` | Keep raw detection events, observable flips and predictions in the result (large). |
| `noise_scale` | `None` | Your own scaling function `callable(CircuitArgs, lam) -> dict` of the four noise rates. |

### `CircuitArgs`
| Field | Default | Description |
|---|---|---|
| `code_task` | `"rotated_memory_z"` | Or `"rotated_memory_x"`. |
| `distance` | `3` | Odd integer, at least 3. |
| `rounds` | `None` | Syndrome rounds. `None` means the same as `distance`. |
| `after_clifford_depolarization` | `1e-3` | Noise after each gate. |
| `before_round_data_depolarization` | `1e-3` | Noise on data qubits each round. |
| `before_measure_flip_probability` | `1e-3` | Measurement flip probability. |
| `after_reset_flip_probability` | `1e-3` | Reset flip probability. |

These four rates are your **base rates** (`lambda = 1`). The simulated rate at noise
scale `lambda` is `lambda ×` each of them.

### `SamplerArgs`
| Field | Default | Description |
|---|---|---|
| `seed` | `None` | `None` picks a random seed. The seed used is always recorded. |
| `shots` | `100_000` | Shots **per noise scale**. |
| `batch_size` | `50_000` | Shots simulated at once; lower it to save memory. |

### `DecoderArgs`
| Field | Default | Description |
|---|---|---|
| `decoder` | `"matching"` | Built-in matching decoder. |
| `decoder_factory` | `None` | Your own decoder: `callable(graph)` returning an object with `decode_batch(det_events)`. |
| `max_exact_component` | `60` | Largest event cluster solved exactly. |
| `max_states` | `200_000` | Dynamic-programming state cap per cluster before the greedy fallback. |

### `ZNEArgs`
| Field | Default | Description |
|---|---|---|
| `target` | `1.0` | Noise scale to extrapolate to. `1.0` is your base rates. |
| `models` | `("powerlaw",)` | Fit models. The first is the primary result; extra models give a spread. |
| `poly_order` | `None` | Polynomial degree. `None` means `ceil(distance / 2)`. |
| `quantity` | `"logical_error_rate"` | Or `"expectation"` (the value `1 − 2·p_L`). |
| `n_bootstrap` | `500` | Bootstrap resamples for the confidence interval. |
| `confidence` | `0.95` | Confidence level. |
| `threshold_guard` | `0.007` | Warn if any simulated physical rate exceeds this. |
| `bootstrap_seed` | `None` | Seed for the bootstrap. |

### Validation of the config
`run()` raises `ValueError` before simulating if: `lambdas` is empty, has
non-positive or duplicate values; `distance` is not an odd integer ≥ 3; `rounds`
< 1; the code task is unsupported; a noise rate is outside [0, 1] (including after
scaling by lambda); `shots` or `batch_size` ≤ 0; or the decoder name is unknown.

## 7. Results reference

`run()` returns a `RunResult`. Call `result.to_dict()` or `result.to_json(path)`
to export everything.

### `RunResult`
| Field | Description |
|---|---|
| `config` | The full configuration used. |
| `versions` | Versions of `qec_qem`, NumPy, SciPy, Python. |
| `seed` | Base sampling seed. Noise scale number `i` uses `seed + i`. |
| `points` | List of `PointResult`, one per noise scale. |
| `zne` | The `ZNEResult`, or `None` if the fit was impossible (for example only one noise scale). |
| `zne_error` | Why `zne` is `None`. The simulated `points` are always kept. |

### `PointResult` (one noise scale)
| Field | Description |
|---|---|
| `lam` | Noise scale. |
| `noise` | The four scaled rates used. |
| `max_noise_rate` | Largest scaled rate. |
| `rounds` | Rounds used. |
| `seed` | Seed used for this point. |
| `shots`, `failures` | Shots run and logical failures. |
| `p_L`, `stderr` | Logical error rate and its binomial standard error. |
| `stats` | `num_qubits`, `num_detectors`, `num_observables`, `num_measurements`, `num_ticks`. |
| `circuit_text` | The circuit as readable text. |
| `dem_text` | The matching graph as text, one `error(p) D.. [L0]` line per edge. |
| `decoder` | `"matching"` or `"custom"`. |
| `decoder_info` | `effective_distance`, `n_graph_edges`, `unique_syndromes_decoded`, `approximate_clusters`. |
| `raw` | With `store_raw=True`: `det_events`, `obs_flips`, `predictions`. Otherwise `None`. |

### `ZNEResult`
| Field | Description |
|---|---|
| `primary_model` | Name of the first model that fitted. |
| `value`, `ci_low`, `ci_high` | Extrapolated value and interval of the primary model. The interval is `None` if too few bootstrap fits succeeded. |
| `fits` | List of `FitResult`, one per successful model. |
| `model_spread` | Max minus min of `value` across models. `None` with one model. |
| `failed_models` | Models that could not be fitted, with the reason. |
| `warnings` | Guard warnings (see section 8). |

### `FitResult` (one model)
| Field | Description |
|---|---|
| `model`, `quantity`, `target` | What was fitted and where it was evaluated. |
| `value`, `ci_low`, `ci_high`, `boot_std` | Extrapolated value, interval, bootstrap standard deviation. |
| `p_L_equivalent` | The value as a logical error rate. |
| `params`, `param_cov` | Fitted parameters and covariance matrix. |
| `chi2`, `dof`, `reduced_chi2` | Goodness of fit. |
| `n_points_used`, `n_boot_valid` | Points used; successful bootstrap fits. |
| `warnings` | Warnings specific to this fit. |

## 8. The extrapolation (ZNE)

### Models
| Model | Form | Use for | `params` keys |
|---|---|---|---|
| `powerlaw` | `y = A·λ^k` (weighted fit in log-log) | The logical error rate below threshold. **Default.** | `A`, `log_A`, `exponent` |
| `polynomial` | degree `poly_order` polynomial | **Logical expectation values** (use degree ⌈d/2⌉) and as a cross-check. | `c0`, `c1`, … |
| `exponential` | `y = A·exp(b·λ)` | Unprotected (unencoded) circuits. A poor fit for error-corrected logical results. | `A`, `b` |

**Which model for which quantity.** With QEC the logical error rate follows a
power law, roughly `p_L ∝ λ^⌈d/2⌉`, not an exponential. So for the expectation
`E = 1 − 2·p_L`, a polynomial of degree ⌈d/2⌉ matches the physics and an
exponential does not. On synthetic power-law data the exponential fit overshot
and fitted badly, while the polynomial recovered the true value.

`powerlaw` applies only to the logical error rate and needs `target > 0` (`p_L`
is trivially zero at zero noise). `polynomial` and `exponential` also allow
`target = 0`.

### Uncertainty
- Each point is weighted by its binomial standard error.
- The confidence interval comes from a bootstrap: failures at each noise scale
  are resampled from a binomial, the model is refit, and percentiles of the
  extrapolated values are taken.
- List several models in `zne.models`; `model_spread` shows how much the choice of
  model changes the answer.
- The interval covers statistical noise only. A wrong fit model adds a bias the
  interval does not include, so check `reduced_chi2` and the model spread.

### Warnings
| Warning | Meaning and what to do |
|---|---|
| dropped lambdas | A noise scale had zero (or all) failures, so it has no variance estimate and was excluded. Use more shots or a larger lambda. |
| threshold_guard | A simulated rate is above `threshold_guard`. The low-noise scaling law may not hold; use smaller lambdas. |
| long extrapolation | The target is more than twice the sampled lambda range below the lowest lambda. Add lower lambdas. |
| target above range | The target is above all sampled lambdas, so this is not extrapolation to lower noise. |
| poor fit | Reduced chi² above 3. Try another model or different lambdas. |
| bootstrap shortfall | Too few bootstrap fits succeeded, so no interval is reported. |
| model skipped | A model could not be fitted (reason given). |

### Choosing lambdas
- Keep every scaled rate `lambda × p` below roughly 0.5–1% (the circuit-level
  threshold region). The default `threshold_guard` is 0.007.
- Use at least one more usable point than the model has parameters.
- Choose lambdas and shots so that each point has enough failures (tens to
  hundreds) to give a usable error bar.
- A wide lambda range reduces statistical noise but increases model bias; a narrow
  range does the opposite. Try several ranges and compare.

## 9. Recipes

### Compare fit models
```python
cfg.zne.models = ("powerlaw", "polynomial")
result = run(cfg)
for f in result.zne.fits:
    print(f.model, f.value, f.ci_low, f.ci_high, f.reduced_chi2)
print(result.zne.model_spread)
```

### Extrapolate the logical expectation to zero noise
```python
cfg.lambdas = (1.0, 1.5, 2.0, 2.5, 3.0)
cfg.zne.quantity = "expectation"     # y = 1 - 2*p_L
cfg.zne.models = ("polynomial",)     # degree defaults to ceil(distance / 2)
cfg.zne.target = 0.0
```
For the memory circuits the ideal value is exactly 1, so this shows how closely
the extrapolation recovers it. `verify_low_distance_zne.py` does exactly this.

### Use a larger code
```python
cfg.circuit.distance = 5      # rounds defaults to the distance
```

### Save memory
```python
cfg.sampler.batch_size = 20_000
```

### Keep the raw data
```python
cfg.store_raw = True
result = run(cfg)
raw = result.points[0].raw     # det_events, obs_flips, predictions
```

### Scale only some noise channels
```python
from qec_qem import Config, default_scale

def measurement_only(args, lam):
    rates = default_scale(args, 1.0)               # base rates, unscaled
    rates["before_measure_flip_probability"] *= lam
    return rates

cfg = Config(noise_scale=measurement_only)
```

### Use your own decoder
`decoder_factory` receives the `DecodingGraph` and returns an object whose
`decode_batch(det_events)` takes a boolean array `(shots, n_detectors)` and returns
predicted observable flips with shape `(shots, 1)`.
```python
class MyDecoder:
    def __init__(self, graph): ...      # graph.edges: (u, v, p, obs); v == -1 is the boundary
    def decode_batch(self, det_events): ...

cfg.decoder.decoder_factory = lambda graph: MyDecoder(graph)
```

### Read a saved result
```python
import json
data = json.load(open("result.json"))
data["zne"]["value"], data["points"][0]["stats"]["num_qubits"]
```

## 10. Reproducibility and speed

- Set `sampler.seed` to fix the sampling. Noise scale number `i` uses `seed + i`,
  so scales get independent samples. Set `zne.bootstrap_seed` to fix the interval.
- The JSON export stores the config, circuit and graph text for every noise scale,
  counts, versions and all fit results.
- Speed depends on the machine. On a typical computer, `d = 3` runs about
  a million shots in a few seconds and `d = 5` about 100,000 shots per second or
  so; decoding time grows with the noise level and the distance. A phone will be
  slower. Lower `shots` for a quick first look.

## 11. Validation

The test suite (`tests/`) checks the simulator against known physics, because
there is no external reference inside the package:

- **Code structure:** qubit counts `2d² − 1`; equal numbers of X and Z checks;
  X and Z checks commute; no qubit conflicts in any gate layer; consistent X/Z
  ordering; a noiseless circuit produces no detection events.
- **Effective distance:** the matching graph's shortest logical path equals `d`
  for d = 3, 5, 7 (hook errors do not reduce the distance).
- **Error suppression:** the fitted exponent of `p_L` against `p` was 1.92 at
  d = 3 and 2.89 at d = 5 (expected about 2 and 3).
- **Threshold:** larger distance wins below threshold and loses above it. The
  d = 3 and d = 5 curves cross between 0.6% and 0.8%, consistent with the usual
  circuit-level surface-code threshold of roughly 0.6–1%.
- **Symmetry:** X-memory and Z-memory give the same logical error rate.
- **Sampler against the error model:** detector firing probabilities from sampling
  match those predicted from the enumerated faults.
- **Decoder:** the exact solver matches a brute-force minimum-weight matching on
  random syndromes.

These checks show the simulator behaves like a correct surface code. They do **not**
prove its logical error rates are identical to Stim's. If you need a cross-check,
compare a few `p_L` values against Stim and PyMatching on a machine that has them.

## 12. Troubleshooting
| Symptom | Likely cause and fix |
|---|---|
| `ValueError: ... distance ...` | Use an odd integer distance, 3 or more. |
| `ValueError: ... noise rate ... outside [0, 1]` | `lambda ×` a base rate exceeded 1. Use a smaller lambda or base rate. |
| `result.zne` is `None` | The extrapolation could not be fitted; see `result.zne_error`. Usually fewer usable points than model parameters, or zero failures. |
| `FitError: power law cannot be evaluated at target <= 0` | Use `target > 0`, or use the polynomial model with `quantity="expectation"`. |
| "dropped lambdas" warning | Zero failures at that noise scale. Increase `shots` or use larger lambdas. |
| No confidence interval | Too few bootstrap fits succeeded. Add points or shots. |
| Very large extrapolated error bars | Few failures or a long extrapolation. Add shots and lower lambdas. |
| Out of memory | Lower `sampler.batch_size` and leave `store_raw=False`. |
| Greedy-fallback warning | A large cluster of events was matched approximately. Lower the noise or raise `max_exact_component` / `max_states`. |
| Slow | Lower `shots`, use a smaller distance, or lower the noise scales. |

`FitError` can be imported with `from qec_qem.zne import FitError`.

## 13. Limitations
- Extrapolation gives a **model-based estimate**, not a guarantee. It is reliable
  only when all simulated noise levels are below threshold and the fit model
  matches the real scaling.
- It costs extra simulations at the amplified noise levels.
- Only rotated surface-code **memory** experiments (X or Z) with the four uniform
  noise channels above. No custom circuits, no idle noise, crosstalk, biased noise,
  logical gates, or routing between patches.
- Odd distance only.
- The decoder ignores correlations between the pieces of decomposed errors, as
  standard matching decoders do.
- Written in Python, so it is far slower than compiled simulators such as Stim.
- Surface-code qubit counts in section 1 are for one rotated patch.

## 14. Development
```
qec_qem/
  circuit.py    rotated surface-code memory circuit
  frame.py      Pauli-frame sampler and fault enumeration
  graph.py      matching graph and shortest paths
  decoder.py    minimum-weight matching decoder
  simulate.py   one noise scale: build, sample, decode, count
  config.py     configuration dataclasses
  results.py    result dataclasses and JSON export
  zne.py        fit models, bootstrap, warnings
  run.py        run(config)
tests/          test suite (source distribution only)
examples/       runnable example
```
Run the tests from the source distribution with `pip install -e .[test]` and `pytest`.
The extrapolation of a low-distance result is also checked end to end by the
separate script `verify_low_distance_zne.py`.

# Surface-code decoder comparison

A fresh Python research project for comparing Minimum-Weight Perfect Matching
(MWPM) with a random-start, probability-weighted Metropolis MCMC decoder on a
rotated surface code generated using Stim.

## Research question

"Can a Metropolis MCMC decoder, beginning from a random error configuration
consistent with the measured surface-code syndrome, find a low/minimum-weight
error configuration using physical error probabilities, and how does its
performance compare with MWPM?"

MWPM and MCMC are independent decoders. They receive the same measured syndrome.
MCMC will initialize from a random syndrome-compatible error configuration;
it will never initialize from MWPM or depend on an MWPM prediction.

## Planned architecture

```text
               same syndrome s
                 /        \
                /          \
             MWPM       random MCMC
              |              |
          prediction      random valid E
                             |
                         MCMC search
                             |
                           E_best
                             |
                         prediction
```

The eventual MCMC target uses physical edge probabilities:

```text
w_e = log((1 - p_e) / p_e)
W(E) = sum(w_e for e in E)
P(E) proportional to exp(-W(E))
delta_W = W(E_prime) - W(E)
P_accept = min(1, exp(-delta_W))
accept when log(U) < min(0, -delta_W), with U uniform on (0, 1)
```

Each proposal is a fresh uniform solution of the affine GF(2) syndrome constraint.
Random MCMC uses no cycle flips, `nx.cycle_basis`, or learned proposals.
A conditional binary model can be trained on independent-edge samples and used
by the separate `FlowProposalMCMCDecoder`. The original random MCMC and MWPM
baseline remain independent.

## Project layout

```text
surface_code_mcmc/
    README.md
    requirements.txt
    surface_code/
        __init__.py
        circuit.py          # Stim circuit construction
        decoding_graph.py   # Graph and physical edge metadata
        mwpm_decoder.py     # Independent MWPM baseline
        gf2.py              # Binary linear algebra utilities
        random_mcmc.py      # Random valid initialization and Metropolis search
        benchmark.py        # Compare decoders on the same syndromes
        plotting.py         # Visualize benchmark results
    tests/
        test_circuit.py
        test_graph.py
        test_gf2.py
        test_mcmc.py
    results/                # Future experiment outputs
```

## Setup

From this directory, using Python 3:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the independent decoder benchmark:

```bash
python -m surface_code.benchmark --stim-seed 42 --mcmc-seed 7 --output results/example
```

Defaults are distance 3, rounds 3, physical noise probability 0.005, 100 shots,
and 5000 MCMC iterations per shot. Override these with `--distance`, `--rounds`,
`--p`, `--shots`, and `--iterations`. For a quick check, use `--shots 2 --iterations 10`.

Each new output directory contains `shots.csv`, `summary.json`, `settings.json`,
and one MCMC trace CSV per shot under `traces/`. Existing output directories are
not overwritten. If seeds are omitted, generated seeds are saved in settings.
Both algorithms use the same fresh measured syndromes independently; MCMC uses
no MWPM results. Weights refer to the additive component-edge model, without
enforcing correlated DEM component groups. Decoder timings exclude Stim shot
generation and file output.

Enable the third, independent flow-assisted MCMC decoder with a matching
checkpoint:

```bash
python -m surface_code.benchmark \
    --flow-checkpoint flow_models/d3_r3_p0005/best.pt \
    --flow-probability 0.9 \
    --shots 3 --iterations 10 --stim-seed 42 --mcmc-seed 7 \
    --output results/three_method_example
```

All three decoders receive copies of the same measured syndrome. Both MCMC
methods start from independent uniform GF(2) configurations. Their RNG streams
are separate; `--flow-seed` optionally overrides a seed derived from the random
MCMC seed without consuming its stream. Neither MCMC receives matching output.

Three-method runs retain every legacy `mcmc_*` column, `traces/` file, and summary
key. They add the explicit `random_mcmc_*` and `flow_mcmc_*` shot metrics, both
flow-component acceptance rates, the three pairwise best-weight differences,
and per-shot traces under `flow_traces/`. Settings record the checkpoint path
and hash, mixture probability, and flow seed. Summaries include logical error
rates, weights, runtimes, and component acceptance statistics. Per-shot timings
include random initialization; checkpoint loading and reusable flow-decoder
construction are excluded. Without `--flow-checkpoint`, the original two-method
benchmark format remains available. Every run requires a new output directory.

Plot a completed run with an optional MWPM reference on the convergence plot:

```bash
python -m surface_code.plotting results/example --shot 0 --mwpm-reference
```

This saves `plots/convergence_shot_0.png` and `plots/weight_comparison.png` inside
the run directory. Use `--output` to choose a different plot directory. Omit
`--mwpm-reference` to show only current and best MCMC weights on the convergence
plot. Plotting reads saved results and does not invoke either decoder.

For a three-method run, the same plotting command automatically writes all five
comparisons:

```bash
python -m surface_code.plotting results/three_method_smoke_d3_r3_p0005 --shot 0
```

The default destination is `<run_dir>/plots/`:

- `convergence_comparison_shot_0.png`: both best-weight traces, including random
  initial states at iteration zero, plus the MWPM weight as a reference.
- `proposal_weight_histogram.png`: all uniform-baseline proposals versus only
  flow-labeled mixture proposals, across every shot. Accepted and rejected
  proposals are included; shared bins and density normalization account for
  different sample counts.
- `acceptance_rate_comparison.png`: mean per-shot acceptance rates for the
  uniform and mixture chains. The latter includes both proposal components.
- `per_shot_best_weight.png`: MWPM and both MCMC best weights on the same shots.
- `logical_error_rate_summary.png`: empirical mismatch rates with mismatch
  counts and the total number of shots.

`--shot` selects the convergence example; the other plots use the entire run.
The three-method convergence reference is always shown. Raw benchmark files are
read only, and the original two-method plotting behavior remains available.
New runs track the lowest-weight visited chain state (initial or accepted).
Historical traces may have included rejected proposals in their best field;
plotting reads those saved values unchanged. Best-weight curves do not by
themselves establish chain mixing or logical decoding improvement.

## Conditional proposal training

Train the conditional autoregressive Bernoulli model on a dataset generated by
`surface_code.flow_dataset`:

```bash
python -m surface_code.train_flow \
    --dataset flow_data/d3_r3_p0005 \
    --output flow_models/d3_r3_p0005 \
    --epochs 50 --batch-size 256 --learning-rate 1e-3 --seed 271828
```

Training minimizes `-mean(log q_theta(z | s))` using Adam and the dataset's
original train/validation arrays. It consumes only `(s, z)` pairs. Validation
does not contribute gradients. CPU initialization and mini-batch shuffling are
seeded; the default thread count is one (`--threads`).

Each epoch prints training and validation NLL in nats per complete coordinate
vector. Training NLL averages pre-update mini-batch losses; validation NLL
evaluates the end-of-epoch model. Early stopping defaults to five epochs without
improvement (`--patience 5`, `--min-delta 0`). The lowest validation NLL always
selects `best.pt`, independently of the early-stopping improvement threshold.

The new output directory contains `best.pt`, `last.pt`, `training_history.csv`,
and `metadata.json`. Metadata records dataset hashes, graph/basis fingerprints,
settings, versions, and the best checkpoint's validation mean log probability.
Both checkpoints retain architecture and graph/basis identity and load through
`ConditionalAutoregressiveBernoulli.load_checkpoint(..., expected_fingerprints=dataset_metadata)`.
They contain model parameters, not optimizer state for resuming training.
Existing output directories are refused; saved benchmark runs remain separate.

Compare raw proposal weights on saved validation syndromes:

```bash
python -m surface_code.validate_flow \
    --dataset flow_data/d3_r3_p0005 \
    --checkpoint flow_models/d3_r3_p0005/best.pt \
    --output flow_validation/d3_r3_p0005 \
    --num-syndromes 256 --proposals-per-syndrome 4 --seed 161803
```

This uses the existing uniform GF(2) sampler as the baseline, samples the learned
model independently, reconstructs both through affine coordinates, and checks
every syndrome. `validation.csv` records paired weights and binary syndrome/z
strings for reconstruction. `proposal_weights.png` compares distributions;
`summary.json` records mean, median, minimum, 5th/95th percentiles, fingerprints,
and replay settings. No decoder is invoked. The selected syndromes come from
model-selection validation data, so this is a proposal-weight comparison rather
than an independent test of decoding accuracy.

## Frozen flow-proposal MCMC

```python
from surface_code.flow_mcmc import FlowProposalMCMCDecoder

decoder = FlowProposalMCMCDecoder.from_checkpoint(
    graph, "flow_models/d3_r3_p0005/best.pt", flow_probability=0.9
)
result = decoder.decode(syndrome, iterations=100, seed=7)
error = result.best_configuration
```

The graph must match the checkpoint's edge-ordering and H/nullspace fingerprints.
Each chain starts from the existing uniform GF(2) sampler. At each subsequent
iteration, `flow_probability` (default 0.9) selects a proposal from a frozen
private copy of the model; otherwise the existing uniform sampler generates it.
Set this probability to 1 for pure flow or 0 for pure uniform proposals.
The independent Metropolis-Hastings log ratio is
`W_current - W_proposed + log_q_mix_current - log_q_mix_proposed`.
Both densities are the full mixture, regardless of which component generated
the proposal: `q_mix = alpha*q_theta + (1-alpha)*2**(-z_dim)`.
They are computed with stable log-sum-exp, without exponentiating tiny densities.
Every proposal is checked against its syndrome. No online training occurs.

The result includes initial/final/best configurations, L0 parity of the best
configuration, acceptance statistics, and a per-iteration trace. Trace
`proposal_source` records `flow` or `uniform`; `log_q_mix_current` is the
pre-decision value used in the ratio, and `current_weight` is the post-decision
weight. The older `log_q_current`/`log_q_proposed` attributes alias the mixture
densities. Results include `flow_proposals`, `uniform_proposals`,
`flow_accepted_moves`, `uniform_accepted_moves`, and per-component acceptance
rates (zero when that component has no proposals). The lowest-weight visited
state is retained, including initialization and accepted states only, exactly
as in the random baseline. Rejected proposals cannot update the best state.
New benchmark settings record this as
`flow_best_configuration_policy: initial_and_accepted_states`.

Run tests with `python -m pytest`.

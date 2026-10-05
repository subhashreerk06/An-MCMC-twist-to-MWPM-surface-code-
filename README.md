# MCMC-assisted surface-code decoding

This project explores whether Markov Chain Monte Carlo (MCMC) can add useful
information to Minimum-Weight Perfect Matching (MWPM) for surface-code decoding.

The surface code is generated with [Stim](https://github.com/quantumlib/Stim).
MWPM is used as the standard decoding baseline, while MCMC independently
explores error configurations that are compatible with the same measured
syndrome.

## Idea

For an error edge with probability \(p_e\), I assign the usual log-likelihood
weight

\[
w_e = \log\left(\frac{1-p_e}{p_e}\right).
\]

For an error configuration \(E\),

\[
W(E) = \sum_{e\in E} w_e,
\]

so that

\[
P(E) \propto e^{-W(E)}.
\]

The MCMC decoder starts from a random configuration satisfying the measured
syndrome and uses Metropolis sampling,

\[
P_{\mathrm{acc}}(E\rightarrow E')
=
\min(1,e^{-[W(E')-W(E)]}).
\]

Unlike MWPM, the MCMC decoder does not start from the minimum-weight matching.

## Why MCMC?

A syndrome can correspond to many different physical error configurations.
MWPM efficiently finds a minimum-weight explanation, but it does not explicitly
explore the full set of compatible configurations.

The aim of this project is to test whether sampling this space provides useful
information beyond the single MWPM solution.

A uniform random proposal was found to explore the space very inefficiently, so
the project also includes a learned conditional proposal model. The learned
model proposes more physically likely configurations while the
Metropolis-Hastings acceptance step keeps the correct target distribution.

## Project structure

- `surface_code/circuit.py` — Stim surface-code circuit
- `surface_code/decoding_graph.py` — decoding graph and edge probabilities
- `surface_code/mwpm_decoder.py` — MWPM baseline
- `surface_code/gf2.py` — GF(2) syndrome constraints
- `surface_code/random_mcmc.py` — random-start Metropolis MCMC
- `surface_code/flow_mcmc.py` — learned-proposal MCMC
- `surface_code/benchmark.py` — decoder comparison
- `surface_code/plotting.py` — result plots
- `tests/` — unit tests

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

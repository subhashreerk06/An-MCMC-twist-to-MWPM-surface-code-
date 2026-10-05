# MCMC-assisted surface-code decoding

This project explores whether Markov Chain Monte Carlo (MCMC) sampling can add useful information to Minimum-Weight Perfect Matching (MWPM) when decoding the surface code.

MWPM is already a very strong decoder. Given a measured syndrome, it efficiently finds a minimum-weight error configuration that could have produced it.

The question explored here is slightly different:

> A measured syndrome can be explained by many different physical error configurations. Can exploring this larger set of possibilities reveal useful information that is lost when we keep only one minimum-weight solution?

Surface-code circuits are generated using [Stim](https://github.com/quantumlib/Stim). MWPM is used as the standard baseline, while MCMC independently explores other error configurations compatible with the same syndrome.

---

## The decoding problem

A noisy surface-code experiment produces a detector syndrome $s$.

The syndrome tells us where detection events occurred, but it does **not** uniquely identify the physical error that caused them.

Many different error configurations can produce exactly the same syndrome.

Each possible error mechanism $e$ in the decoding graph is assigned an error probability $p_e$.

Its log-likelihood weight is

$$w_e=\ln\left(\frac{1-p_e}{p_e}\right)$$

so that more probable errors have smaller weights.

For a complete error configuration $E$, the total weight is

$$W(E)=\sum_{e\in E}w_e$$

Under the independent-edge noise model,

$$P(E)\propto e^{-W(E)}$$

so lower-weight configurations are more probable.

---

## The syndrome constraint

An error configuration can be represented by a binary vector

$$\mathbf{e}=(e_1,e_2,\ldots,e_M)$$

where each component is either 0 or 1.

Here,

- $e_i=1$ means error edge $i$ is present,
- $e_i=0$ means error edge $i$ is absent.

A valid error configuration must reproduce the measured syndrome.

This condition can be written over GF(2) as

$$H\mathbf{e}=\mathbf{s}\pmod 2$$

where $H$ is the detector-edge incidence matrix.

The set of all configurations compatible with the measured syndrome is therefore

$$\mathcal{E}(\mathbf{s})=\{\mathbf{e}:H\mathbf{e}=\mathbf{s}\pmod 2\}$$

The important point is that this set can contain many different physical error configurations.

---

# MWPM baseline

Minimum-Weight Perfect Matching searches for the lowest-weight configuration compatible with the measured syndrome.

Conceptually,

$$E_{\mathrm{MWPM}}=\mathrm{arg\,min}_{E:\,HE=s}W(E)$$

The logical prediction is then determined from the logical parity of this configuration.

MWPM is extremely efficient because the decoding problem can be transformed into a matching problem and solved directly.

However, MWPM primarily answers the question:

> Which individual syndrome-compatible error configuration has the lowest weight?

There is another question that can also matter:

> Which logical class contains the greatest total probability when all compatible configurations are considered?

That distinction motivates the MCMC part of this project.

---

# Degeneracy and logical classes

Different physical error configurations can produce the same syndrome.

They can also differ by their logical effect.

For the single logical observable considered here,

$$L(E)\in\{0,1\}$$

All syndrome-compatible configurations can therefore be separated into two logical classes.

The total posterior weight of logical class $L=0$ is

$$Z_0(s)=\sum_{\substack{E:\,HE=s\\L(E)=0}}e^{-W(E)}$$

and for logical class $L=1$,

$$Z_1(s)=\sum_{\substack{E:\,HE=s\\L(E)=1}}e^{-W(E)}$$

The corresponding logical probabilities are

$$P(L=0\mid s)=\frac{Z_0(s)}{Z_0(s)+Z_1(s)}$$

and

$$P(L=1\mid s)=\frac{Z_1(s)}{Z_0(s)+Z_1(s)}$$

This means that the class containing the single lowest-weight configuration is not necessarily guaranteed to contain the greatest total probability.

For example, MWPM may find one particularly low-weight configuration in logical class 0, while logical class 1 may contain many slightly heavier configurations whose probabilities add up to a larger total.

This is the degeneracy information that the MCMC method is designed to investigate.

---

# Random-start MCMC

The MCMC decoder is deliberately independent of MWPM.

It does **not** start from the MWPM correction.

Instead, it begins from a random configuration satisfying

$$H\mathbf{e}=\mathbf{s}\pmod 2$$

The initial configuration is generated directly from the GF(2) syndrome constraint.

Starting from a current configuration $E$, another syndrome-compatible configuration $E'$ is proposed.

The change in weight is

$$\Delta W=W(E')-W(E)$$

The Metropolis acceptance probability is

$$P_{\mathrm{acc}}(E\rightarrow E')=\min\left(1,e^{-\Delta W}\right)$$

If the proposed configuration has lower weight,

$$\Delta W\leq0$$

it is always accepted.

If the proposal has higher weight,

$$\Delta W>0$$

it can still be accepted with probability

$$P_{\mathrm{acc}}=e^{-\Delta W}$$

The Markov chain therefore samples from a distribution proportional to

$$\pi(E\mid s)\propto e^{-W(E)}$$

while remaining inside the space of configurations compatible with the measured syndrome.

---

# Why uniform random proposals struggled

The first MCMC implementation generated completely random configurations satisfying the syndrome constraint.

These configurations were mathematically valid, but most of them had extremely large total weights.

As a result:

- most proposals were physically very unlikely,
- very few moves were accepted,
- the chain explored the useful low-weight region very inefficiently.

This led to an important observation:

> Generating a syndrome-compatible configuration is easy. Generating a useful syndrome-compatible configuration is much harder.

This motivated the introduction of a learned proposal model.

---

# Representing all valid configurations

The solution space of the syndrome equation can be written as

$$\mathbf{e}=\mathbf{e}_p(\mathbf{s})+N\mathbf{z}\pmod 2$$

where:

- $\mathbf{e}_p(\mathbf{s})$ is one particular solution of the syndrome equation,
- $N$ spans the null space of $H$,
- $\mathbf{z}$ contains the free binary variables.

Because

$$HN=0$$

changing $\mathbf{z}$ does not change the syndrome.

Therefore,

$$H\mathbf{e}=\mathbf{s}$$

for every allowed value of $\mathbf{z}$.

This gives a convenient way to represent the full space of syndrome-compatible error configurations.

---

# Learned proposals

A conditional binary generative model is trained to learn a proposal distribution

$$q_\theta(\mathbf{z}\mid\mathbf{s})$$

The model learns which regions of the syndrome-compatible configuration space are more likely to contain physically relevant errors.

Importantly, the learned model does **not** replace the physical probability distribution.

The physical target is still determined by

$$W(E)=\sum_{e\in E}w_e$$

and

$$\pi(E\mid s)\propto e^{-W(E)}$$

The learned model only changes how candidate configurations are proposed.

Its purpose is to help MCMC reach the low-weight region much more efficiently than uniform random proposals.

---

# Metropolis-Hastings with learned proposals

Once proposals are no longer uniform, their proposal probabilities must also be included in the acceptance rule.

The Metropolis-Hastings ratio is

$$R=\frac{\pi(E'\mid s)\,q_\theta(E\mid s)}{\pi(E\mid s)\,q_\theta(E'\mid s)}$$

Since

$$\pi(E\mid s)\propto e^{-W(E)}$$

the logarithmic ratio becomes

$$\ln R=W(E)-W(E')+\ln q_\theta(E\mid s)-\ln q_\theta(E'\mid s)$$

A proposal is accepted when

$$\ln U<\min(0,\ln R)$$

where $U$ is drawn uniformly between 0 and 1.

The target distribution remains

$$\pi(E\mid s)\propto e^{-W(E)}$$

so the learned model improves the search without changing the physical distribution that MCMC is supposed to sample.

---

# From minimum-weight search to posterior decoding

Initially, MCMC was used mainly as a search method.

The lowest-weight configuration encountered during the chain was recorded as

$$E_{\mathrm{best}}=\mathrm{arg\,min}_{E\in\mathrm{visited}}W(E)$$

This was useful for checking whether MCMC could reach the same low-weight region as MWPM.

However, MWPM is already designed specifically to solve a minimum-weight problem.

The more interesting use of MCMC is therefore to keep many sampled configurations and ask how the posterior probability is distributed between logical classes.

---

# Posterior MCMC decoder

After an initial burn-in period, the remaining MCMC states are retained.

Suppose there are $N$ retained samples.

Let

$$N_1=\text{number of retained samples with }L(E)=1$$

and

$$N_0=\text{number of retained samples with }L(E)=0$$

with

$$N=N_0+N_1$$

The logical posterior probabilities are estimated by

$$\hat P(L=1\mid s)=\frac{N_1}{N}$$

and

$$\hat P(L=0\mid s)=\frac{N_0}{N}$$

The posterior decoder then chooses the logical class with the greater sampled probability.

If

$$\hat P(L=1\mid s)>0.5$$

the posterior prediction is logical class 1.

If

$$\hat P(L=1\mid s)<0.5$$

the posterior prediction is logical class 0.

This changes the question from

> Which individual error configuration has minimum weight?

to

> Which logical class carries the greatest total posterior probability?

---

# MWPM + posterior correction

The final experiment compares the ordinary MWPM prediction with the posterior MCMC prediction.

For every measured syndrome, both methods are evaluated independently.

There are four possible outcomes:

| MWPM | Posterior MCMC | Interpretation |
|---|---|---|
| Correct | Correct | Both decoders succeed |
| Wrong | Correct | MCMC fixes an MWPM failure |
| Correct | Wrong | MCMC changes a correct MWPM result into a failure |
| Wrong | Wrong | Both decoders fail |

The main performance measure is the logical error rate,

$$\mathrm{LER}=\frac{N_{\mathrm{logical\ failures}}}{N_{\mathrm{shots}}}$$

A genuine decoding improvement would require

$$\mathrm{LER}_{\mathrm{posterior}}<\mathrm{LER}_{\mathrm{MWPM}}$$

Another useful measure is

$$N_{\mathrm{net}}=N_{\mathrm{fixed}}-N_{\mathrm{broken}}$$

where:

- $N_{\mathrm{fixed}}$ is the number of MWPM failures corrected by posterior MCMC,
- $N_{\mathrm{broken}}$ is the number of correct MWPM predictions changed into failures.

If $N_{\mathrm{net}}>0$, the posterior layer has produced a net improvement on the tested sample.

---

# Current result

The current experiment uses

$$d=3,\qquad r=3,\qquad p=0.005$$

with

$$N_{\mathrm{shots}}=100$$

independent surface-code shots.

MWPM produced one logical failure:

$$\mathrm{LER}_{\mathrm{MWPM}}=\frac{1}{100}=0.01$$

The posterior MCMC decoder also produced one logical failure:

$$\mathrm{LER}_{\mathrm{posterior}}=0.01$$

The paired comparison gave:

- 99 shots where both methods were correct,
- 0 MWPM failures corrected by MCMC,
- 0 correct MWPM predictions broken by MCMC,
- 1 shot where both methods failed.

Therefore,

$$\Delta\mathrm{LER}=\mathrm{LER}_{\mathrm{posterior}}-\mathrm{LER}_{\mathrm{MWPM}}=0$$

for this sample.

The posterior decoder did not change the logical prediction produced by MWPM on any of the 100 tested shots.

---

# Interpretation

The result does **not** mean that error degeneracy does not exist.

The MCMC posterior explicitly includes information from many syndrome-compatible configurations that ordinary MWPM does not explicitly sum over.

However, for the syndromes observed in this low-noise experiment, the additional degeneracy information did not change which logical class was preferred.

In other words, for these shots,

$$\mathrm{arg\,min}_{E}W(E)$$

and

$$\mathrm{arg\,max}_{L}P(L\mid s)$$

led to the same final logical prediction.

So the additional posterior information was successfully included, but it did not alter the decoding decision.

For this tested regime, MWPM was already sufficient.

---

# Computational cost

There is also a large difference in computational cost.

For the 100-shot posterior benchmark, MWPM required approximately

$$1.9\times10^{-3}\ \mathrm{s/shot}$$

while the current posterior MCMC implementation required approximately

$$3.7\times10^{2}\ \mathrm{s/shot}$$

The posterior method therefore required far more computation while producing the same logical predictions in this experiment.

For the tested regime, the additional sampling cost was not justified by an improvement in decoding accuracy.

---

# What the project shows so far

## 1. Uniform random MCMC is inefficient

Random syndrome-compatible configurations usually lie in very high-weight regions of the solution space.

This produces low acceptance rates and inefficient exploration.

## 2. Learned proposals greatly improve MCMC exploration

The learned proposal model guides the chain toward physically relevant, low-weight configurations.

In small tests, learned-proposal MCMC reached weights close to those found by MWPM after only a small number of iterations, while uniform random MCMC remained at much larger weights.

## 3. Degeneracy-aware posterior sampling did not improve MWPM in the tested regime

Once posterior sampling became practical, the full logical-class probability was compared with the ordinary MWPM prediction.

For

$$d=3,\qquad p=0.005$$

MWPM and posterior MCMC produced the same logical prediction on all 100 tested shots.

So, in this regime, explicitly accounting for degeneracy did not provide a measurable improvement over MWPM.

---

# What this does and does not imply

The current result suggests that the minimum-weight approximation used by MWPM is already very effective for this particular low-noise surface-code regime.

It does **not** show that degeneracy is always irrelevant.

More difficult regimes may behave differently, including:

- higher physical error probabilities,
- larger code distances,
- correlated noise,
- biased noise,
- leakage or crosstalk,
- syndromes close to logical ambiguity.

The next scientific question is therefore:

> Under what physical noise conditions does the probability of an entire logical class differ enough from the minimum-weight prediction for degeneracy-aware decoding to matter?

---

# Project structure

```text
surface_code/
    circuit.py
    decoding_graph.py
    mwpm_decoder.py
    gf2.py
    random_mcmc.py
    flow_dataset.py
    discrete_flow.py
    train_flow.py
    validate_flow.py
    flow_mcmc.py
    posterior_mcmc.py
    mwpm_posterior_corrector.py
    benchmark.py
    posterior_benchmark.py
    plotting.py

tests/
```

---

# Setup

Create a Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the tests with:

```bash
python -m pytest
```

---

# Baseline benchmark

Run the MWPM versus random-start MCMC comparison with:

```bash
python -m surface_code.benchmark \
  --distance 3 \
  --rounds 3 \
  --p 0.005 \
  --shots 100 \
  --iterations 5000 \
  --stim-seed 2026092701 \
  --mcmc-seed 2026092702 \
  --output results/baseline_example
```

---

# Flow-assisted MCMC

After training a compatible proposal model, run:

```bash
python -m surface_code.benchmark \
  --distance 3 \
  --rounds 3 \
  --p 0.005 \
  --shots 100 \
  --iterations 100 \
  --stim-seed 2026092701 \
  --mcmc-seed 2026092702 \
  --flow-checkpoint flow_models/d3_r3_p0005/best.pt \
  --flow-probability 0.9 \
  --output results/flow_comparison
```

---

# Posterior benchmark

Run the degeneracy-aware comparison with:

```bash
python -m surface_code.posterior_benchmark \
  --distance 3 \
  --rounds 3 \
  --p 0.005 \
  --shots 100 \
  --iterations 1000 \
  --burn-in 200 \
  --flow-checkpoint flow_models/d3_r3_p0005/best.pt \
  --flow-probability 0.9 \
  --stim-seed 2026100201 \
  --mcmc-seed 2026100202 \
  --correction-margin 0.0 \
  --min-logical-transitions 0 \
  --min-logical-ess 0 \
  --output results/posterior_d3_r3_p0005_100shots
```

---

# Main idea

The project is based on the distinction

**MWPM:** find the most likely individual error configuration.

**MCMC posterior:** estimate which logical class carries the greatest total probability.

For the low-noise regime tested so far, both approaches led to the same final logical predictions.

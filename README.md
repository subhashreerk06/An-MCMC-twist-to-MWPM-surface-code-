# MCMC-assisted surface-code decoding

This project explores whether Markov Chain Monte Carlo (MCMC) sampling can add useful information to Minimum-Weight Perfect Matching (MWPM) for decoding the surface code.

MWPM is already a very strong decoder. It efficiently finds a minimum-weight error configuration compatible with a measured syndrome.

The motivation here is slightly different:

> A syndrome can be explained by many different physical error configurations. Can exploring this larger set of configurations reveal information that is missed when we keep only the single minimum-weight solution?

The surface-code circuits are generated using [Stim](https://github.com/quantumlib/Stim). MWPM is used as the decoding baseline, while MCMC independently explores error configurations compatible with the same measured syndrome.

---

## The decoding problem

A noisy surface-code experiment produces a detector syndrome $s$.

The syndrome tells us which detectors changed, but it does **not** uniquely determine the physical error that caused them.

Many different error configurations can produce exactly the same syndrome.

Each possible error mechanism $e$ in the decoding graph has an associated error probability $p_e$.

The corresponding log-likelihood weight is

$$w_e=\ln\left(\frac{1-p_e}{p_e}\right)$$

so that more probable errors receive smaller weights.

For a complete error configuration $E$, the total weight is

$$W(E)=\sum_{e\in E}w_e$$

Under the independent-edge noise model,

$$P(E)\propto e^{-W(E)}$$

so configurations with smaller total weight are more probable.

---

## Syndrome constraint

An error configuration can be represented by a binary vector

$$\mathbf{e}=(e_1,e_2,\ldots,e_M)$$

where $e_i=1$ means that error edge $i$ is present and $e_i=0$ means that it is absent.

A valid error configuration must reproduce the measured detector syndrome.

This condition can be written over GF(2) as

$$H\mathbf{e}=\mathbf{s}\pmod 2$$

where $H$ is the detector-edge incidence matrix.

The full set of configurations compatible with a syndrome is therefore

$$\mathcal{E}(\mathbf{s})=\left\{\mathbf{e}\;|\;H\mathbf{e}=\mathbf{s}\pmod 2\right\}$$

The main difficulty is that this set can contain many different physical error configurations.

---

# MWPM baseline

Minimum-Weight Perfect Matching searches for a minimum-weight error configuration compatible with the measured syndrome.

Conceptually,

$$E_{\mathrm{MWPM}}=\underset{E:\,HE=s}{\operatorname{argmin}}\,W(E)$$

The logical prediction is then determined from the logical parity of this configuration.

MWPM is extremely efficient because the decoding problem can be transformed into a matching problem and solved directly.

However, MWPM primarily answers the question:

> Which **individual** syndrome-compatible error configuration has the smallest weight?

There is another closely related question:

> Which **logical class as a whole** contains the greatest total probability?

That distinction motivates the MCMC part of this project.

---

# Degeneracy and logical classes

Different physical error configurations can produce the same syndrome and can also belong to the same logical class.

For the single logical observable considered here,

$$L(E)\in\{0,1\}$$

The total probability associated with logical class $L=0$ is proportional to

$$Z_0(s)=\sum_{\substack{E:\,HE=s\\L(E)=0}}e^{-W(E)}$$

while the corresponding quantity for logical class $L=1$ is

$$Z_1(s)=\sum_{\substack{E:\,HE=s\\L(E)=1}}e^{-W(E)}$$

The logical posterior probabilities are therefore

$$P(L=0\mid s)=\frac{Z_0(s)}{Z_0(s)+Z_1(s)}$$

and

$$P(L=1\mid s)=\frac{Z_1(s)}{Z_0(s)+Z_1(s)}$$

This means that the logical class containing the single lowest-weight configuration is not necessarily guaranteed to have the largest total probability.

For example, MWPM may find one particularly good configuration in class $L=0$, while there may be many slightly heavier configurations in class $L=1$ whose probabilities add up to a larger value.

This is the degeneracy information that the MCMC approach is designed to explore.

---

# Random-start MCMC

The MCMC decoder is deliberately independent of MWPM.

It does **not** start from the MWPM solution.

Instead, it begins from a randomly generated configuration satisfying

$$H\mathbf{e}=\mathbf{s}\pmod 2$$

The random valid configuration is obtained using GF(2) linear algebra.

Starting from a current error configuration $E$, another syndrome-compatible configuration $E'$ is proposed.

The change in weight is

$$\Delta W=W(E')-W(E)$$

The Metropolis acceptance probability is

$$P_{\mathrm{acc}}(E\rightarrow E')=\min\left(1,e^{-\Delta W}\right)$$

If the proposed configuration has lower weight,

$$\Delta W\leq0$$

and the proposal is always accepted.

If the proposed configuration has higher weight,

$$\Delta W>0$$

then it is accepted with probability

$$P_{\mathrm{acc}}=e^{-\Delta W}$$

The resulting Markov chain targets

$$\pi(E\mid s)\propto e^{-W(E)}$$

This allows MCMC to explore many different physical configurations compatible with the same measured syndrome.

---

# Why uniform random proposals struggled

The first MCMC implementation proposed completely random configurations satisfying the syndrome constraint.

These configurations were mathematically valid, but most of them had extremely large weights.

This caused two problems:

- most proposed configurations were physically very unlikely,
- the Metropolis acceptance rate became extremely small.

The chain therefore struggled to reach the low-weight region of configuration space.

This led to an important observation:

> Generating a configuration with the correct syndrome is easy. Generating a **useful** configuration with the correct syndrome is much harder.

This motivated the introduction of a learned proposal distribution.

---

# Free-variable representation

The solution space of the syndrome equation can be written as

$$\mathbf{e}=\mathbf{e}_p(\mathbf{s})+N\mathbf{z}\pmod 2$$

where

- $\mathbf{e}_p(\mathbf{s})$ is one particular solution of the syndrome equation,
- $N$ spans the null space of $H$,
- $\mathbf{z}$ contains the free binary degrees of freedom.

Because

$$HN=0$$

every value of $\mathbf{z}$ automatically produces another error configuration with the same syndrome:

$$H\mathbf{e}=H\mathbf{e}_p(\mathbf{s})=\mathbf{s}$$

This provides a convenient unconstrained binary coordinate system for generating syndrome-compatible error configurations.

---

# Learned proposals

A conditional binary generative model is trained to learn a proposal distribution

$$q_\theta(\mathbf{z}\mid\mathbf{s})$$

The purpose of the learned model is **not** to replace the physical noise model.

The physical target distribution is still determined by the edge probabilities and the weight

$$W(E)=\sum_{e\in E}w_e$$

The learned model only tries to propose configurations in regions that are more physically relevant than a completely uniform random draw.

This makes MCMC exploration much more efficient.

---

# Metropolis-Hastings with learned proposals

Once proposals are no longer uniform, the ordinary Metropolis acceptance rule must be corrected for the proposal probability.

The Metropolis-Hastings ratio is

$$R=\frac{\pi(E'\mid s)\,q_\theta(E\mid s)}{\pi(E\mid s)\,q_\theta(E'\mid s)}$$

Using

$$\pi(E\mid s)\propto e^{-W(E)}$$

the logarithmic acceptance ratio becomes

$$\ln R=W(E)-W(E')+\ln q_\theta(E\mid s)-\ln q_\theta(E'\mid s)$$

The proposal is accepted when

$$\ln U<\min(0,\ln R)$$

with

$$U\sim\mathrm{Uniform}(0,1)$$

The target distribution therefore remains unchanged:

$$\pi(E\mid s)\propto e^{-W(E)}$$

The learned model only changes **how candidate configurations are proposed**.

---

# From optimization to posterior decoding

Initially, MCMC was used as a search algorithm.

The lowest-weight state encountered during the chain was stored as

$$E_{\mathrm{best}}=\underset{E\in\text{visited states}}{\operatorname{argmin}}\,W(E)$$

This was useful for checking whether MCMC could reach the same low-weight region as MWPM.

However, MWPM is already specifically designed to solve a minimum-weight problem.

The more interesting use of MCMC is therefore not to compete with MWPM at finding

$$\min_E W(E)$$

but to sample many syndrome-compatible configurations and estimate how the total probability is distributed between logical classes.

---

# Posterior MCMC decoder

After a burn-in period, the Markov-chain states are retained.

Suppose the retained configurations are

$$E_1,E_2,\ldots,E_N$$

Each configuration has a logical class

$$L(E_i)\in\{0,1\}$$

The posterior probability of logical class $1$ is estimated as

$$\hat P(L=1\mid s)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\left[L(E_i)=1\right]$$

and therefore

$$\hat P(L=0\mid s)=1-\hat P(L=1\mid s)$$

The posterior decoder chooses the class with the larger estimated probability.

For example,

$$\hat P(L=1\mid s)>0.5\quad\Rightarrow\quad L_{\mathrm{post}}=1$$

while

$$\hat P(L=1\mid s)<0.5\quad\Rightarrow\quad L_{\mathrm{post}}=0$$

This changes the decoding question from

> Which individual error configuration has the smallest weight?

to

> Which logical class contains the greatest total posterior probability?

---

# MWPM + posterior correction

The final experiment compares the ordinary MWPM prediction with the posterior MCMC prediction.

For each syndrome, both methods are evaluated independently.

There are four possible outcomes:

| MWPM | Posterior MCMC | Interpretation |
|---|---|---|
| Correct | Correct | Both decoders succeed |
| Wrong | Correct | MCMC fixes an MWPM failure |
| Correct | Wrong | MCMC damages a correct MWPM prediction |
| Wrong | Wrong | Both decoders fail |

The main performance quantity is the logical error rate,

$$\mathrm{LER}=\frac{N_{\mathrm{logical\ failures}}}{N_{\mathrm{shots}}}$$

A genuine improvement would require

$$\mathrm{LER}_{\mathrm{posterior}}<\mathrm{LER}_{\mathrm{MWPM}}$$

Another useful quantity is the net number of MWPM failures corrected,

$$N_{\mathrm{net}}=N_{\mathrm{fixed}}-N_{\mathrm{broken}}$$

where $N_{\mathrm{fixed}}$ counts MWPM failures corrected by posterior MCMC and $N_{\mathrm{broken}}$ counts correct MWPM predictions changed into failures.

A positive $N_{\mathrm{net}}$ would indicate that the MCMC posterior contains useful information beyond the original MWPM prediction.

---

# Current result

The current test uses

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
- 0 correct MWPM predictions damaged by MCMC,
- 1 shot where both methods were wrong.

Therefore,

$$\boxed{\Delta\mathrm{LER}=\mathrm{LER}_{\mathrm{posterior}}-\mathrm{LER}_{\mathrm{MWPM}}=0}$$

for this sample.

The posterior decoder did not change the logical prediction produced by MWPM on any of the 100 tested shots.

---

# Interpretation

The result does **not** mean that the degeneracy problem does not exist.

The posterior MCMC explicitly includes information from many syndrome-compatible configurations that ordinary MWPM does not explicitly sum over.

However, for the syndromes sampled in this low-noise regime, the additional degeneracy information did not change the dominant logical class.

In other words, for these shots,

$$\underset{E}{\operatorname{argmin}}\,W(E)$$

and

$$\underset{L}{\operatorname{argmax}}\,P(L\mid s)$$

led to the same final logical prediction.

So although MCMC incorporated more information about the space of compatible errors, that additional information did not improve the logical decision.

For this tested regime, MWPM was already sufficient.

---

# Computational cost

There is also a large computational difference.

For the 100-shot posterior benchmark, MWPM required approximately

$$1.9\times10^{-3}\ \mathrm{s/shot}$$

while the posterior MCMC implementation required roughly

$$3.7\times10^{2}\ \mathrm{s/shot}$$

The posterior calculation therefore required much more computation while producing the same final logical predictions in this experiment.

For this regime, the additional sampling cost was not justified by an improvement in decoding accuracy.

---

# What the project shows so far

## 1. Uniform random MCMC is inefficient

Random syndrome-compatible configurations usually lie in extremely high-weight regions of configuration space.

This produces very low acceptance rates and inefficient exploration.

## 2. Learned proposals dramatically improve MCMC exploration

The conditional proposal model guides the chain toward physically relevant low-weight configurations.

In small tests, learned-proposal MCMC reached weights close to MWPM after only a small number of iterations, while uniform random MCMC remained at very large weights.

## 3. Degeneracy-aware posterior sampling did not improve MWPM in the tested regime

Once posterior exploration became practical, the logical-class probability was compared directly with the ordinary MWPM prediction.

For

$$d=3,\qquad p=0.005$$

the posterior decoder and MWPM produced the same logical prediction on all 100 tested shots.

So, in this regime, explicitly accounting for degeneracy did not provide a measurable decoding improvement.

---

# What this does and does not imply

The current result suggests that the minimum-weight approximation used by MWPM is already very effective for the tested low-noise surface-code regime.

It does **not** show that degeneracy is always irrelevant.

More difficult regimes may behave differently, including:

- higher physical error probabilities,
- larger code distances,
- correlated errors,
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

A basic MWPM versus random-start MCMC comparison can be run with:

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

After training a compatible proposal model, the learned-proposal decoder can be included using:

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

The degeneracy-aware posterior comparison can be run with:

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

The project can be summarized by the distinction

$$\boxed{\text{MWPM finds the best individual error configuration}}$$

while

$$\boxed{\text{MCMC estimates which logical class carries the greatest total probability}}$$

The current result is that, for the tested low-noise regime, these two approaches led to the same final logical predictions.

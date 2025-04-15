# Chapter 2: Neuroscience Foundations for AI

## 2.0 Why This Chapter?

Before we leap into convolutional networks, policy gradients, or billion-parameter transformers, we need mental "ground truth" about the biological machine that inspired all of them. By the end of this chapter you should be able to:

- Sketch a real neuron and label the parts that correspond (roughly!) to weights, bias, and activation in an artificial unit.
- Explain why Hebb's 1949 slogan "cells that fire together wire together" foreshadows the weight-update rule inside a modern optimiser.
- Map the cortex–hippocampus–basal-ganglia trio onto supervised, episodic-memory, and reinforcement-learning motifs.
- Run—and modify—a minimal leaky-integrate-and-fire simulator with a local Hebbian learning rule.

Keep two caveats in mind throughout:

- The metaphors run in both directions but never line up perfectly; equivalence is neither the goal nor the reality.
- Biological plausibility is a spectrum, not a binary. Sometimes we pursue it to understand brains, sometimes to inspire better silicon algorithms, and sometimes we happily ignore it for the sake of performance.

## 2.1 Neuron Anatomy & Electrophysiology

At the basic level, brains and computers both traffic in "bits" of information, but their substrates look very different.

### Structure cheat-sheet
- **Dendrites**: tree-like input fibres. Thousands of synapses live on tiny protrusions called spines.
- **Soma** (cell body): sums incoming currents; site of most protein synthesis.
- **Axon hillock**: narrow neck where action potentials are born if the membrane hits threshold.
- **Axon**: long cable (up to a metre in the human motor neuron) that propagates spikes.
- **Synaptic terminal**: packs vesicles filled with neurotransmitter; releases them into the synaptic cleft.

### Electrical basics

At rest a neuron sits ≈ –70 mV inside relative to outside. Ion channels act as nano-switches for Na⁺, K⁺, Ca²⁺, Cl⁻. When excitatory post-synaptic potentials (EPSPs) depolarise the membrane to roughly –55 mV, voltage-gated Na⁺ channels open, causing a 1 ms "action potential" hump. After the refractory period the game can restart, making a spike train whose frequency encodes information.

### Chemical messenger story

An arriving spike triggers Ca²⁺ influx, vesicles fuse, and transmitter molecules (e.g. glutamate, GABA, dopamine) drift across a 20 nm synaptic cleft. The postsynaptic membrane hosts ligand-gated channels that either depolarise (excitatory) or hyperpolarise (inhibitory). Note that the sign of the synapse is fixed by receptor type; a glutamatergic synapse never "turns negative."

### Analogy to an artificial perceptron

- Dendritic spine → input line xᵢ
- Synaptic strength (number & efficacy of receptors) → weight wᵢ
- Somatic integration → weighted sum Σ wᵢxᵢ
- Firing threshold → activation function f(⋅)
- All-or-nothing spike → binary output y

```{figure} ../figures/neuron_vs_perceptron.svg
---
height: 300px
name: neuron-vs-perceptron
---
Side-by-side cartoon: biological neuron (labelled) vs perceptron diagram.
```

But note three mismatches:

1. **Time**: ANNs usually compute in static passes; real neurons integrate continuously.
2. **Signal space**: Perceptron inputs are dimensionless numbers; dendrites receive precise spike timings and neuro-chemical signatures.
3. **Activation**: An action potential is always the same height; in ANNs the output can vary (sigmoid, ReLU, etc.). Information in brains is carried more by timing and rate than amplitude.

## 2.2 Neural Circuits & Layers

The human cerebral cortex folds a 2-mm-thick sheet into six histological layers. A canonical microcircuit (largely conserved across vision, audition, touch) routes signals like a clever factory conveyor belt.

### Layer guide (sensory cortex example)

- **L4**: main input station from thalamus; packed with spiny stellate cells.
- **L2/3**: intra-cortical relay; outputs to neighbouring columns.
- **L5**: thick pyramidal neurons project to sub-cortical targets and back to thalamus.
- **L6**: feedback to thalamus; modulates gain.

### Feed-forward vs recurrent

Early textbooks sold the cortex as a stack of feed-forward filters. We now know that ~80% of synapses are horizontal or feedback. This recurrence enables context, expectation, and predictive coding—features still being imported into AI via ResNets, U-Nets, and transformers.

### Convergence & divergence

A macaque V1 neuron might receive 6,000 synapses and send its axon to tens of thousands. Fan-in/out mirrors the fully-connected layers in a multilayer perceptron (MLP). However, cortical networks balance excitation with ~20% inhibitory interneurons that sculpt activity and prevent runaway excitation—a biological answer to the exploding-gradient problem.

```{figure} ../figures/cortical_column_vs_ann.svg
---
height: 350px
name: cortical-column-vs-ann
---
Schematic of a cortical column with arrows marking feed-forward, lateral, and feedback paths; alongside a 3-layer ANN highlighting similar motifs.
```

## 2.3 Learning & Plasticity

Donald Hebb (1949) proposed: "When an axon of cell A is near enough to excite cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells." Translation: correlated activity strengthens the synapse.

### 3 flavours you'll encounter:

1. **Classical Hebbian**: Δwᵢⱼ = η·xᵢ·yⱼ (pre × post).
2. **Spike-Timing-Dependent Plasticity (STDP)**: weight change depends on Δt = t_post – t_pre. Δw ∝ exp(–|Δt|/τ) with positive sign if pre leads post, negative otherwise.
3. **BCM Theory** (Bienenstock, Cooper, Munro): introduces a sliding threshold ϑ_M; synapses potentiate above it and depress below, producing competition and homeostasis.

### Long-term potentiation & depression

In hippocampal slices, high-frequency bursts (≈ 100 Hz) can double EPSP size for hours—a cellular memory trace called LTP. Low-frequency trains (≈ 1 Hz) cause LTD. Molecular mediators include NMDA-receptor-gated Ca²⁺ entry and cascades that add or remove AMPA receptors.

### Gradient descent vs. plasticity

- **Locality**: Hebbian rules need only xᵢ and yⱼ; back-prop needs global error δ.
- **Signs**: LTP/LTD gate only the learning rate, not the instantaneous error direction.
- **Timing**: STDP encodes millisecond precision; SGD deals in batch steps.

Research crossroads: can we design learning algorithms that marry back-prop efficiency with Hebbian locality? Ideas include feedback alignment, target propagation, and predictive coding networks.

## 2.4 Brain Organisation Cheat-sheet

- **Cortex**: hierarchical feature extractor & planner. ANN analogy = deep feed-forward & recurrent stacks.
- **Hippocampus**: indexes episodes via place cells and pattern separation. Analogy = content-addressable memory / auto-encoder bottleneck.
- **Basal ganglia**: dopamine-gated winner-take-all loop that learns action values. Analogy = actor-critic reinforcement learner.
- **Cerebellum**: supervised error-corrector for timing & coordination; millions of tiny granule cells converge on Purkinje output. Analogy = massively parallel shallow learners (think ensemble boosting).
- **Thalamus**: smart relay & attention gate, not just a dumb switchboard. Analogy = transformer attention head that routes information based on context.

```{figure} ../figures/brain_regions_table.svg
---
height: 300px
name: brain-regions-table
---
Table with five rows (region, main job, canonical cell type, ML metaphor).
```

```{tip}
Evolution re-uses building blocks. The same Hebbian synapse supports vision in occipital cortex and monetary reward learning in orbitofrontal cortex—proof that algorithmic principles scale.
```

## 2.5 Code Lab – Spikes & Hebbian Learning

Fire up Python 3.10+, NumPy, and Matplotlib. The following single cell is intentionally explicit rather than hyper-optimised so readers can tweak each line.

```python
# Leaky-Integrate-and-Fire neuron + Hebbian 2-neuron network
# Author: R. Young & ChatGPT, 2025
import numpy as np
import matplotlib.pyplot as plt

# ---------- Part A: single LIF neuron ----------
T = 300      # ms total simulation time
dt = 0.1     # ms time step
steps = int(T/dt)

tau_m = 20.0       # membrane time constant (ms)
V_rest = -70.0     # mV
V_thresh = -54.0   # spike threshold (mV)
V_reset = -80.0    # reset potential (mV)
R_m = 10.0         # MΩ
I_ext = 1.8        # nA external current

V = np.zeros(steps)
V[0] = V_rest
spikes = []

for t in range(1, steps):
    dV = (-(V[t-1]-V_rest) + R_m*I_ext) / tau_m
    V[t] = V[t-1] + dt*dV
    if V[t] >= V_thresh:
        V[t-1] = 20.0     # for pretty spike in plot
        V[t] = V_reset
        spikes.append(t*dt)

# ---------- Part B: 2-neuron Hebbian network ----------
eta = 0.01                # learning rate
w = np.random.randn(2,2)*0.1   # weight matrix
pre_spike_prob = 0.05     # probability pre neuron fires each ms

w_history = []

for epoch in range(200):        # training epochs
    pre = (np.random.rand(2) < pre_spike_prob).astype(float)
    post_input = pre @ w        # simple linear sum
    post = (post_input > 0.3).astype(float)   # threshold non-linearity
    
    # Hebbian update: Δw = η * pre^T * post
    dw = eta * np.outer(pre, post)
    w += dw
    w_history.append(w.copy())

w_history = np.array(w_history)

# ---------- Visualisation ----------
fig, ax = plt.subplots(1,2, figsize=(11,4))

# Membrane potential trace
ax[0].plot(np.arange(steps)*dt, V, lw=1.2)
ax[0].set(title="Leaky-Integrate-and-Fire Trace",
          xlabel="Time (ms)", ylabel="Membrane potential (mV)")
for s in spikes:
    ax[0].axvline(s, color='r', alpha=0.3)

# Weight matrix evolution heat-map
im = ax[1].imshow(w_history.transpose(1,2,0).reshape(4,-1),
                  aspect='auto', cmap='magma', origin='lower')
ax[1].set(title="Hebbian Weight Growth",
          xlabel="Training step", ylabel="Weight index")
fig.colorbar(im, ax=ax[1], shrink=0.8, label='w value')
plt.tight_layout()
```

### Try-this-now exercises

1. Increase τₘ to 40 ms; observe slower decay & higher firing rate.
2. Make one synapse inhibitory in Part B by initialising it negative; trace how competition evolves.
3. Replace the binary post-neuron with a noisy sigmoid: σ(β · input) and watch weight saturation.

```{note}
Complete code for this lab is available in the accompanying `ch02_demo.py` file.
```

## 2.6 Key Take-aways

- Biological neurons fire at kilohertz at best, yet 86 billion of them work in parallel while consuming only 20 W—roughly a dim light bulb.
- Plasticity is local and continuous. Back-prop is global and episodic. Bridging the gap is an active frontier (see "feedback alignment").
- Studying the real brain supplies "design priors" that have already birthed convolution, recurrence, attention, and spiking ASICs. Ignoring biology wholesale risks reinventing wheels—or worse, building brittle systems that fail outside curated datasets.

## 2.7 Further Reading & Media

### Books
- Bear, Connors & Paradiso – *Neuroscience: Exploring the Brain*, 4th ed., Ch 2-3, 23-27.
- Dayan & Abbott – *Theoretical Neuroscience*, Ch 8 (spike-based plasticity).

### Articles
- Song, Miller & Abbott (2000) – "Competitive Hebbian Learning Through STDP", *Nat. Neuro*.
- Lillicrap et al. (2020) – "Backpropagation and the Brain", *Nat. Rev. Neurosci*.

### Videos / MOOCs
- Crash Course Neuroscience Ep 1-3 (Intro & Neuron Anatomy).
- 3Blue1Brown – "What is a Neural Network?" for the ANN perspective.
- Neuromatch Academy (YouTube) – Model neurons & STDP day.

### Software / Demos
- Brian2 simulator (Python) – friendly spiking-network sandbox.
- NEST / NEURON – large-scale brain simulators if you crave HPC.
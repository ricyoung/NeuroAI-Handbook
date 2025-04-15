# Chapter 5: Default-Mode vs Executive Control Networks

## 5.0 Chapter Goals
- Understand large-scale brain networks and their functional significance
- Contrast default-mode and executive control networks
- Connect network dynamics to cognitive functions
- Explore computational models of brain network interactions

## 5.1 Network Neuroscience Fundamentals
- From local circuits to distributed networks
- Methods: fMRI, EEG, MEG, graph theory
- Functional vs structural connectivity
- Network metrics and their interpretation

## 5.2 The Default Mode Network (DMN)
- Discovery and components
- Self-referential processing
- Mind-wandering and creativity
- Relation to memory and imagination

## 5.3 Executive Control Networks
- Frontoparietal control network
- Salience network
- Cognitive control functions
- Working memory and attention

## 5.4 Network Interactions
- Anti-correlation and competitive dynamics
- Task-positive vs task-negative networks
- Dynamic network reconfiguration
- Balance and dysregulation in disorders

## 5.5 Computational Approaches
- Graph theoretical models
- Dynamic causal modeling
- Neural mass models
- Links to reservoir computing

## 5.6 Code Lab
- Network analysis of brain connectivity data
- Simulating coupled oscillator networks
- Visualizing network dynamics

## 5.7 Take-aways
- Brain function emerges from network interactions
- Balance between networks supports adaptive cognition
- Network properties constrain and enable computation

## 5.8 Further Reading & Media
- Raichle (2015) - "The Brain's Default Mode Network"
- Cole et al. (2014) - "Intrinsic and Task-Evoked Network Architectures of the Human Brain"
- Bassett & Sporns (2017) - "Network neuroscience"

---

## In-Depth Content

### The Default Mode Network (DMN)

The Default Mode Network (DMN) is a set of brain regions (medial prefrontal cortex, posterior cingulate, precuneus, angular gyrus, etc.) that consistently show high activity during rest and self-referential thought. The DMN is most active when we recall memories, imagine the future, or reflect on ourselves – essentially, when we are not focused on an external task. It is often called the brain's "task-negative" or introspective network.

Key point: The DMN activates in the resting state and is involved in self-referential and introspective thought. It deactivates during external goal-driven tasks, suggesting a toggle between internal and external modes of cognition.

### Executive Control Networks

In contrast, when the brain engages in a demanding task, other networks kick in – e.g., the Fronto-Parietal Network (FPN) or dorsal attention network – which handle attention, problem-solving, and working memory. These can be seen as the brain's "task-positive" networks, directing focus to external goals.

### Dynamic Interaction

The DMN and executive networks are often anti-correlated (when one is active, the other quiets down). This balance might be analogous to how an AI agent could have a mode for planning (internally simulate scenarios) versus a mode for direct action/perception. We consider whether today's AI models have an equivalent of a default mode – for instance, a language model generating text without a prompt could be likened to "daydreaming," or generative models creating data akin to imagination.

### Transcranial Direct Current Stimulation (tDCS)

Transcranial Direct Current Stimulation (tDCS) is a form of non-invasive brain stimulation that uses a weak electrical current applied to the scalp to change the likelihood of neuronal firing and modulate brain activity. Stimulating regions of the DMN or executive network can alter cognitive performance (some studies attempt to enhance learning or creativity by targeting these networks). This raises an interesting parallel: in AI, we sometimes "fine-tune" or adjust parts of a network externally – could ideas from tDCS inform non-gradient interventions in AI models?

### Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import signal
import pandas as pd
import seaborn as sns

# Simulate a simple brain network model with DMN and Executive Network dynamics
def simulate_brain_networks(duration=600, sampling_rate=10, noise_level=0.1, task_timing=None):
    """
    Simulate time series data from DMN and Executive Network with anti-correlation
    
    Parameters:
    duration: simulation length in seconds
    sampling_rate: Hz
    noise_level: amount of random fluctuation
    task_timing: list of tuples (start_time, end_time) for task periods
    
    Returns:
    time: time points
    dmn_activity: simulated DMN time series
    exec_activity: simulated executive network time series
    """
    # Create time axis
    n_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, n_samples)
    
    # Base oscillations (slow intrinsic fluctuations ~0.1 Hz)
    freq_dmn = 0.05  # DMN fluctuation frequency
    freq_exec = 0.03  # Exec network fluctuation frequency
    
    # Generate intrinsic oscillations
    dmn_activity = 0.5 * np.sin(2 * np.pi * freq_dmn * time)
    exec_activity = 0.3 * np.sin(2 * np.pi * freq_exec * time)
    
    # Add random noise
    dmn_activity += np.random.normal(0, noise_level, n_samples)
    exec_activity += np.random.normal(0, noise_level, n_samples)
    
    # Add anti-correlation factor (when one goes up, the other tends to go down)
    anticorr_factor = 0.4
    dmn_activity -= anticorr_factor * exec_activity
    exec_activity -= anticorr_factor * dmn_activity
    
    # Simulate task engagement (executive network up, DMN down)
    if task_timing:
        for start, end in task_timing:
            start_idx = int(start * sampling_rate)
            end_idx = int(end * sampling_rate)
            
            # During task: boost executive network and suppress DMN
            task_mask = np.zeros(n_samples)
            task_mask[start_idx:end_idx] = 1
            
            exec_activity += 0.8 * task_mask
            dmn_activity -= 0.5 * task_mask
    
    # Normalize
    dmn_activity = (dmn_activity - np.mean(dmn_activity)) / np.std(dmn_activity)
    exec_activity = (exec_activity - np.mean(exec_activity)) / np.std(exec_activity)
    
    return time, dmn_activity, exec_activity

# Visualize network dynamics
def plot_network_timeseries(time, dmn_activity, exec_activity, task_timing=None):
    plt.figure(figsize=(12, 6))
    
    plt.plot(time, dmn_activity, 'b-', label='Default Mode Network', linewidth=2)
    plt.plot(time, exec_activity, 'r-', label='Executive Control Network', linewidth=2)
    
    # Highlight task periods
    if task_timing:
        for start, end in task_timing:
            plt.axvspan(start, end, color='gray', alpha=0.2)
            plt.text((start + end)/2, 1.8, 'Task', horizontalalignment='center')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Network Activity (z-score)')
    plt.title('Simulated DMN vs Executive Network Dynamics')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add correlation coefficient
    corr = np.corrcoef(dmn_activity, exec_activity)[0, 1]
    plt.text(0.02, 0.95, f'Correlation: {corr:.2f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.ylim(-2, 2)
    plt.tight_layout()
    plt.show()

# Create a graph visualization of brain networks
def visualize_brain_networks():
    # Create graph
    G = nx.Graph()
    
    # Define network nodes
    dmn_nodes = ['mPFC', 'PCC', 'Precuneus', 'Angular_L', 'Angular_R', 'MTL_L', 'MTL_R']
    exec_nodes = ['dlPFC_L', 'dlPFC_R', 'IPS_L', 'IPS_R', 'ACC', 'pre-SMA']
    
    # Add nodes
    G.add_nodes_from([(node, {'network': 'DMN'}) for node in dmn_nodes])
    G.add_nodes_from([(node, {'network': 'Executive'}) for node in exec_nodes])
    
    # Add intra-network edges (strong connections)
    for i, node1 in enumerate(dmn_nodes):
        for node2 in dmn_nodes[i+1:]:
            G.add_edge(node1, node2, weight=np.random.uniform(0.6, 0.9))
    
    for i, node1 in enumerate(exec_nodes):
        for node2 in exec_nodes[i+1:]:
            G.add_edge(node1, node2, weight=np.random.uniform(0.6, 0.9))
    
    # Add inter-network edges (weaker connections)
    for node1 in dmn_nodes:
        for node2 in exec_nodes:
            if np.random.random() < 0.3:  # Only some inter-network connections
                G.add_edge(node1, node2, weight=np.random.uniform(0.1, 0.4))
    
    # Create positions
    pos = {}
    # DMN nodes in a circle on the left
    angles = np.linspace(0, 2*np.pi, len(dmn_nodes), endpoint=False)
    radius = 10
    for i, node in enumerate(dmn_nodes):
        pos[node] = [radius * np.cos(angles[i]) - 15, radius * np.sin(angles[i])]
    
    # Executive nodes in a circle on the right
    angles = np.linspace(0, 2*np.pi, len(exec_nodes), endpoint=False)
    for i, node in enumerate(exec_nodes):
        pos[node] = [radius * np.cos(angles[i]) + 15, radius * np.sin(angles[i])]
    
    # Prepare for visualization
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    dmn_nodes_list = [node for node in G.nodes() if G.nodes[node]['network'] == 'DMN']
    exec_nodes_list = [node for node in G.nodes() if G.nodes[node]['network'] == 'Executive']
    
    nx.draw_networkx_nodes(G, pos, nodelist=dmn_nodes_list, node_color='blue', 
                          node_size=700, alpha=0.8, label='DMN')
    nx.draw_networkx_nodes(G, pos, nodelist=exec_nodes_list, node_color='red',
                          node_size=700, alpha=0.8, label='Executive Network')
    
    # Draw within-network edges
    dmn_edges = [(u, v) for u, v in G.edges() if G.nodes[u]['network'] == 'DMN' and G.nodes[v]['network'] == 'DMN']
    exec_edges = [(u, v) for u, v in G.edges() if G.nodes[u]['network'] == 'Executive' and G.nodes[v]['network'] == 'Executive']
    between_edges = [(u, v) for u, v in G.edges() if G.nodes[u]['network'] != G.nodes[v]['network']]
    
    # Get weights for edge thickness
    dmn_weights = [G[u][v]['weight'] * 5 for u, v in dmn_edges]
    exec_weights = [G[u][v]['weight'] * 5 for u, v in exec_edges]
    between_weights = [G[u][v]['weight'] * 5 for u, v in between_edges]
    
    nx.draw_networkx_edges(G, pos, edgelist=dmn_edges, width=dmn_weights, alpha=0.7, edge_color='blue')
    nx.draw_networkx_edges(G, pos, edgelist=exec_edges, width=exec_weights, alpha=0.7, edge_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=between_edges, width=between_weights, alpha=0.5, 
                           edge_color='gray', style='dashed')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title("Brain Networks: Default Mode Network & Executive Control Network")
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Simulate tDCS effects on network balance
def simulate_tdcs_effect(duration=300, stim_period=(100, 200), target_network='DMN'):
    """
    Simulate how tDCS might affect the balance between DMN and Executive Network
    
    Parameters:
    duration: total simulation time in seconds
    stim_period: tuple of (start, end) time for stimulation in seconds
    target_network: which network to stimulate ('DMN' or 'Executive')
    """
    # Standard simulation without stimulation
    time, dmn, exec_net = simulate_brain_networks(duration=duration)
    
    # Create a stimulation time series (effect of tDCS)
    stim_effect = np.zeros_like(time)
    stim_start_idx = np.where(time >= stim_period[0])[0][0]
    stim_end_idx = np.where(time >= stim_period[1])[0][0]
    
    # Ramp up effect
    ramp_duration = 10  # seconds for effect to reach maximum
    ramp_samples = int(ramp_duration * (len(time) / duration))
    
    # Ramp up
    ramp_indices = np.arange(stim_start_idx, min(stim_start_idx + ramp_samples, len(time)))
    stim_effect[ramp_indices] = np.linspace(0, 1, len(ramp_indices))
    
    # Plateau
    plateau_indices = np.arange(stim_start_idx + ramp_samples, stim_end_idx)
    stim_effect[plateau_indices] = 1.0
    
    # Ramp down
    ramp_down_indices = np.arange(stim_end_idx, min(stim_end_idx + ramp_samples, len(time)))
    stim_effect[ramp_down_indices] = np.linspace(1, 0, len(ramp_down_indices))
    
    # Apply effect to target network
    dmn_stim = dmn.copy()
    exec_stim = exec_net.copy()
    
    stim_magnitude = 0.8
    
    if target_network == 'DMN':
        # Stimulating DMN increases DMN and decreases Executive
        dmn_stim += stim_effect * stim_magnitude
        exec_stim -= stim_effect * (stim_magnitude * 0.6)  # Secondary effect on other network
    else:
        # Stimulating Executive increases Executive and decreases DMN
        exec_stim += stim_effect * stim_magnitude
        dmn_stim -= stim_effect * (stim_magnitude * 0.6)  # Secondary effect on other network
    
    # Re-normalize
    dmn_stim = (dmn_stim - np.mean(dmn_stim)) / np.std(dmn_stim)
    exec_stim = (exec_stim - np.mean(exec_stim)) / np.std(exec_stim)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, dmn, 'b-', label='DMN (baseline)', alpha=0.5)
    plt.plot(time, exec_net, 'r-', label='Executive (baseline)', alpha=0.5)
    plt.axvspan(stim_period[0], stim_period[1], color='yellow', alpha=0.2)
    plt.text((stim_period[0] + stim_period[1])/2, 1.8, f'tDCS: {target_network}', 
             horizontalalignment='center')
    plt.title('Baseline Network Dynamics')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-2, 2)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, dmn_stim, 'b-', label='DMN (with tDCS)', linewidth=2)
    plt.plot(time, exec_stim, 'r-', label='Executive (with tDCS)', linewidth=2)
    plt.axvspan(stim_period[0], stim_period[1], color='yellow', alpha=0.2)
    plt.title(f'Effect of tDCS on {target_network}')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-2, 2)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation changes due to stimulation
    baseline_corr = np.corrcoef(dmn, exec_net)[0, 1]
    stim_corr = np.corrcoef(dmn_stim, exec_stim)[0, 1]
    
    print(f"Baseline DMN-Executive correlation: {baseline_corr:.3f}")
    print(f"With tDCS to {target_network}, correlation: {stim_corr:.3f}")
    print(f"Change in anti-correlation: {stim_corr - baseline_corr:.3f}")

# Example calls:
# time_series, dmn, exec_net = simulate_brain_networks(task_timing=[(100, 150), (300, 350)])
# plot_network_timeseries(time_series, dmn, exec_net, task_timing=[(100, 150), (300, 350)])
# visualize_brain_networks()
# simulate_tdcs_effect(target_network='Executive')
```

### Cognition in AI

Exploring whether AI architectures explicitly incorporate something analogous to DMN: Some advanced AI might have a planning module (imagination) separate from perception modules. Research on generative models that simulate possible futures (planning algorithms in reinforcement learning that use Monte Carlo rollouts – conceptually similar to imagining scenarios) provides interesting parallels. While not a direct analog, it sparks discussion on modularizing AI cognition.

### Metacognition and Self-Supervision

The brain's ability to introspect (think about its own thoughts) via networks like the DMN can be loosely tied to AI models that evaluate or explain their own decisions. Current research on AI "metacognition" or self-monitoring (e.g., models that can detect when they might be wrong) represents an exciting area at the intersection of neuroscience-inspired AI.

### Suggested Readings

- Raichle et al. (2001), "A default mode of brain function," PNAS – Seminal paper that first named the Default Mode Network, discussing its discovery and functions.
- Buckner et al. (2008), "The brain's default network: Anatomy, function, and relevance to disease," Annals of the NY Academy of Sciences – Reviews the DMN and its role in internal mentation and how it contrasts with active tasks.
- Bzdok et al. (2015), "Intrinsic brain activity reveals memory states and predictive recall," Frontiers in Human Neuroscience – Connects DMN activity to memory and future imagination, relevant for thinking how an AI might simulate data (imagination) when not perceiving input.
- Nitsche & Paulus (2000), "Excitability changes induced in the human motor cortex by weak tDCS," Journal of Physiology – foundational study of tDCS effects.

### Supplementary Videos/Lectures

- Marcus Raichle – "The Brain's Default Mode Network" (lecture) – A talk by the scientist who coined the term "DMN," explaining its discovery and significance in accessible terms.
- YouTube: "Default Mode Network explained" (Psychology/Biology channel videos) – short videos describing what the DMN is and its link to daydreaming and self-referential thought.
- MIT Lecture – "Attention and Executive Control" – covers brain networks for attention (dorsal/ventral attention systems) and can complement understanding of task-positive networks in contrast to the DMN.
- BrainCraft video on tDCS – "Can Electricity Make You Smarter?" – a gentle introduction to what tDCS is, how people use it (even DIY), and what science says about its effects (cautioning that results are mixed).
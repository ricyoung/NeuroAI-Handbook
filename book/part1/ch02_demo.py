# Leaky-Integrate-and-Fire neuron + Hebbian 2-neuron network
# Author: R. Young & ChatGPT, 2025
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ---------- Part A: single LIF neuron ----------
def simulate_lif_neuron(T=300, dt=0.1, tau_m=20.0, V_rest=-70.0, V_thresh=-54.0, 
                        V_reset=-80.0, R_m=10.0, I_ext=1.8):
    """
    Simulate a leaky integrate-and-fire neuron.
    
    Parameters:
    -----------
    T : float
        Total simulation time in ms
    dt : float
        Time step in ms
    tau_m : float
        Membrane time constant in ms
    V_rest : float
        Resting potential in mV
    V_thresh : float
        Spike threshold in mV
    V_reset : float
        Reset potential after spike in mV
    R_m : float
        Membrane resistance in MΩ
    I_ext : float
        External current in nA
        
    Returns:
    --------
    V : array
        Membrane potential over time
    spikes : list
        Times of spikes in ms
    """
    steps = int(T/dt)
    
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
    
    return V, spikes

def plot_lif_simulation(V, spikes, dt=0.1, title="Leaky-Integrate-and-Fire Trace"):
    """Plot the membrane potential and spikes of the LIF neuron."""
    T = len(V) * dt
    time = np.arange(len(V)) * dt
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, V, lw=1.2)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title(title)
    
    for s in spikes:
        plt.axvline(s, color='r', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ---------- Part B: 2-neuron Hebbian network ----------
def simulate_hebbian_network(epochs=200, eta=0.01, pre_spike_prob=0.05, threshold=0.3, 
                            initial_weight_scale=0.1):
    """
    Simulate a simple 2-neuron network with Hebbian learning.
    
    Parameters:
    -----------
    epochs : int
        Number of training epochs
    eta : float
        Learning rate
    pre_spike_prob : float
        Probability of presynaptic neuron firing
    threshold : float
        Activation threshold for postsynaptic neuron
    initial_weight_scale : float
        Scale of initial random weights
        
    Returns:
    --------
    w : array
        Final weight matrix
    w_history : array
        History of weight matrices during training
    """
    # Initialize weights
    w = np.random.randn(2, 2) * initial_weight_scale
    w_history = []
    
    for epoch in range(epochs):
        # Generate random presynaptic activity
        pre = (np.random.rand(2) < pre_spike_prob).astype(float)
        
        # Compute postsynaptic input and activity
        post_input = pre @ w        # simple linear sum
        post = (post_input > threshold).astype(float)   # threshold non-linearity
        
        # Hebbian update: Δw = η * pre^T * post
        dw = eta * np.outer(pre, post)
        w += dw
        w_history.append(w.copy())
    
    w_history = np.array(w_history)
    return w, w_history

def plot_weight_evolution(w_history, title="Hebbian Weight Growth"):
    """Plot the evolution of weights during training."""
    plt.figure(figsize=(10, 4))
    
    # Reshape for visualization
    reshaped_w = w_history.transpose(1, 2, 0).reshape(4, -1)
    
    im = plt.imshow(reshaped_w, aspect='auto', cmap='magma', origin='lower')
    plt.colorbar(im, shrink=0.8, label='Weight value')
    plt.xlabel("Training step")
    plt.ylabel("Weight index")
    plt.title(title)
    
    plt.tight_layout()
    plt.show()

# ---------- Interactive demos ----------
def interactive_lif_demo(tau_values=[10, 20, 40], current_values=[1.0, 1.5, 1.8, 2.0]):
    """Interactive demo showing LIF neuron behavior with different parameters."""
    plt.figure(figsize=(15, 10))
    
    for i, tau_m in enumerate(tau_values):
        for j, I_ext in enumerate(current_values):
            plt.subplot(len(tau_values), len(current_values), i*len(current_values) + j + 1)
            
            V, spikes = simulate_lif_neuron(tau_m=tau_m, I_ext=I_ext)
            time = np.arange(len(V)) * 0.1
            
            plt.plot(time, V, lw=1.0)
            for s in spikes:
                plt.axvline(s, color='r', alpha=0.3, lw=0.5)
            
            if len(spikes) > 0:
                rate = len(spikes) / (time[-1] / 1000)  # spikes per second
                plt.title(f"τ={tau_m}ms, I={I_ext}nA\nRate: {rate:.1f} Hz")
            else:
                plt.title(f"τ={tau_m}ms, I={I_ext}nA\nNo spikes")
                
            if i == len(tau_values)-1:
                plt.xlabel("Time (ms)")
            if j == 0:
                plt.ylabel("Voltage (mV)")
    
    plt.tight_layout()
    plt.show()

def compare_hebbian_variants():
    """Compare different variants of Hebbian learning."""
    # Standard Hebbian rule
    w_standard, w_hist_standard = simulate_hebbian_network()
    
    # With inhibitory synapse
    w_init = np.random.randn(2, 2) * 0.1
    w_init[0, 1] = -0.2  # Make one synapse inhibitory
    
    w_inhib = w_init.copy()
    w_hist_inhib = [w_init.copy()]
    
    for epoch in range(200):
        pre = (np.random.rand(2) < 0.05).astype(float)
        post_input = pre @ w_inhib
        post = (post_input > 0.3).astype(float)
        
        dw = 0.01 * np.outer(pre, post)
        w_inhib += dw
        w_hist_inhib.append(w_inhib.copy())
    
    # With sigmoid activation instead of threshold
    def sigmoid(x, beta=5):
        return 1 / (1 + np.exp(-beta * x))
    
    w_sigmoid = np.random.randn(2, 2) * 0.1
    w_hist_sigmoid = [w_sigmoid.copy()]
    
    for epoch in range(200):
        pre = (np.random.rand(2) < 0.05).astype(float)
        post_input = pre @ w_sigmoid
        post = sigmoid(post_input, beta=5)  # Continuous activation
        
        dw = 0.01 * np.outer(pre, post)
        w_sigmoid += dw
        w_hist_sigmoid.append(w_sigmoid.copy())
    
    # Plot comparisons
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(np.array(w_hist_standard).transpose(1, 2, 0).reshape(4, -1), 
                    aspect='auto', cmap='magma', origin='lower')
    plt.colorbar(im1, shrink=0.8)
    plt.title("Standard Hebbian")
    plt.xlabel("Training step")
    plt.ylabel("Weight index")
    
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(np.array(w_hist_inhib).transpose(1, 2, 0).reshape(4, -1), 
                    aspect='auto', cmap='magma', origin='lower')
    plt.colorbar(im2, shrink=0.8)
    plt.title("With Inhibitory Synapse")
    plt.xlabel("Training step")
    
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(np.array(w_hist_sigmoid).transpose(1, 2, 0).reshape(4, -1), 
                    aspect='auto', cmap='magma', origin='lower')
    plt.colorbar(im3, shrink=0.8)
    plt.title("With Sigmoid Activation")
    plt.xlabel("Training step")
    
    plt.tight_layout()
    plt.show()

# ---------- Main execution ----------
if __name__ == "__main__":
    print("Part A: Leaky Integrate-and-Fire Neuron")
    # Basic simulation
    V, spikes = simulate_lif_neuron()
    plot_lif_simulation(V, spikes)
    
    print("\nPart B: Hebbian Learning in a 2-neuron network")
    # Basic simulation
    _, w_history = simulate_hebbian_network()
    plot_weight_evolution(w_history)
    
    print("\nInteractive demos:")
    print("1. LIF neuron with different parameters")
    interactive_lif_demo()
    
    print("\n2. Comparing Hebbian learning variants")
    compare_hebbian_variants()
    
    print("\nTry-this-now exercises:")
    print("1. Increase τₘ to 40 ms; observe slower decay & higher firing rate.")
    print("2. Make one synapse inhibitory in Part B by initialising it negative; trace how competition evolves.")
    print("3. Replace the binary post-neuron with a noisy sigmoid: σ(β · input) and watch weight saturation.")
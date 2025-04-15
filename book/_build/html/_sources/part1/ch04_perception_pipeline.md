# Chapter 4: Perception Pipeline – Visual Cortex → CNNs

## 4.0 Chapter Goals
- Trace the visual processing pipeline from retina to higher cortical areas
- Understand receptive fields and hierarchical feature extraction
- Connect biological vision to convolutional neural networks
- Implement simple visual filters and feature detectors

## 4.1 Visual System Architecture
- From retina to LGN to V1
- Ventral vs dorsal streams ("what" vs "where")
- Hierarchical organization of visual cortex
- Feedback connections and top-down processing

## 4.2 Receptive Fields
- Simple, complex, and hypercomplex cells
- Gabor filters and orientation selectivity
- Center-surround organization
- Feature hierarchy from V1 to IT

## 4.3 CNN Parallels
- Convolutional layers as receptive fields
- Pooling operations and invariance
- Feature hierarchies in deep networks
- Visualization of CNN features

## 4.4 From Biology to Deep Learning
- Hubel & Wiesel's discoveries → Neocognitron → LeNet → AlexNet
- Biological constraints vs engineering solutions
- Normalization and gain control mechanisms
- Attention mechanisms

## 4.5 Beyond Feedforward Processing
- Recurrence in visual processing
- Predictive coding and generative models
- Integration of context and prior knowledge
- The free energy principle

## 4.6 Code Lab
- Implementing simple Gabor filters
- Building a basic convolutional layer
- Feature visualization techniques
- Transfer learning with pre-trained CNNs

## 4.7 Take-aways
- Visual processing uses hierarchical feature extraction
- CNNs formalize key principles from visual neuroscience
- Both systems balance specificity with generalization

## 4.8 Further Reading & Media
- DiCarlo, Zoccolan & Rust (2012) - "How Does the Brain Solve Visual Object Recognition?"
- Kriegeskorte (2015) - "Deep Neural Networks: A New Framework for Modeling Biological Vision"
- Yamins & DiCarlo (2016) - "Using goal-driven deep learning models to understand sensory cortex"

---

## In-Depth Content

![Visual Cortex vs CNN](../figures/cortical_column_vs_ann.svg)

### The Human Visual System

The visual processing pipeline begins with the retina, where photoreceptors convert light into electrical signals. These signals pass through retinal ganglion cells, which perform the first stage of processing - edge detection and contrast enhancement through center-surround receptive fields. The signals then travel via the optic nerve to the lateral geniculate nucleus (LGN) of the thalamus, which acts as a relay station, before reaching the primary visual cortex (V1) in the occipital lobe.

In V1, neurons have small receptive fields that respond to oriented edges in specific parts of the visual field. As information flows to higher areas (V2, V4, IT cortex), neurons have progressively larger receptive fields and respond to more complex patterns - from corners and simple shapes to specific objects or faces. This hierarchical organization forms the basis for the layered architecture in artificial neural networks designed for visual processing.

### Receptive Fields and Hierarchies

A key discovery by Hubel and Wiesel in the 1960s was that neurons in the visual cortex act as feature detectors. Simple cells in V1 respond to oriented edges at specific locations, complex cells respond to oriented edges regardless of exact position, and hypercomplex cells (end-stopped cells) respond to edges of specific lengths or corners.

The receptive field of a visual neuron is the region of visual space that, when stimulated, affects the firing of that neuron. As visual information progresses through the cortical hierarchy, receptive fields become larger and encode more complex features:

1. V1 neurons: Oriented edges, spatial frequency, color
2. V2 neurons: Contours, figure-ground separation
3. V4 neurons: Shape features, curvature
4. IT neurons: Complex objects, faces

This progressive abstraction of visual features - from simple to complex - is the key insight that informed the development of convolutional neural networks.

### Convolutional Neural Networks

CNNs mirror the architecture of the visual cortex in several key ways:

1. **Convolutional layers**: Like visual cortex neurons, CNN filters scan across the image and respond to specific patterns within their receptive field. Early layers detect simple features (edges, textures) while deeper layers combine these to detect complex objects.

2. **Local connectivity**: Rather than connecting to every input pixel (as in fully-connected networks), CNN neurons connect only to a small region, similar to the receptive fields in the visual system.

3. **Hierarchical processing**: CNNs stack multiple layers, progressively building more complex representations - from edges to textures to object parts to whole objects.

4. **Pooling operations**: These provide translation invariance by summarizing features across small spatial regions, similar to how complex cells in V1 respond to oriented edges regardless of their exact position.

Individual CNN neurons, like cortical neurons, respond to inputs in a restricted region (their receptive field) and collectively form a retinotopic map of visual space.

### Examples of Biological Inspiration

Hubel and Wiesel's discovery of edge-detecting cells in the cat visual cortex foreshadowed the edge-detecting filters learned by the first layer of CNNs. When visualized, the filters learned by the first convolutional layer of networks like AlexNet show striking similarity to the Gabor-like filters found in V1 neurons.

### Python/PyTorch Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Implementing Gabor filters - similar to V1 simple cell receptive fields
def gabor_filter(size, lambda_val, theta, sigma, gamma, psi):
    """
    Create a Gabor filter (similar to V1 simple cell receptive fields)
    
    Parameters:
    size (int): Size of the filter
    lambda_val (float): Wavelength
    theta (float): Orientation in radians
    sigma (float): Standard deviation of the Gaussian envelope
    gamma (float): Spatial aspect ratio
    psi (float): Phase offset
    
    Returns:
    np.array: Gabor filter kernel
    """
    x, y = np.meshgrid(np.arange(-size//2, size//2+1), np.arange(-size//2, size//2+1))
    
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    # Gabor function
    gb = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(2 * np.pi * x_theta / lambda_val + psi)
    
    return gb

# Create and visualize a bank of Gabor filters with different orientations
def visualize_gabor_bank():
    # Parameters
    size = 31
    lambda_val = 10.0
    sigma = 5.0
    gamma = 0.5
    psi = 0
    
    # Create filters at different orientations
    orientations = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]
    filters = [gabor_filter(size, lambda_val, theta, sigma, gamma, psi) for theta in orientations]
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, (filt, theta) in enumerate(zip(filters, orientations)):
        axes[i].imshow(filt, cmap='gray')
        axes[i].set_title(f'Orientation: {theta*180/np.pi:.1f}°')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return filters

# A simple CNN to demonstrate visual processing hierarchy
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv layer - like V1 simple cells (edge detectors)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        # Second conv layer - like V2 (combining edges into shapes)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        # Third conv layer - like V4 (more complex shapes)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Final layer - like IT (object recognition)
        self.fc = nn.Linear(64 * 3 * 3, 10)  # For MNIST digits (10 classes)
        
    def forward(self, x):
        # Hierarchical feature extraction
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Conv + pooling (like complex cells)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Greater invariance
        x = F.relu(F.max_pool2d(self.conv3(x), 2))  # Higher-level features
        x = x.view(-1, 64 * 3 * 3)  # Flatten
        x = self.fc(x)  # Classification
        return x

# Function to visualize the activations at different layers - like recording from different visual areas
def visualize_layer_activations(model, image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Hook to capture activations
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    hooks.append(model.conv1.register_forward_hook(hook_fn('conv1')))
    hooks.append(model.conv2.register_forward_hook(hook_fn('conv2')))
    hooks.append(model.conv3.register_forward_hook(hook_fn('conv3')))
    
    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize activations from different layers
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    
    # First layer (V1-like)
    for i in range(5):
        if i < activations['conv1'].size(1):
            axes[0, i].imshow(activations['conv1'][0, i].numpy(), cmap='viridis')
            axes[0, i].set_title(f'V1-like Filter {i+1}')
            axes[0, i].axis('off')
    
    # Second layer (V2-like)
    for i in range(5):
        if i < activations['conv2'].size(1):
            axes[1, i].imshow(activations['conv2'][0, i].numpy(), cmap='viridis')
            axes[1, i].set_title(f'V2-like Filter {i+1}')
            axes[1, i].axis('off')
    
    # Third layer (V4-like)
    for i in range(5):
        if i < activations['conv3'].size(1):
            axes[2, i].imshow(activations['conv3'][0, i].numpy(), cmap='viridis')
            axes[2, i].set_title(f'V4-like Filter {i+1}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Using a pre-trained model to visualize hierarchical features
def visualize_pretrained_features():
    # Load pre-trained VGG16 model
    vgg = models.vgg16(pretrained=True)
    
    # Print the model architecture to see the hierarchical organization
    print(vgg.features)
    
    # Plot filters from the first convolutional layer (edge detectors like V1)
    filters = vgg.features[0].weight.data.numpy()
    
    # Normalize filter values to 0-1 range for display
    filters = (filters - filters.min()) / (filters.max() - filters.min())
    
    # Plot first 16 filters
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(16):
        axes[i].imshow(filters[i, 0])
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# gabor_bank = visualize_gabor_bank()
# model = SimpleCNN()
# visualize_layer_activations(model, 'path_to_image.jpg')
# visualize_pretrained_features()
```

### Suggested Readings

- Hubel & Wiesel (1962), "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex," Journal of Physiology – Classic paper detailing edge-detecting neurons (historical context for CNN inspiration).
- Fukushima (1980), "Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position," Biol. Cybernetics – Early CNN-like model influenced by neuroscience.
- Yamins & DiCarlo (2016), "Using goal-driven deep learning models to understand sensory cortex," Nature Neuroscience – Demonstrates that deep CNNs not only draw inspiration from the brain but can predict neural responses in the primate visual system, validating CNNs as models of vision.

### Supplementary Videos/Lectures

- TED-Ed: "How Your Eyes Make Sense of the World" – Visual explanation of the human visual processing pathway.
- Stanford CS231n (Convolutional Neural Networks for Visual Recognition) – Lecture 1 – Introduction that connects biological vision to CNN design (includes historical notes on Hubel & Wiesel).
- "Do convolutional neural networks mimic the human visual system?" (MIT Quest for Intelligence seminar) – Discusses the parallels and differences between CNNs and actual brain networks for vision.
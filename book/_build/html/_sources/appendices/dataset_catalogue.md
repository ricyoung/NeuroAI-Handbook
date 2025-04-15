# Appendix B: Dataset Catalogue

## B.1 Neuroscience Datasets

### Electrophysiology Data
- Allen Brain Observatory
  - Single-cell recordings from visual cortex
  - Access: https://observatory.brain-map.org/
- Collaborative Research in Computational Neuroscience (CRCNS)
  - Diverse electrophysiology datasets
  - Access: https://crcns.org/
- Neurodata Without Borders (NWB)
  - Standardized electrophysiology datasets
  - Access: https://www.nwb.org/

### Neuroimaging Data
- Human Connectome Project (HCP)
  - Large-scale fMRI, diffusion MRI, and MEG data
  - Access: https://www.humanconnectome.org/
- OpenNeuro
  - Repository for fMRI, EEG, MEG, iEEG
  - Access: https://openneuro.org/
- NeuroVault
  - Repository for statistical maps of the brain
  - Access: https://neurovault.org/

### Behavior and Cognition
- International Brain Laboratory (IBL)
  - Mouse behavioral and neural data
  - Access: https://www.internationalbrainlab.com/
- Brain Imaging Data Structure (BIDS) Examples
  - Standardized cognitive neuroscience datasets
  - Access: https://github.com/bids-standard/bids-examples

## B.2 AI Benchmark Datasets

### Computer Vision
- ImageNet
  - Image classification benchmark
  - Access: https://www.image-net.org/
- CIFAR-10/100
  - Small-scale image classification
  - Access: https://www.cs.toronto.edu/~kriz/cifar.html
- MS COCO
  - Object detection and segmentation
  - Access: https://cocodataset.org/

### Natural Language Processing
- GLUE Benchmark
  - General Language Understanding Evaluation
  - Access: https://gluebenchmark.com/
- SQuAD
  - Stanford Question Answering Dataset
  - Access: https://rajpurkar.github.io/SQuAD-explorer/
- WikiText
  - Language modeling dataset
  - Access: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/

### Reinforcement Learning
- OpenAI Gym
  - RL environment collection
  - Access: https://gym.openai.com/
- DeepMind Lab
  - 3D learning environment
  - Access: https://github.com/deepmind/lab
- MuJoCo
  - Physics-based control tasks
  - Access: https://github.com/openai/mujoco-py

## B.3 NeuroAI Specific Datasets

### Brain-Score
- Benchmark for neural network models of the visual system
- Access: https://www.brain-score.org/

### Neural Data Challenge
- Neural decoding competitions
- Access: https://neurodatachallenge.org/

### Algonauts Project
- Bridging human and machine vision
- Access: http://algonauts.csail.mit.edu/

## B.4 Data Loading Examples

```python
# Example: Loading Allen Brain Observatory data
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import matplotlib.pyplot as plt

# Initialize the cache
boc = BrainObservatoryCache(manifest_file='boc_manifest.json')

# Get experiment containers for specific targeted structures and imaging depths
conts = boc.get_experiment_containers(targeted_structures=['VISp'], 
                                     imaging_depths=[175])

# Access an example experiment ID
experiment_id = boc.get_ophys_experiments(
    experiment_container_ids=[conts[0]['id']],
    stimuli=['natural_scenes'])[0]['id']

# Get neural data
data_set = boc.get_ophys_experiment_data(experiment_id)

# Get the fluorescence traces
ts, traces = data_set.get_fluorescence_traces()

# Plot example trace
plt.figure(figsize=(10, 4))
plt.plot(ts, traces[0])
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence')
plt.title('Example Neuron Activity')
plt.show()
```
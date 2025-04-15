# Appendix C: Colab Setup Tips

## C.1 Getting Started with Google Colab

### What is Google Colab?
- Free cloud-based Jupyter notebook environment
- No setup required, runs in browser
- Access to GPUs and TPUs for accelerated computation
- Integrated with Google Drive for storage

### Basic Setup
- Accessing Colab: https://colab.research.google.com/
- Notebook overview and interface
- Cell types: code, text (markdown), and output
- Runtime management and memory limits

## C.2 Environment Configuration

### Package Installation
```python
# Install packages not included in Colab's default environment
!pip install -q neurom allensdk seaborn nibabel mne

# Verify installation
import neurom
import allensdk
import mne
print(f"MNE version: {mne.__version__}")
print(f"neurom version: {neurom.__version__}")
```

### Managing GPU Resources
```python
# Check if GPU is available
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Check GPU memory usage
!nvidia-smi
```

### Setting Up Custom Paths
```python
# Create directories for data storage
!mkdir -p ./data/raw ./data/processed ./models ./results

# Define paths
RAW_DATA_PATH = './data/raw'
PROCESSED_DATA_PATH = './data/processed'
MODEL_PATH = './models'
RESULTS_PATH = './results'
```

## C.3 Data Management

### Mounting Google Drive
```python
from google.colab import drive
drive.mount('/content/gdrive')

# Create path to your Drive folder
DRIVE_PATH = '/content/gdrive/My Drive/NeuroAI-Handbook'
!mkdir -p $DRIVE_PATH
```

### Downloading Datasets
```python
# Example: Download a dataset from a URL
!wget -P $RAW_DATA_PATH https://example.com/dataset.zip
!unzip -q $RAW_DATA_PATH/dataset.zip -d $RAW_DATA_PATH

# Example: Get data from Kaggle
!pip install -q kaggle
!mkdir -p ~/.kaggle
!echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d dataset/name -p $RAW_DATA_PATH
```

### Persistent Storage Solutions
- Saving to mounted Google Drive
- Using Google Cloud Storage
- Exporting and importing notebooks with data

## C.4 Collaboration Features

### Sharing Notebooks
- Generating shareable links
- Collaborative editing
- Version history and comments

### Publishing and Embedding
- GitHub integration
- Saving to GitHub Gist
- Embedding in websites or blogs

## C.5 Advanced Tips

### Monitoring and Debugging
```python
# Memory management monitoring
!pip install -q psutil
import psutil

def print_memory_usage():
    """Print current memory usage of the Colab instance."""
    mem = psutil.virtual_memory()
    print(f"Total memory: {mem.total / 1e9:.1f} GB")
    print(f"Available memory: {mem.available / 1e9:.1f} GB")
    print(f"Used memory: {mem.used / 1e9:.1f} GB ({mem.percent}%)")
    
print_memory_usage()
```

### Automated Visualization Setup
```python
# Standard visualization setup for neuroscience plots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_visualization(style='whitegrid', context='talk', palette='viridis'):
    """Set up standardized visualization environment."""
    sns.set_theme(style=style, context=context, palette=palette)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    return "Visualization environment configured"

setup_visualization()
```

### Avoiding Session Timeouts
```python
# Function to prevent Colab from disconnecting (use cautiously)
from IPython.display import display, Javascript
import time

def keep_alive(delay_minutes=55):
    """Prevents Colab from disconnecting by clicking connect button periodically."""
    delay_seconds = delay_minutes * 60
    display(Javascript('''
        function click_connect(){
            console.log("Clicking connect button");
            document.querySelector("colab-connect-button").click()
        }
        setInterval(click_connect, ''' + str(delay_seconds * 1000) + ''');
    '''))
    print(f"Keep-alive service started. Will refresh every {delay_minutes} minutes.")

# Uncomment to use (be considerate of resources)
# keep_alive(delay_minutes=55)
```
# The Neuroscience of AI Handbook

[![build-and-deploy-book](https://github.com/neuralinterfacinglab/NeuroAI-Handbook/actions/workflows/book.yml/badge.svg)](https://github.com/neuralinterfacinglab/NeuroAI-Handbook/actions/workflows/book.yml)

A comprehensive handbook bridging neuroscience and artificial intelligence concepts, with practical Python implementations and interactive code examples.

## About This Book

This handbook explores the intersection of neuroscience and artificial intelligence, showing how biological principles inspire computational models. The content progresses from fundamental neuroscience to cutting-edge AI architectures.

### Structure

1. **Part I: Brains & Inspiration**
   - ✅ Introduction to Neuroscience ↔ AI
   - ✅ Neuroscience Foundations for AI
   - ✅ Spatial Navigation – Place & Grid Cells
   - ✅ Perception Pipeline – Visual Cortex → CNNs

2. **Part II: Brains Meet Math & Data**
   - ✅ Default-Mode vs Executive Control Networks
   - ✅ Neuro-stimulation & Plasticity
   - ✅ Information Theory Essentials
   - ✅ Data-Science Pipeline in Python

3. **Part III: Learning Machines**
   - ✅ Classical Machine-Learning Foundations
   - ✅ Deep Learning: Training & Optimisation
   - ✅ Sequence Models: RNN → Attention → Transformer

4. **Part IV: Frontier Models**
   - ✅ Large Language Models & Fine-Tuning
   - ✅ Multimodal & Diffusion Models

5. **Part V: Reflection & Futures**
   - ✅ Where Next for Neuro-AI?

6. **Appendices**
   - ✅ Comprehensive glossary with cross-references
   - ✅ Math & Python mini-refresher
   - ✅ Dataset catalogue
   - ✅ Colab setup tips

### Implemented Content

#### Core Chapters

- **Chapter 1: Introduction to Neuroscience ↔ AI** - Overview of the bidirectional relationship between neuroscience and artificial intelligence, with historical context and future directions.
- **Chapter 2: Neuroscience Foundations for AI** - Core neuroscience concepts relevant to AI, including neuron anatomy, neural circuits, and brain organization with Python implementations.
- **Chapter 3: Spatial Navigation – Place & Grid Cells** - Detailed exploration of spatial cognition in the brain, including place cells, grid cells, and cognitive maps with computational models and implementations.
- **Chapter 4: Perception Pipeline – Visual Cortex → CNNs** - Analysis of the visual processing pathways in the brain and their parallels to convolutional neural networks, with implementations for visual processing tasks.
- **Chapter 5: Default-Mode vs Executive Control Networks** - Comprehensive exploration of large-scale brain networks, their dynamics, functional significance, and computational modeling with Python implementations for network analysis, visualization, and simulation.
- **Chapter 6: Neurostimulation & Plasticity** - In-depth coverage of neural plasticity mechanisms, neuromodulatory systems, and neurostimulation techniques with computational models connecting neuroscience principles to AI algorithms.
- **Chapter 7: Information Theory Essentials** - Mathematical foundations of information theory with applications to neural coding and AI, including implementations for entropy, mutual information, and efficient coding.
- **Chapter 8: Data-Science Pipeline in Python** - Comprehensive workflow for neural data analysis, including preprocessing, feature extraction, and machine learning applications with detailed code examples.
- **Chapter 9: Classical Machine-Learning Foundations** - Implementation of key ML algorithms with neural data applications, including supervised and unsupervised learning techniques, with visual diagrams illustrating learning paradigms, bias-variance tradeoff, feature selection methods, and neuroscience-ML parallels.
- **Chapter 10: Deep Learning: Training & Optimisation** - Foundations of neural networks, backpropagation, optimization techniques, regularization methods, and connections to biological learning with PyTorch implementations. Features comprehensive SVG diagrams illustrating neural network architectures, optimization algorithms, regularization techniques, advanced architectures, and biological parallels.
- **Chapter 11: Sequence Models: RNN → Attention → Transformer** - Evolution of sequence processing from RNNs through attention mechanisms to transformer architectures, with biological parallels and practical implementations.
- **Chapter 12: Large Language Models & Fine-Tuning** - Comprehensive overview of LLM architectures, fine-tuning approaches (including LoRA and RLHF), prompting techniques, and connections to neural language processing, with practical code labs for implementation.
- **Chapter 13: Multimodal & Diffusion Models** - In-depth exploration of multimodal learning architectures, diffusion model principles, text-to-image generation, and neural multimodal integration, with detailed Python implementations and connections to biological multisensory processing.
- **Chapter 14: Where Next for Neuro-AI?** - Exploration of frontier research directions including neuromorphic computing, continual learning, AI for neuroscience, whole-brain integration, and ethical considerations of brain-inspired AI, with implementable code examples for spiking neural networks and other emerging technologies.

#### Appendices

- **Appendix A: Comprehensive Glossary** - Alphabetically organized glossary of key terms in neuroscience, information theory, machine learning, and artificial intelligence with cross-references to relevant chapters where concepts are discussed in detail.
- **Appendix B: Math & Python Mini-Refresher** - Comprehensive review of essential mathematical concepts (linear algebra, calculus, probability) and Python programming fundamentals for NeuroAI research, with practical examples using NumPy, SciPy, Matplotlib, and other scientific computing libraries.
- **Appendix C: Dataset Catalogue** - Extensive collection of neuroscience datasets (Allen Brain Atlas, Human Connectome Project), AI benchmark datasets, and NeuroAI-specific datasets with examples for data loading, preprocessing, and analysis.
- **Appendix D: Colab Setup for NeuroAI** - Detailed guide for setting up and optimizing Google Colab for neuroscience and AI research, including environment configuration, GPU utilization, data management, memory optimization, and visualization techniques specific to NeuroAI applications.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter notebook/lab
- JupyterBook (for building the book)

### Installation

```bash
# Clone the repository
git clone https://github.com/neuralinterfacinglab/NeuroAI-Handbook.git
cd NeuroAI-Handbook

# Install dependencies
pip install -r book/requirements.txt

# Install JupyterBook if you plan to build the book locally
pip install jupyter-book
```

### Initialization Script

For a more convenient setup, use the provided initialization script:

```bash
# Run the initialization script to set up the environment
python init_neuroai.py --all
```

The initialization script provides several options:
- `--install-deps`: Install required dependencies
- `--check-env`: Check the environment for required packages
- `--create-dirs`: Create standard directory structure for exercises
- `--all`: Perform all setup tasks (default if no flags provided)

### Building the Book

#### Building the HTML Version
```bash
# Build the book
jb build book

# View the book locally (open http://localhost:8000 in your browser)
python -m http.server -d book/_build/html
```

#### Building the PDF Version
```bash
# Build the complete PDF (creates a single PDF with page numbers)
python build_neuroai_handbook.py

# The final PDF will be in pdf_exports/complete_handbook/neuroai_handbook.pdf
```

The build process includes three main steps:
1. Building all individual chapter PDFs using jupyter-book
2. Creating supporting pages (cover, TOC, dividers, etc.)
3. Combining everything into a single PDF with page numbers

You can also customize the build with a custom output directory:
```bash
python build_neuroai_handbook.py --output /path/to/output
```

#### Custom Cover Page
To use a custom cover page, place a `cover.pdf` file in the `_assets` directory.

#### Opening the PDF
Once built, you can quickly open the PDF with:
```bash
python open_pdf.py
```

### Running the Examples

Each chapter contains executable code examples that can be run directly in the Jupyter Book interface or in Jupyter notebooks:

1. **Interactive Web Version**: Visit the [online version](https://neuralinterfacinglab.github.io/NeuroAI-Handbook) to run code examples directly in your browser
2. **Google Colab**: Most examples include "Open in Colab" buttons for easy execution in the cloud
3. **Local Execution**: Run the notebooks locally from the `book/` directory
```

### Required Libraries

The examples in this handbook use the following Python libraries:

#### Core Scientific Python
- **NumPy** and **SciPy** for numerical operations and scientific computing
- **Matplotlib** and **Seaborn** for data visualization
- **Pandas** for data manipulation and analysis
- **Jupyter** for interactive notebooks

#### Neuroscience-Specific
- **MNE-Python** for neurophysiological data processing (EEG/MEG)
- **NiBabel** and **Nilearn** for neuroimaging data
- **Neo** for electrophysiology data
- **AllenSDK** for Allen Brain Atlas data access
- **NeuroM** for morphological analysis of neurons
- **PyNN** for neural simulations

#### Machine Learning & AI
- **scikit-learn** for classical machine learning implementations
- **PyTorch** for deep learning
- **TensorFlow** and **Keras** (alternative deep learning frameworks)
- **Transformers** (Hugging Face) for working with language models
- **Diffusers** for diffusion model implementations

#### Specialized Libraries
- **NetworkX** and **Brain Connectivity Toolbox** for network analysis
- **statsmodels** for statistical modeling
- **Information Theoretic Learning** and **pyEntropy** for information theory applications
- **Optuna** for hyperparameter optimization
- **Biopython** for biological sequence analysis

## Contributing

We welcome contributions to the NeuroAI Handbook! Here's how you can help:

1. **Report Issues**: If you find errors or have suggestions, please [open an issue](https://github.com/neuralinterfacinglab/NeuroAI-Handbook/issues)
2. **Submit Improvements**: Feel free to submit pull requests with corrections or additional content
3. **Share Examples**: If you have interesting neuroscience or AI examples, consider contributing them

### Contribution Guidelines

- Follow the existing code style and documentation patterns
- For new content, maintain consistency with the handbook's structure
- Include appropriate citations for scientific claims
- Test any code examples before submitting
- Update relevant documentation when adding features

## Maintenance and Troubleshooting

### SVG Figure Maintenance

When creating or editing SVG figures, be aware of these common issues:

1. **XML Escaping**: Always properly escape special characters in SVG files:
   - Replace `&` with `&amp;`
   - Replace `<` with `&lt;`
   - Replace `>` with `&gt;`
   - Replace `"` with `&quot;`

2. **SVG Validation**: Consider validating SVG files with a tool like [W3C Validator](https://validator.w3.org/) before including them in the book.

3. **Troubleshooting Display Issues**: If a figure doesn't display correctly:
   - Check for unescaped special characters (especially `&` in text elements)
   - Verify that fonts used in the SVG are standard or embedded
   - Check for invalid XML in the SVG file

### Interactive Content Requirements

For running the interactive Jupyter notebooks:
- Ensure all required packages are installed (`torch`, `matplotlib`, etc.)
- Some notebooks require specific Python packages that need to be installed manually
- Check `book/requirements.txt` for core dependencies

## License

This project is licensed under multiple licenses:
- Content: [LICENSE-content](LICENSE-content)
- Code: [LICENSE-code](LICENSE-code)

© 2025 Richard Young. All rights reserved.
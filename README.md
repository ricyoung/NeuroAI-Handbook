# The Neuroscience of AI Handbook

[![build-and-deploy-book](https://github.com/neuralinterfacinglab/NeuroAI-Handbook/actions/workflows/book.yml/badge.svg)](https://github.com/neuralinterfacinglab/NeuroAI-Handbook/actions/workflows/book.yml)
[![License: Content](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE-content)
[![License: Code](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-code)

A comprehensive handbook bridging neuroscience and artificial intelligence, featuring 24 chapters with practical Python implementations, interactive examples, and cutting-edge research at the intersection of brain science and AI.

## About This Book

This handbook serves as a comprehensive guide to the bidirectional relationship between neuroscience and artificial intelligence. It demonstrates how biological neural systems inspire AI architectures and how AI methods advance our understanding of the brain.

### Key Features

- 📚 **24 comprehensive chapters** covering neuroscience fundamentals through advanced AI
- 💻 **Practical Python implementations** with executable code examples
- 🧠 **Biological foundations** for major AI breakthroughs
- 🔬 **Real-world applications** in neuroscience research and AI development
- 📊 **Interactive visualizations** and hands-on exercises
- 🎓 **Suitable for** researchers, students, and practitioners in both fields

### Book Structure

#### Part I: Brains & Inspiration
1. **Introduction to Neuroscience ↔ AI** - Historical context and bidirectional influence
2. **Neuroscience Foundations for AI** - Core concepts with Python implementations
3. **Spatial Navigation** - Place cells, grid cells, and cognitive maps
4. **Perception Pipeline** - Visual cortex to CNNs

#### Part II: Brains Meet Math & Data
5. **Brain Networks** - Default-mode vs executive control networks
6. **Neurostimulation & Plasticity** - Neural adaptation and learning
7. **Information Theory Essentials** - Mathematical foundations for neural coding
8. **Data-Science Pipeline** - Neural data analysis in Python

#### Part III: Learning Machines
9. **Classical Machine Learning** - Foundations with neural applications
10. **Deep Learning** - Training, optimization, and biological parallels
11. **Sequence Models** - RNNs to Transformers evolution

#### Part IV: Frontier Models
12. **Large Language Models** - Architecture, fine-tuning, and neural language
13. **Multimodal & Diffusion Models** - Cross-modal learning and generation

#### Part V: Ethics & Future
14. *[Placeholder for Ethics chapter]*
15. **Ethical AI** - Responsible development of brain-inspired AI
16. **Future Directions** - Emerging frontiers in NeuroAI

#### Part VI: Advanced Topics
17. **Brain-Computer Interfaces** - Neural decoding and human-AI interaction
18. **Neuromorphic Computing** - Brain-inspired hardware
19. **Cognitive Neuroscience meets Deep Learning** - Representational analysis
20. **Case Studies** - Real-world NeuroAI applications
21. **AI for Neuroscience Discovery** - ML-driven brain research
22. **Embodied AI & Robotics** - Sensorimotor integration
23. **Lifelong Learning** - Continual adaptation in brains and AI
24. **Quantum Computing & NeuroAI** - Future computational paradigms

#### Appendices
- **A: Comprehensive Glossary** - Key terms with cross-references
- **B: Math & Python Refresher** - Essential foundations
- **C: Dataset Catalogue** - Neuroscience and AI datasets
- **D: Colab Setup Guide** - Optimized cloud computing

## Current Status

### ✅ Completed Content

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

#### Supporting Materials

- **Appendix A: Comprehensive Glossary** - 200+ key terms with cross-references
- **Appendix B: Math & Python Mini-Refresher** - Essential mathematical and programming foundations
- **Appendix C: Dataset Catalogue** - Curated neuroscience and AI datasets with loading examples
- **Appendix D: Colab Setup for NeuroAI** - Cloud computing optimization guide

### 🔧 Recent Updates

- **Fixed:** All SVG figures now have properly escaped XML characters
- **Fixed:** Missing viewBox attributes added to 49 SVG files
- **Validated:** All image references verified and working
- **Identified:** Content gaps for future expansion (see Contributing section)

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

### Priority Areas for Contribution

1. **Missing Standard Sections**: Chapters 4-8, 11-12 need "Take-aways" and "Further Reading" sections
2. **Placeholder Code**: Chapters 20, 22, 23 contain placeholder implementations to be expanded
3. **New Content Areas**:
   - Reinforcement Learning (dopamine pathways to Deep RL)
   - Graph Neural Networks for brain connectivity
   - Spiking Neural Networks with practical examples
   - Foundation models for neuroscience
   - Computational psychiatry applications

### How to Contribute

1. **Report Issues**: [Open an issue](https://github.com/neuralinterfacinglab/NeuroAI-Handbook/issues) for errors or suggestions
2. **Submit PRs**: Corrections, new examples, or content expansions welcome
3. **Add Examples**: Share interesting NeuroAI implementations or case studies

### Contribution Guidelines

- Follow existing code style (PEP 8 for Python)
- Use NumPy-style docstrings
- Include citations for scientific claims
- Test code examples before submitting
- Escape special characters in SVG files
- Clear Jupyter notebook outputs before committing

## Technical Notes

### SVG Figure Guidelines

- **Always escape** special characters: `&` → `&amp;`, `<` → `&lt;`, `>` → `&gt;`, `"` → `&quot;`
- **Include viewBox** attributes for proper scaling
- **Validate** SVGs before committing ([W3C Validator](https://validator.w3.org/))
- **Use standard fonts** or embed custom fonts

### Build Troubleshooting

- **PDF build fails**: Check `logs/` directory for detailed error messages
- **Missing figures**: Verify SVG files are valid XML
- **Notebook errors**: Ensure all dependencies in `requirements.txt` are installed
- **Memory issues**: Use `--low-memory` flag for large builds

### Performance Tips

- Build individual chapters during development: `jupyter-book build book/part1/ch01_intro.md --builder pdfhtml`
- Use cached builds when possible
- Clear build artifacts if encountering issues: `jb clean book`

## Citation

If you use this handbook in your research or teaching, please cite:

```bibtex
@book{young2025neuroai,
  title={The Neuroscience of AI Handbook},
  author={Young, Richard},
  year={2025},
  publisher={Neural Interfacing Lab},
  url={https://github.com/neuralinterfacinglab/NeuroAI-Handbook}
}
```

## License

- **Content**: [Creative Commons BY-NC-SA 4.0](LICENSE-content) - Share and adapt with attribution
- **Code**: [MIT License](LICENSE-code) - Use freely in your projects

© 2025 Richard Young. All rights reserved.

---

<p align="center">
  <a href="https://neuralinterfacinglab.github.io/NeuroAI-Handbook">📖 Read Online</a> •
  <a href="https://github.com/neuralinterfacinglab/NeuroAI-Handbook/issues">🐛 Report Issues</a> •
  <a href="https://github.com/neuralinterfacinglab/NeuroAI-Handbook/discussions">💬 Discussions</a>
</p>
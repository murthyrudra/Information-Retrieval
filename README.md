# Introduction to Information Retrieval

This tutorial covers the materials on Information Retrieval. The tutorial is hosted on GitHub Pages using Material for MkDocs. To view it online, go to:

<https://murthyrudra.github.io/Information-Retrieval/>

## Project Structure

```
information-retrieval/
├── docs/               # Documentation files for MkDocs
├── notebooks/          # Jupyter notebooks containing tutorials and examples
│   ├── ilab_sdg/      # Specialized notebooks for specific use cases
│   ├── *.ipynb        # Main tutorial notebooks
│   └── *.md           # Markdown versions of notebooks
├── src/               # Source code and utility functions
├── .github/           # GitHub configuration files
├── venv/              # Python virtual environment (not tracked in git)
└── requirements.txt   # Python package dependencies
```

### Directory Descriptions

- `docs/`: Contains the documentation source files that are used to generate the GitHub Pages website
- `notebooks/`: Contains all Jupyter notebooks and their markdown versions
  - `ilab_sdg/`: Contains specialized notebooks for specific use cases
  - Main notebooks include:
    - `boolean_retriever.ipynb`: Boolean retrieval implementation
    - `tf_idf.ipynb`: TF-IDF implementation
    - `NeuralRetriever.ipynb`: Neural network-based retrieval
- `src/`: Contains Python source code, utility functions, and helper modules
- `.github/`: Contains GitHub-specific configurations and workflows
- `venv/`: Local Python virtual environment (not tracked in git)

## Running Locally with JupyterLab

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- For GPU support (optional):
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit 11.x or 12.x
  - cuDNN

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/murthyrudra/Information-Retrieval.git
cd Information-Retrieval
```

2. Create and activate a virtual environment:

For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

For Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install jupyterlab
pip install -r requirements.txt
```

4. Install FAISS:

For CPU-only version (recommended for most users):
```bash
# For Windows
pip install faiss-cpu

# For Linux
pip install faiss-cpu
```

For GPU version (if you have NVIDIA GPU):
```bash
# For Windows
pip install faiss-gpu

# For Linux
pip install faiss-gpu
```

To verify FAISS installation:
```python
import faiss
print(faiss.__version__)  # Should print the version number
```

5. Download Required NLTK Data:
```python
import nltk

# Download essential NLTK data
nltk.download('punkt')      # For tokenization
nltk.download('stopwords')  # For stop words
nltk.download('wordnet')    # For lemmatization
nltk.download('averaged_perceptron_tagger')  # For POS tagging
```

You can run this code in a Python console or add it to your notebook. The data will be downloaded to your NLTK data directory.

6. Launch JupyterLab:
```bash
jupyter lab
```

This will open JupyterLab in your default web browser. You can then navigate to the notebooks in the repository and run them.

### Additional Notes
- Make sure to keep your virtual environment activated while working on the project
- If you encounter any issues with dependencies, try updating pip: `pip install --upgrade pip`
- If you encounter issues with FAISS installation:
  - Make sure you have the latest pip: `pip install --upgrade pip`
  - For GPU version, ensure CUDA is properly installed and accessible
  - Try installing the CPU version first to verify basic functionality
  - Check your Python version compatibility (FAISS works best with Python 3.8-3.10)




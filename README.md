# Topic Modeling Pipelines

A Python library for building and experimenting with different topic modeling approaches, with a focus on statistical validation and network-based methods.

## Overview

This repository implements multiple topic modeling techniques designed for extracting meaningful topics from document collections. The first implementation is **Word Co-occurrence SVN Topic Modeling (WCSVNtm)**, with BERTopic planned as the next addition.

## Current Implementation: WCSVNtm

### What is WCSVNtm?

Word Co-occurrence SVN Topic Modeling is a network-based approach that:

1. **Builds bipartite networks** between sentences and words
2. **Statistically validates word co-occurrences** using hypergeometric tests with Bonferroni correction
3. **Projects validated relationships** into word and document networks
4. **Detects communities** using Leiden/Louvain algorithms for topic discovery
5. **Computes topic importance** based on modularity contributions

### Key Features

- **Statistical rigor**: Uses hypergeometric testing with multiple testing correction
- **Scalable computation**: Vectorized operations with Polars for large datasets
- **Fallback compatibility**: Pure Python implementations when dependencies unavailable
- **Comprehensive logging**: Detailed progress tracking and debugging information
- **Modular design**: Clean separation of pipeline stages

## Installation

### Prerequisites

- Python 3.13+
- UV package manager (recommended) or pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd topic_modeling

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Development Setup

```bash
# Install development dependencies
uv sync --group dev

# Set up pre-commit hooks
uv run pre-commit install
```

## Usage

### Basic Usage

```python
from WCSVNtm.wcsvntm_draft import run_wcsvntm
from WCSVNtm.data_io import DataReader

# Load documents
reader = DataReader("data/")
documents = reader.read_all_txt_files()

# Run topic modeling
results = run_wcsvntm(
    documents,
    alpha_word=0.01,  # Statistical threshold for word pairs
    alpha_doc=0.01    # Statistical threshold for document pairs
)

# Access results
topics = results["topics"]              # Word communities
doc_clusters = results["doc_clusters"]  # Document communities
importance = results["importance"]      # Topic importance scores
associations = results["associations"]  # Topic-document associations
```

### Running the Demo

```bash
# Run with sample data
python -m WCSVNtm.wcsvntm_draft

# Or using the main entry point
python main.py
```

### Data Format

Place your text files in the `data/` directory. The system supports:

- **Multi-document files**: Documents separated by double newlines (`\n\n`)
- **Single-line documents**: Each line treated as a separate document
- **Mixed formats**: Automatically detected based on content structure

## Algorithm Pipeline

### 1. Document Preprocessing
- Sentence tokenization using NLTK
- Word tokenization and normalization
- Stopword removal (configurable)
- Statistical logging of corpus properties

### 2. Bipartite Graph Construction
- Sentence-word relationships
- Unique word tracking per sentence
- Efficient graph representation using NetworkX

### 3. Statistical Validation
- **Word pairs**: Hypergeometric test for co-occurrence significance
- **Document pairs**: Statistical validation of shared topic patterns
- **Multiple testing correction**: Bonferroni adjustment
- **Vectorized computation**: Polars-based optimization

### 4. Network Projection
- Word co-occurrence network from validated pairs
- Document similarity network
- Edge weights based on statistical strength

### 5. Community Detection
- Primary: Leiden algorithm (if available)
- Fallback: Louvain algorithm
- Configurable resolution parameters

### 6. Topic Analysis
- **Importance scoring**: Modularity-based word importance within topics
- **Document association**: Fisher's exact test with FDR correction
- **Topic characterization**: Statistical significance of topic-document relationships

## Configuration

### Key Parameters

- `alpha_word`: Statistical threshold for word pair validation (default: 0.01)
- `alpha_doc`: Statistical threshold for document pair validation (default: 0.01)
- `fdr_alpha`: False discovery rate for topic-document associations (default: 0.05)

### Logging Configuration

```python
from WCSVNtm.logging_config import setup_logging

# Configure logging level and output
setup_logging(level="INFO", log_file="wcsvntm.log")
```

## Dependencies

### Core Dependencies
- `networkx`: Graph operations and community detection
- `nltk`: Natural language processing
- `polars`: High-performance data manipulation
- `scipy`: Statistical computations
- `pydantic`: Data validation

### Development Dependencies
- `pytest`: Testing framework
- `ruff`: Code linting and formatting
- `pyright`: Type checking
- `pre-commit`: Git hooks for code quality

## Roadmap

### Upcoming Features
- **BERTopic integration**: Transformer-based topic modeling
- **Comparative analysis tools**: Side-by-side method evaluation
- **Visualization components**: Interactive topic exploration
- **Performance benchmarks**: Scalability testing and optimization
- **Additional algorithms**: LDA, NMF, and other classical methods

### Current Status
-  WCSVNtm implementation complete
-  Statistical validation pipeline
-  Vectorized computation optimization
- =§ BERTopic integration (planned)
- =§ Visualization tools (planned)

## Contributing

### Development Workflow

1. **Set up environment**: `uv sync --group dev`
2. **Install hooks**: `uv run pre-commit install`
3. **Run tests**: `uv run pytest`
4. **Check types**: `uv run pyright`
5. **Format code**: `uv run ruff format`

### Code Quality

This project uses:
- **Ruff** for linting and formatting
- **Pyright** for type checking
- **Pre-commit hooks** for automated quality checks
- **Pytest** for testing

## License

[Add your license here]

## Citation

If you use WCSVNtm in your research, please cite:

```bibtex
[Add citation information here]
```

## Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Documentation**: [Full Documentation](link-to-docs)

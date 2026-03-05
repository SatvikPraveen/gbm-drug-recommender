# Contributing to GBM Drug Analysis and Recommendation

Thank you for your interest in contributing to this project! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to foster an inclusive and collaborative environment.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git for version control

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/GBM_drug_analysis_and_recommendation.git
   cd GBM_drug_analysis_and_recommendation
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Setup Script** (if needed)
   ```bash
   bash setup.sh
   ```

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Include a clear description of the problem
- Provide steps to reproduce the issue
- Include error messages and stack traces if applicable
- Specify your Python version and OS

### Suggesting Enhancements

- Open an issue with the tag "enhancement"
- Clearly describe the proposed feature
- Explain why this enhancement would be useful
- Provide examples if possible

### Pull Requests

1. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards

3. Test your changes thoroughly

4. Commit your changes with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request with a clear title and description

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions, classes, and modules
- Keep functions focused and concise (ideally < 50 lines)
- Use type hints where appropriate

### Example Function Documentation

```python
def predict_drug_efficacy(drug_name: str, cell_line: str) -> float:
    """
    Predict drug efficacy for a given cell line.
    
    Args:
        drug_name: Name of the drug to test
        cell_line: Cell line identifier
        
    Returns:
        Predicted IC50 value
        
    Raises:
        ValueError: If drug or cell line not found in dataset
    """
    pass
```

### Code Organization

- Place data processing functions in `src/data_processing.py`
- Add new models to `src/models/`
- Utility functions go in `src/utils/`
- Keep configuration in `src/config.py`

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests

- Add tests for all new features
- Place test files in a `tests/` directory
- Use descriptive test function names: `test_feature_description()`
- Aim for high code coverage (>80%)

## Submitting Changes

### Checklist Before Submitting

- [ ] Code follows the project's style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated if needed
- [ ] No unnecessary files or debugging code included
- [ ] Commit messages are clear and descriptive

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be concise (50 chars or less)
- Reference issues and pull requests when relevant

Examples:
```
Add drug combination analysis module
Fix issue with missing SMILES data handling
Update pathway enrichment analysis algorithm
Refactor model comparison code for better performance
```

## Project Structure

```
GBM_drug_analysis_and_recommendation/
├── src/                      # Source code
│   ├── models/              # Machine learning models
│   ├── similarity/          # Drug similarity algorithms
│   └── utils/               # Utility functions
├── data/                    # Data files (raw and processed)
├── notebooks/               # Jupyter notebooks for exploration
├── results/                 # Analysis results and outputs
├── main.py                  # Main execution script
├── dashboard.py             # Visualization dashboard
└── requirements.txt         # Python dependencies
```

### Key Modules

- **data_processing.py**: Data cleaning and preprocessing
- **feature_extraction.py**: Molecular feature extraction
- **pathway_analysis.py**: Pathway enrichment analysis
- **drug_interactions.py**: Drug interaction prediction
- **combination_therapy.py**: Combination therapy recommendations

## Data Guidelines

- Do not commit large data files to the repository
- Place data files in `data/raw/` or `data/processed/`
- Update `data/README.md` with information about new datasets
- Use relative paths in code for portability

## Questions?

If you have questions or need help, please:
- Open an issue on GitHub
- Review existing issues and pull requests
- Check the README.md for project documentation

Thank you for contributing to GBM drug analysis research! 🎉

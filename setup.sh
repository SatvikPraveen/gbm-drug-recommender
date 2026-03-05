#!/bin/bash
# Setup script for GBM Drug Analysis Project (macOS/Metal Support)

echo "🚀 Setting up GBM Drug Analysis and Recommendation System..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core requirements
echo "📚 Installing core dependencies..."
pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly tqdm requests pyyaml pyreadr

# Install RDKit (chemistry library)
echo "🧪 Installing RDKit..."
pip install rdkit

# Install PubChemPy
echo "🔬 Installing PubChemPy..."
pip install pubchempy

# Install NetworkX
echo "🕸️  Installing NetworkX..."
pip install networkx

# Install PyTorch for Mac (MPS/Metal support)
echo "🔥 Installing PyTorch with Metal Performance Shaders (MPS) support..."
pip install torch torchvision torchaudio

# Install PyTorch Geometric and dependencies
echo "📊 Installing PyTorch Geometric..."
pip install torch-geometric

# Install PyG dependencies (for macOS, use CPU wheels)
echo "🔧 Installing PyTorch Geometric dependencies..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install UMAP for dimensionality reduction
echo "🗺️  Installing UMAP..."
pip install umap-learn

# Install Jupyter
echo "📓 Installing Jupyter..."
pip install jupyter ipykernel ipywidgets

# Create IPython kernel
echo "🔧 Creating IPython kernel..."
python -m ipykernel install --user --name=gbm_env --display-name="Python (GBM Analysis)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "    source venv/bin/activate"
echo ""
echo "To start Jupyter Notebook, run:"
echo "    jupyter notebook"
echo ""
echo "To run the main pipeline, run:"
echo "    python main.py"
echo ""

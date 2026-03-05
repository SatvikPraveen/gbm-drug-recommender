"""
Interactive Streamlit Dashboard for GBM Drug Analysis

Provides a comprehensive web interface for exploring the complete pipeline results:

Features:
- Overview - Summary statistics and key metrics
- Drug Predictions - Rankings with decision scores and promising candidates
- Drug Similarity - Interactive heatmaps for Tanimoto, MCS, GCN similarity
- Combination Therapy - Top drug pairs with synergy scores and rationales
- Pathway Analysis - Enriched pathways across KEGG, Reactome, BioPlanet, GO
- Model Comparison - Performance metrics for 5 ML models
- Drug Interactions - Safety profiles and severity classifications

Launch:
    streamlit run dashboard.py
    # Opens at http://localhost:8501

Requirements:
- Pipeline must be executed first (main.py) to generate results
- All visualizations load from results/ directory
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR


# Page configuration
st.set_page_config(
    page_title="GBM Drug Analysis Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_predictions():
    """Load drug predictions."""
    pred_file = RESULTS_DIR / 'models' / 'drug_predictions.csv'
    if pred_file.exists():
        return pd.read_csv(pred_file)
    return None


@st.cache_data
def load_similarity_data():
    """Load similarity matrices."""
    sim_dir = RESULTS_DIR / 'similarity'
    
    similarities = {}
    for method in ['tanimoto', 'mcs', 'gcn']:
        sim_file = sim_dir / f'{method}_similarity_matrix.csv'
        if sim_file.exists():
            similarities[method] = pd.read_csv(sim_file, index_col=0)
    
    return similarities


@st.cache_data
def load_combination_therapy():
    """Load combination therapy analysis."""
    combo_file = RESULTS_DIR / 'combination_therapy' / 'drug_combinations.csv'
    if combo_file.exists():
        return pd.read_csv(combo_file)
    return None


@st.cache_data
def load_pathway_data():
    """Load pathway enrichment results."""
    pathway_file = RESULTS_DIR / 'pathways' / 'pathway_enrichment_summary.csv'
    if pathway_file.exists():
        return pd.read_csv(pathway_file)
    return None


@st.cache_data
def load_model_comparison():
    """Load model comparison results."""
    model_file = RESULTS_DIR / 'models' / 'model_comparison_results.csv'
    if model_file.exists():
        return pd.read_csv(model_file)
    return None


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🧬 GBM Drug Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Precision Oncology Through Machine Learning")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        ["Overview", "Drug Predictions", "Drug Similarity", "Combination Therapy", 
         "Pathway Analysis", "Model Comparison"]
    )
    
    # Load data
    predictions = load_predictions()
    similarities = load_similarity_data()
    combinations = load_combination_therapy()
    pathways = load_pathway_data()
    models = load_model_comparison()
    
    # Page routing
    if page == "Overview":
        show_overview(predictions, similarities, combinations, pathways, models)
    
    elif page == "Drug Predictions":
        show_predictions(predictions)
    
    elif page == "Drug Similarity":
        show_similarity(similarities)
    
    elif page == "Combination Therapy":
        show_combinations(combinations)
    
    elif page == "Pathway Analysis":
        show_pathways(pathways)
    
    elif page == "Model Comparison":
        show_models(models)


def show_overview(predictions, similarities, combinations, pathways, models):
    """Display overview page."""
    st.markdown('<h2 class="sub-header">📊 Analysis Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if predictions is not None:
            n_drugs = len(predictions)
            st.metric("Total Drugs Analyzed", n_drugs)
        else:
            st.metric("Total Drugs Analyzed", "N/A")
    
    with col2:
        if predictions is not None:
            promising = len(predictions[predictions['prediction'] == 1])
            st.metric("Promising Candidates", promising)
        else:
            st.metric("Promising Candidates", "N/A")
    
    with col3:
        if combinations is not None:
            st.metric("Top Combinations", len(combinations))
        else:
            st.metric("Top Combinations", "N/A")
    
    with col4:
        st.metric("Similarity Methods", len(similarities) if similarities else 0)
    
    # Top candidates
    if predictions is not None:
        st.markdown("### 🎯 Top Drug Candidates")
        
        # Filter promising drugs
        promising_drugs = predictions[predictions['prediction'] == 1]
        
        if len(promising_drugs) > 0:
            # Sort by decision score
            promising_drugs = promising_drugs.sort_values('decision_score', ascending=False)
            
            # Display top 10
            top_10 = promising_drugs.head(10)
            
            fig = px.bar(
                top_10,
                x='drug_name',
                y='decision_score',
                title='Top 10 Promising Drug Candidates',
                color='decision_score',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(
                top_10[['drug_name', 'decision_score', 'is_promising']],
                use_container_width=True
            )
        else:
            st.warning("No promising drug candidates found in predictions.")
    
    # Analysis summary
    st.markdown("### 📈 Analysis Pipeline Status")
    
    status_data = []
    status_data.append({"Stage": "Drug Predictions", "Status": "✅ Complete" if predictions is not None else "❌ Not Run"})
    status_data.append({"Stage": "Similarity Analysis", "Status": f"✅ Complete ({len(similarities)} methods)" if similarities else "❌ Not Run"})
    status_data.append({"Stage": "Combination Therapy", "Status": "✅ Complete" if combinations is not None else "❌ Not Run"})
    status_data.append({"Stage": "Pathway Enrichment", "Status": "✅ Complete" if pathways is not None else "❌ Not Run"})
    status_data.append({"Stage": "Model Comparison", "Status": "✅ Complete" if models is not None else "❌ Not Run"})
    
    st.table(pd.DataFrame(status_data))


def show_predictions(predictions):
    """Display drug predictions page."""
    st.markdown('<h2 class="sub-header">💊 Drug Predictions</h2>', unsafe_allow_html=True)
    
    if predictions is None:
        st.error("No prediction data available. Please run the pipeline first.")
        return
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        class_filter = st.selectbox(
            "Filter by Prediction",
            ["All", "Promising (Class 1)", "Not Promising (Class 0)"]
        )
    
    with col2:
        min_score = st.slider(
            "Minimum Decision Score",
            -1.0, 1.0, -1.0, 0.05
        )
    
    # Apply filters
    filtered_df = predictions.copy()
    
    if class_filter == "Promising (Class 1)":
        filtered_df = filtered_df[filtered_df['prediction'] == 1]
    elif class_filter == "Not Promising (Class 0)":
        filtered_df = filtered_df[filtered_df['prediction'] == -1]
    
    filtered_df = filtered_df[filtered_df['decision_score'] >= min_score]
    
    # Display results
    st.markdown(f"### Showing {len(filtered_df)} drugs")
    
    # Scatter plot
    fig = px.scatter(
        filtered_df,
        x='drug_name',
        y='decision_score',
        color='prediction',
        hover_data=['is_promising'],
        title='Drug Predictions by Decision Score',
        labels={'prediction': 'Prediction Class', 'decision_score': 'Decision Score'},
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Results",
        data=csv,
        file_name="filtered_predictions.csv",
        mime="text/csv"
    )


def show_similarity(similarities):
    """Display similarity analysis page."""
    st.markdown('<h2 class="sub-header">🔬 Drug Similarity Analysis</h2>', unsafe_allow_html=True)
    
    if not similarities:
        st.error("No similarity data available. Please run the pipeline first.")
        return
    
    # Method selection
    method = st.selectbox(
        "Select Similarity Method",
        list(similarities.keys())
    )
    
    sim_matrix = similarities[method]
    method_name = str(method).upper() if method else "UNKNOWN"
    
    # Heatmap
    st.markdown(f"### {method_name} Similarity Matrix")
    
    fig = px.imshow(
        sim_matrix,
        labels=dict(color="Similarity"),
        title=f"{method_name} Similarity Heatmap",
        color_continuous_scale='Blues',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Drug pair selection
    st.markdown("### Compare Specific Drugs")
    
    col1, col2 = st.columns(2)
    
    drugs = sim_matrix.index.tolist()
    
    with col1:
        drug1 = st.selectbox("Select Drug 1", drugs, key='drug1')
    
    with col2:
        drug2 = st.selectbox("Select Drug 2", drugs, key='drug2')
    
    if drug1 and drug2:
        similarity_score = sim_matrix.loc[drug1, drug2]
        st.metric(f"Similarity: {drug1} vs {drug2}", f"{similarity_score:.4f}")


def show_combinations(combinations):
    """Display combination therapy page."""
    st.markdown('<h2 class="sub-header">💊+💊 Combination Therapy</h2>', unsafe_allow_html=True)
    
    if combinations is None:
        st.error("No combination data available. Please run the pipeline first.")
        return
    
    # Top combinations
    st.markdown("### Top Drug Combinations")
    
    # Bar chart
    top_combos = combinations.head(15)
    
    fig = px.bar(
        top_combos,
        x='Combination',
        y='Total_Score',
        title='Top 15 Drug Combinations by Synergy Score',
        color='Total_Score',
        color_continuous_scale='Greens',
        hover_data=['Pathway_Score', 'Target_Score', 'Similarity_Score']
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed view
    st.markdown("### Combination Details")
    
    selected_combo = st.selectbox(
        "Select a combination to view details",
        combinations['Combination'].tolist()
    )
    
    combo_details = combinations[combinations['Combination'] == selected_combo].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Score", f"{combo_details['Total_Score']:.3f}")
    
    with col2:
        st.metric("Pathway Score", f"{combo_details['Pathway_Score']:.3f}")
    
    with col3:
        st.metric("Target Score", f"{combo_details['Target_Score']:.3f}")
    
    st.info(f"**Rationale:** {combo_details['Rationale']}")
    
    # Full table
    st.markdown("### All Combinations")
    st.dataframe(combinations, use_container_width=True)


def show_pathways(pathways):
    """Display pathway analysis page."""
    st.markdown('<h2 class="sub-header">🧬 Pathway Enrichment Analysis</h2>', unsafe_allow_html=True)
    
    if pathways is None:
        st.error("No pathway data available. Please run the pipeline first.")
        return
    
    st.info("Pathway enrichment analysis based on all drug targets combined from the dataset.")
    
    # Library filter
    if 'Library' in pathways.columns:
        libraries = ['All'] + sorted(pathways['Library'].unique().tolist())
        selected_library = st.selectbox("Filter by Pathway Database", libraries)
        
        if selected_library != 'All':
            filtered_pathways = pathways[pathways['Library'] == selected_library]
        else:
            filtered_pathways = pathways
    else:
        filtered_pathways = pathways
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pathways", len(filtered_pathways))
    with col2:
        sig_pathways = len(filtered_pathways[filtered_pathways['P-value'] < 0.05]) if 'P-value' in filtered_pathways.columns else 0
        st.metric("Significant Pathways (p<0.05)", sig_pathways)
    with col3:
        if 'Library' in filtered_pathways.columns:
            st.metric("Databases", filtered_pathways['Library'].nunique())
    
    # Top pathways visualization
    st.markdown("### Top 20 Enriched Pathways")
    
    top_pathways = filtered_pathways.head(20).copy()
    
    if 'P-value' in top_pathways.columns:
        # Convert p-value to -log10(p)
        top_pathways['-log10(P)'] = -np.log10(top_pathways['P-value'] + 1e-300)
        
        fig = px.bar(
            top_pathways,
            y='Term',
            x='-log10(P)',
            title='Top 20 Most Significant Pathways',
            orientation='h',
            color='-log10(P)',
            color_continuous_scale='Reds',
            hover_data=['Library', 'Overlapping Genes'] if 'Library' in top_pathways.columns else None
        )
        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title='-log10(P-value)',
            yaxis_title='Pathway'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Pathway by library breakdown
    if 'Library' in filtered_pathways.columns:
        st.markdown("### Pathways by Database")
        library_counts = filtered_pathways['Library'].value_counts().reset_index()
        library_counts.columns = ['Database', 'Count']
        
        fig2 = px.pie(
            library_counts,
            values='Count',
            names='Database',
            title='Distribution of Enriched Pathways by Database'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Full data table
    st.markdown("### Pathway Enrichment Results")
    st.dataframe(
        filtered_pathways[[c for c in ['Rank', 'Term', 'P-value', 'Adjusted P-value', 
                                       'Combined Score', 'Overlapping Genes', 'Library'] 
                          if c in filtered_pathways.columns]],
        use_container_width=True,
        height=400
    )


def show_models(models):
    """Display model comparison page."""
    st.markdown('<h2 class="sub-header">🤖 Model Comparison</h2>', unsafe_allow_html=True)
    
    if models is None:
        st.error("No model comparison data available.")
        return
    
    # Performance comparison
    st.markdown("### Model Performance Comparison")
    
    # Bar chart for main metric
    if 'F1_Score' in models.columns:
        metric = 'F1_Score'
    elif 'R2_Score' in models.columns:
        metric = 'R2_Score'
    else:
        metric = models.columns[2]  # Use third column as default
    
    fig = px.bar(
        models,
        x='Model',
        y=metric,
        title=f'Model Comparison by {metric}',
        color=metric,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("### Detailed Metrics")
    st.dataframe(models, use_container_width=True)
    
    # Cross-validation scores if available
    if 'CV_Score_Mean' in models.columns:
        st.markdown("### Cross-Validation Performance")
        
        fig = go.Figure()
        
        for idx, row in models.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Model']],
                y=[row['CV_Score_Mean']],
                error_y=dict(type='data', array=[row['CV_Score_Std']]),
                name=row['Model']
            ))
        
        fig.update_layout(
            title='Cross-Validation Scores (Mean ± Std)',
            yaxis_title='CV Score',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

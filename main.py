#!/usr/bin/env python3
"""
Main Pipeline for GBM Drug Analysis and Recommendation System

A comprehensive 10-stage precision oncology pipeline that identifies promising
drug candidates and combinations for Glioblastoma Multiforme (GBM) treatment.

Pipeline Stages:
1. Data Loading & Preprocessing - Process GDSC drug screening data
2. Feature Extraction - Extract molecular descriptors from SMILES
3. Similarity Analysis - Compute Tanimoto, MCS, and GCN-based similarity
4. Clustering Analysis - Group drugs using KMeans, DBSCAN, Hierarchical
5. Drug Prediction - One-Class SVM to identify promising candidates
6. Pathway Enrichment - Analyze biological pathways via Enrichr API
7. Visualization - Generate heatmaps, distributions, and cluster plots
8. Combination Therapy - Score drug pairs for synergistic potential
9. Model Comparison - Benchmark 5 ML models (KNN, RF, SVM, NN, XGBoost)
10. Drug-Drug Interactions - Safety analysis for recommended combinations

Outputs:
- Molecular features, similarity matrices, clustering assignments
- Promising drug predictions with decision scores
- Pathway enrichment data across KEGG, Reactome, BioPlanet, GO
- Top drug combinations with synergy scores and rationales
- Model performance comparison and trained model files
- Drug interaction safety profiles
- Comprehensive analysis report and interactive dashboard

Usage:
    python main.py [--skip-data-processing] [--skip-similarity] [...]
    streamlit run dashboard.py  # Launch interactive dashboard

Author: GBM Drug Analysis Team
Date: 2026
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Import modules
from src import config
from src.config import *
from src.data_processing import GDSCDataLoader, load_processed_data
from src.feature_extraction import MolecularFeatureExtractor, SMILESManager
from src.similarity import TanimotoSimilarityAnalyzer, MCSimilarityAnalyzer, GCNSimilarityAnalyzer
from src.models import DrugClusteringAnalyzer, OneClassDrugPredictor
from src.pathway_analysis import PathwayAnalyzer, DrugTargetMapper
from src.utils import VisualizationTools
from src.combination_therapy import CombinationTherapyAnalyzer
from src.models.model_comparison import ModelComparison
from src.drug_interactions import DrugInteractionChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='GBM Drug Analysis and Recommendation Pipeline'
    )
    
    parser.add_argument(
        '--skip-data-processing',
        action='store_true',
        help='Skip data processing step (use existing processed data)'
    )
    
    parser.add_argument(
        '--skip-similarity',
        action='store_true',
        help='Skip similarity analysis'
    )
    
    parser.add_argument(
        '--skip-clustering',
        action='store_true',
        help='Skip clustering analysis'
    )
    
    parser.add_argument(
        '--skip-prediction',
        action='store_true',
        help='Skip drug prediction with One-Class SVM'
    )
    
    parser.add_argument(
        '--skip-pathway',
        action='store_true',
        help='Skip pathway enrichment analysis'
    )
    
    parser.add_argument(
        '--skip-combination',
        action='store_true',
        help='Skip combination therapy analysis'
    )
    
    parser.add_argument(
        '--skip-model-comparison',
        action='store_true',
        help='Skip model comparison analysis'
    )
    
    parser.add_argument(
        '--skip-interactions',
        action='store_true',
        help='Skip drug-drug interaction checking'
    )
    
    parser.add_argument(
        '--top-n-drugs',
        type=int,
        default=TOP_N_DRUGS,
        help='Number of top drugs to report'
    )
    
    return parser.parse_args()


def step_1_data_processing(skip=False):
    """Step 1: Load and process GDSC data"""
    logger.info("=" * 80)
    logger.info("STEP 1: DATA PROCESSING")
    logger.info("=" * 80)
    
    if skip and CLEANED_DATA_FILE.exists():
        logger.info("Skipping data processing - loading existing data")
        data = load_processed_data()
    else:
        loader = GDSCDataLoader()
        data = loader.process_pipeline(filter_gbm=True, save_output=True)
    
    logger.info(f"Loaded data shape: {data.shape}")
    return data


def step_2_feature_extraction(data):
    """Step 2: Extract molecular features from SMILES"""
    logger.info("=" * 80)
    logger.info("STEP 2: FEATURE EXTRACTION")
    logger.info("=" * 80)
    
    # Get unique drug names
    if 'drug_name' in data.columns:
        drug_names = data['drug_name'].unique().tolist()
    else:
        logger.error("drug_name column not found")
        return None
    
    logger.info(f"Extracting features for {len(drug_names)} drugs")
    
    # Get or update SMILES
    smiles_manager = SMILESManager()
    smiles_dict = smiles_manager.update_smiles_from_list(drug_names)
    
    # Extract features
    extractor = MolecularFeatureExtractor()
    
    if FEATURES_FILE.exists():
        logger.info("Loading existing molecular features")
        features_df = extractor.load_features()
    else:
        features_df = extractor.process_drug_list(drug_names, smiles_dict)
        features_df = extractor.add_lipinski_features(features_df)
        extractor.save_features(features_df)
    
    logger.info(f"Features extracted: {features_df.shape}")
    return features_df, smiles_dict


def step_3_similarity_analysis(smiles_dict, skip=False):
    """Step 3: Calculate drug similarity using multiple methods"""
    logger.info("=" * 80)
    logger.info("STEP 3: SIMILARITY ANALYSIS")
    logger.info("=" * 80)
    
    if skip:
        logger.info("Skipping similarity analysis")
        return {}
    
    results = {}
    
    # Tanimoto similarity
    logger.info("Computing Tanimoto similarity...")
    tanimoto = TanimotoSimilarityAnalyzer()
    tanimoto_matrix = tanimoto.build_similarity_matrix(smiles_dict)
    tanimoto.save_similarity_matrix(tanimoto_matrix)
    results['tanimoto'] = tanimoto_matrix
    
    # MCS similarity
    logger.info("Computing MCS similarity...")
    mcs = MCSimilarityAnalyzer()
    mcs_matrix = mcs.build_similarity_matrix(smiles_dict)
    mcs.save_similarity_matrix(mcs_matrix)
    results['mcs'] = mcs_matrix
    
    # GCN similarity
    logger.info("Computing GCN-based similarity...")
    gcn = GCNSimilarityAnalyzer()
    gcn_matrix = gcn.build_similarity_matrix(smiles_dict)
    gcn.save_similarity_matrix(gcn_matrix)
    results['gcn'] = gcn_matrix
    
    logger.info("Similarity analysis complete")
    return results


def step_4_clustering_analysis(features_df, skip=False):
    """Step 4: Cluster drugs based on molecular features"""
    logger.info("=" * 80)
    logger.info("STEP 4: CLUSTERING ANALYSIS")
    logger.info("=" * 80)
    
    if skip:
        logger.info("Skipping clustering analysis")
        return None
    
    # Select numeric features for clustering
    feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col in MOLECULAR_DESCRIPTORS]
    
    logger.info(f"Using {len(feature_cols)} features for clustering")
    
    # Perform clustering
    clustering = DrugClusteringAnalyzer()
    results = clustering.analyze_all_methods(features_df, feature_cols)
    
    # Save results
    clustering.save_clustering_results(features_df, results)
    
    logger.info("Clustering analysis complete")
    return results


def step_5_drug_prediction(data, features_df, skip=False):
    """Step 5: Predict promising drugs using One-Class SVM"""
    logger.info("=" * 80)
    logger.info("STEP 5: DRUG PREDICTION")
    logger.info("=" * 80)
    
    if skip:
        logger.info("Skipping drug prediction")
        return None
    
    # Get effective drugs based on IC50 threshold (more relaxed criteria)
    # Use is_effective_ic50 instead of is_effective to get more training samples
    effective_col = 'is_effective_ic50' if 'is_effective_ic50' in data.columns else 'is_effective'
    if effective_col not in data.columns:
        logger.warning(f"{effective_col} column not found - cannot train predictor")
        return None
    
    effective_drugs = data[data[effective_col]]['drug_name'].unique()
    logger.info(f"Training on {len(effective_drugs)} effective drugs (using {effective_col})")
    
    # Handle case where no effective drugs are found
    if len(effective_drugs) == 0:
        logger.warning("No effective drugs found. Skipping prediction step.")
        logger.info("Consider adjusting IC50_THRESHOLD in config.py or using real GDSC data.")
        return None
    
    # Get features for effective drugs
    effective_features = features_df[features_df['drug_name'].isin(effective_drugs)]
    feature_cols = [col for col in MOLECULAR_DESCRIPTORS if col in effective_features.columns]
    
    X_train = effective_features[feature_cols].values
    X_train = np.nan_to_num(X_train, nan=0.0)
    
    # Train One-Class SVM
    predictor = OneClassDrugPredictor()
    predictor.fit(X_train, feature_names=feature_cols)
    
    # Predict on all drugs
    predictions = predictor.identify_promising_drugs(
        features_df, feature_cols, drug_name_col='drug_name'
    )
    
    # Save predictions
    predictions_file = MODEL_RESULTS_DIR / "drug_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to {predictions_file}")
    
    # Save model
    predictor.save_model()
    
    logger.info("Drug prediction complete")
    return predictions


def step_6_pathway_analysis(predictions, top_n=20, skip=False):
    """Step 6: Pathway enrichment analysis"""
    logger.info("=" * 80)
    logger.info("STEP 6: PATHWAY ENRICHMENT ANALYSIS")
    logger.info("=" * 80)
    
    if skip:
        logger.info("Skipping pathway analysis")
        return None
    
    # Get top predicted drugs
    if predictions is not None:
        top_drugs = predictions.nlargest(top_n, 'decision_score')['drug_name'].tolist()
    else:
        logger.warning("No predictions available - skipping pathway analysis")
        return None
    
    logger.info(f"Analyzing pathways for top {len(top_drugs)} drugs")
    
    # Get drug-target mapping
    mapper = DrugTargetMapper()
    drug_targets = mapper.get_all_targets(top_drugs)
    
    logger.info(f"Found targets for {len(drug_targets)} drugs")
    
    if len(drug_targets) == 0:
        logger.warning("No drug-target mappings found")
        return None
    
    # Perform pathway enrichment
    analyzer = PathwayAnalyzer()
    enrichment_results = analyzer.analyze_drug_targets(drug_targets)
    
    # Save results
    analyzer.save_pathway_results(enrichment_results)
    
    logger.info("Pathway analysis complete")
    return enrichment_results


def step_7_visualization(similarity_results, clustering_results, predictions):
    """Step 7: Generate visualizations"""
    logger.info("=" * 80)
    logger.info("STEP 7: VISUALIZATION")
    logger.info("=" * 80)
    
    viz = VisualizationTools()
    
    # Similarity visualizations
    if similarity_results:
        for method_name, sim_matrix in similarity_results.items():
            viz.plot_similarity_heatmap(
                sim_matrix,
                title=f"{method_name.upper()} Similarity Matrix",
                save_name=f"{method_name}_similarity_heatmap.png"
            )
            
            viz.plot_similarity_distribution(
                sim_matrix,
                title=f"{method_name.upper()} Similarity Distribution",
                save_name=f"{method_name}_similarity_dist.png"
            )
    
    # Clustering visualizations
    if clustering_results:
        for method in ['kmeans', 'dbscan', 'hierarchical']:
            if method in clustering_results:
                viz.plot_clustering_2d(
                    clustering_results['umap_2d'],
                    clustering_results[method]['labels'],
                    title=f"{method.upper()} Clustering (UMAP)",
                    method="UMAP",
                    save_name=f"{method}_clustering_umap.png"
                )
    
    # Prediction visualizations
    if predictions is not None:
        viz.plot_top_drugs(
            predictions,
            score_col='decision_score',
            title="Top Predicted Drugs (One-Class SVM)",
            save_name="top_predicted_drugs.png"
        )
    
    logger.info("Visualization complete")


def step_8_combination_therapy(predictions, similarity_results, pathway_results, 
                               top_n=20, skip=False):
    """Step 8: Combination therapy analysis"""
    logger.info("=" * 80)
    logger.info("STEP 8: COMBINATION THERAPY ANALYSIS")
    logger.info("=" * 80)
    
    if skip:
        logger.info("Skipping combination therapy analysis")
        return None
    
    if predictions is None or len(predictions) == 0:
        logger.warning("No predictions available - skipping combination therapy")
        return None
    
    # Get top drugs
    promising_drugs = predictions[predictions['is_promising']].head(top_n)['drug_name'].tolist()
    
    if len(promising_drugs) < 2:
        logger.warning("Need at least 2 drugs for combination analysis")
        return None
    
    logger.info(f"Analyzing combinations of {len(promising_drugs)} top drugs")
    
    # Prepare data for analyzer
    # Note: The pathway analyzer processes all drugs together, not per-drug
    # So we don't have per-drug pathway data for combination analysis
    # The analyzer will use similarity and target data instead
    pathway_data = None
    
    # Initialize analyzer
    analyzer = CombinationTherapyAnalyzer(
        similarity_matrices=similarity_results,
        pathway_data=pathway_data
    )
    
    # Analyze combinations
    combinations = analyzer.analyze_all_combinations(promising_drugs, top_n=20)
    
    # Save results
    combo_dir = RESULTS_DIR / 'combination_therapy'
    combo_file = combo_dir / 'drug_combinations.csv'
    analyzer.export_results(combinations, combo_file, include_matrix=True)
    
    logger.info(f"Combination therapy analysis complete: {len(combinations)} combinations found")
    return combinations


def step_9_model_comparison(data, features_df, skip=False):
    """Step 9: Compare multiple ML models"""
    logger.info("=" * 80)
    logger.info("STEP 9: MODEL COMPARISON")
    logger.info("=" * 80)
    
    if skip:
        logger.info("Skipping model comparison")
        return None
    
    # Prepare data for classification task
    effective_col = 'is_effective_ic50' if 'is_effective_ic50' in data.columns else 'is_effective'
    if effective_col not in data.columns:
        logger.warning("No effectiveness labels - skipping model comparison")
        return None
    
    # Merge data with features
    data_with_features = data.merge(
        features_df,
        left_on='drug_name',
        right_on='drug_name',
        how='inner'
)    
    # Select features
    feature_cols = [col for col in MOLECULAR_DESCRIPTORS if col in data_with_features.columns]
    
    X = data_with_features[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = data_with_features[effective_col].values
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Initialize and train models
    model_comp = ModelComparison(task_type='classification', random_state=42)
    results_df = model_comp.train_and_evaluate(X_train, y_train, X_test, y_test, cv_folds=5)
    
    # Save results
    models_dir = RESULTS_DIR / 'models'
    model_comp.save_models(models_dir)
    
    logger.info(f"Model comparison complete. Best model: {model_comp.best_model_name}")
    return results_df


def step_10_drug_interactions(combinations, smiles_dict, skip=False):
    """Step 10: Check drug-drug interactions"""
    logger.info("=" * 80)
    logger.info("STEP 10: DRUG-DRUG INTERACTION CHECKING")
    logger.info("=" * 80)
    
    if skip:
        logger.info("Skipping interaction checking")
        return None
    
    if combinations is None or len(combinations) == 0:
        logger.warning("No combinations available - skipping interaction check")
        return None
    
    # Extract drug pairs
    drug_pairs = list(zip(combinations['Drug_A'], combinations['Drug_B']))
    
    logger.info(f"Checking interactions for {len(drug_pairs)} drug combinations")
    
    # Initialize checker
    checker = DrugInteractionChecker()
    
    # Check interactions
    interactions = checker.batch_check_interactions(drug_pairs, smiles_dict)
    
    # Save results
    interactions_dir = RESULTS_DIR / 'interactions'
    interactions_file = interactions_dir / 'drug_interactions.csv'
    checker.export_interactions(interactions, interactions_file)
    
    # Filter safe combinations
    safe_combinations = checker.filter_safe_combinations(interactions, max_severity='moderate')
    safe_file = interactions_dir / 'safe_combinations.csv'
    safe_combinations.to_csv(safe_file, index=False)
    logger.info(f"Safe combinations saved to {safe_file}")
    
    logger.info(f"Interaction checking complete: {len(interactions)} pairs analyzed")
    return interactions


def generate_final_report(data, features_df, similarity_results, 
                         clustering_results, predictions, pathway_results,
                         combinations=None, model_comparison=None, interactions=None):
    """Generate final analysis report"""
    logger.info("=" * 80)
    logger.info("GENERATING FINAL REPORT")
    logger.info("=" * 80)
    
    report_path = RESULTS_DIR / "analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GBM DRUG ANALYSIS AND RECOMMENDATION SYSTEM\n")
        f.write("Final Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Data summary
        f.write("1. DATA SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total drug-cell line combinations: {len(data)}\n")
        if 'drug_name' in data.columns:
            f.write(f"Unique drugs: {data['drug_name'].nunique()}\n")
        if 'cell_line' in data.columns:
            f.write(f"GBM cell lines: {data['cell_line'].nunique()}\n")
        f.write(f"\nMolecular features extracted: {features_df.shape[0]} drugs\n")
        f.write("\n")
        
        # Similarity summary
        f.write("2. SIMILARITY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        for method, matrix in similarity_results.items():
            upper_tri = np.triu(matrix.values, k=1)
            sims = upper_tri[upper_tri > 0]
            f.write(f"\n{method.upper()}:\n")
            if len(sims) > 0:
                f.write(f"  Mean similarity: {np.nanmean(sims):.3f}\n")
                f.write(f"  Max similarity: {np.nanmax(sims):.3f}\n")
                f.write(f"  Pairs > 0.7: {np.sum(sims > 0.7)}\n")
            else:
                f.write(f"  No valid similarities computed\n")
        f.write("\n")
        
        # Clustering summary
        if clustering_results:
            f.write("3. CLUSTERING ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for method in ['kmeans', 'dbscan', 'hierarchical']:
                if method in clustering_results:
                    metrics = clustering_results[method]['metrics']
                    f.write(f"\n{method.upper()}:\n")
                    f.write(f"  Clusters: {metrics['n_clusters']}\n")
                    f.write(f"  Silhouette Score: {metrics['silhouette_score']:.3f}\n")
            f.write("\n")
        
        # Prediction summary
        if predictions is not None:
            f.write("4. DRUG PREDICTIONS\n")
            f.write("-" * 80 + "\n")
            promising = predictions[predictions['is_promising']]
            f.write(f"Promising drugs identified: {len(promising)}\n")
            f.write(f"\nTop 20 Predicted Drugs:\n")
            for idx, row in predictions.head(20).iterrows():
                f.write(f"  {row['drug_name']}: {row['decision_score']:.3f}\n")
            f.write("\n")
        
        # Combination therapy summary
        if combinations is not None and len(combinations) > 0:
            f.write("5. COMBINATION THERAPY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Drug combinations analyzed: {len(combinations)}\n")
            f.write(f"\nTop 10 Combinations:\n")
            for idx, row in combinations.head(10).iterrows():
                f.write(f"  {row['Combination']}: {row['Total_Score']:.3f}\n")
                f.write(f"    Rationale: {row['Rationale']}\n")
            f.write("\n")
        
        # Model comparison summary
        if model_comparison is not None and len(model_comparison) > 0:
            f.write("6. MODEL COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write("Model Performance (Cross-Validation):\n")
            for idx, row in model_comparison.iterrows():
                f.write(f"\n{row['Model']}:\n")
                f.write(f"  CV Score: {row['CV_Score_Mean']:.4f} ± {row['CV_Score_Std']:.4f}\n")
                if 'F1_Score' in row:
                    f.write(f"  F1 Score: {row['F1_Score']:.4f}\n")
            f.write("\n")
        
        # Interaction checking summary
        if interactions is not None and len(interactions) > 0:
            f.write("7. DRUG-DRUG INTERACTIONS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Combinations checked: {len(interactions)}\n")
            with_interactions = len(interactions[interactions['Has_Interaction']])
            f.write(f"Interactions found: {with_interactions}\n")
            safe = len(interactions[~interactions['Has_Interaction']])
            f.write(f"Safe combinations: {safe}\n")
            f.write("\nSeverity breakdown:\n")
            for severity in ['high', 'moderate', 'low', 'none']:
                count = len(interactions[interactions['Severity'] == severity])
                f.write(f"  {severity.capitalize()}: {count}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Analysis complete. See results/ directory for detailed outputs.\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Report saved to {report_path}")
    print(f"\n📊 Final report saved to: {report_path}")


def main():
    """Main pipeline execution"""
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("GBM DRUG ANALYSIS AND RECOMMENDATION PIPELINE")
    logger.info("=" * 80)
    config.print_config_summary()
    
    try:
        # Step 1: Data Processing
        data = step_1_data_processing(skip=args.skip_data_processing)
        
        # Step 2: Feature Extraction
        extraction_result = step_2_feature_extraction(data)
        if extraction_result is None:
            logger.error("Feature extraction failed - cannot continue")
            return
        features_df, smiles_dict = extraction_result
        
        # Step 3: Similarity Analysis
        similarity_results = step_3_similarity_analysis(smiles_dict, skip=args.skip_similarity)
        
        # Step 4: Clustering
        clustering_results = step_4_clustering_analysis(features_df, skip=args.skip_clustering)
        
        # Step 5: Drug Prediction
        predictions = step_5_drug_prediction(data, features_df, skip=args.skip_prediction)
        
        # Step 6: Pathway Analysis
        pathway_results = step_6_pathway_analysis(
            predictions, top_n=args.top_n_drugs, skip=args.skip_pathway
        )
        
        # Step 7: Visualization
        step_7_visualization(similarity_results, clustering_results, predictions)
        
        # Step 8: Combination Therapy Analysis
        combinations = step_8_combination_therapy(
            predictions, similarity_results, pathway_results, 
            top_n=args.top_n_drugs, skip=args.skip_combination
        )
        
        # Step 9: Model Comparison
        model_comparison = step_9_model_comparison(data, features_df, skip=args.skip_model_comparison)
        
        # Step 10: Drug-Drug Interactions
        interactions = step_10_drug_interactions(combinations, smiles_dict, skip=args.skip_interactions)
        
        # Generate final report with all results
        generate_final_report(
            data, features_df, similarity_results, 
            clustering_results, predictions, pathway_results,
            combinations, model_comparison, interactions
        )
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 80)
        print("\n✅ Pipeline execution completed successfully!")
        print(f"📁 Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

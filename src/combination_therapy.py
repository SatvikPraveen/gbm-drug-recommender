"""
Combination Therapy Analysis Module

Analyzes drug pairs for potential synergistic effects in GBM treatment.

Scoring Criteria:
1. Pathway Complementarity - Non-overlapping therapeutic pathways
2. Target Profile - Different molecular targets for broader coverage
3. Similarity Balance - Optimal chemical similarity (0.3-0.7 range)
4. Prediction Scores - Both drugs ranked as promising candidates

Algorithm:
- Generates all pairwise combinations from top predicted drugs
- Computes multi-component synergy score (0-1 scale)
- Provides rationale for each recommendation
- Filters and ranks by total synergy potential

Outputs:
- drug_combinations.csv - Ranked combinations with scores
- drug_combinations_matrix.csv - Pairwise synergy matrix

Usage:
    analyzer = CombinationTherapyAnalyzer(similarity_matrices, pathway_data)
    combinations = analyzer.analyze_all_combinations(drug_list, top_n=20)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from itertools import combinations
from scipy.stats import hypergeom
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CombinationTherapyAnalyzer:
    """
    Analyzes drug combinations for potential synergistic effects.
    
    Uses multiple criteria:
    1. Pathway complementarity - drugs targeting different pathways
    2. Target diversity - non-overlapping molecular targets
    3. Optimal similarity - not too similar (redundant) or too different (incompatible)
    4. Synergy scoring - integrated score from multiple factors
    """
    
    def __init__(self, 
                 similarity_matrices: Optional[Dict[str, np.ndarray]] = None,
                 pathway_data: Optional[pd.DataFrame] = None,
                 target_data: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the combination therapy analyzer.
        
        Args:
            similarity_matrices: Dict of similarity matrices from different methods
            pathway_data: DataFrame with pathway enrichment results
            target_data: Dict mapping drug names to target gene lists
        """
        self.similarity_matrices = similarity_matrices or {}
        self.pathway_data = pathway_data
        self.target_data = target_data or {}
        
        # Optimal similarity parameters
        self.min_similarity = 0.3  # Too different below this
        self.max_similarity = 0.7  # Too similar above this
        self.optimal_similarity = 0.5  # Sweet spot
        
    def analyze_all_combinations(self, 
                                 drug_list: List[str],
                                 top_n: int = 20) -> pd.DataFrame:
        """
        Analyze all pairwise drug combinations.
        
        Args:
            drug_list: List of drug names to analyze
            top_n: Number of top combinations to return
            
        Returns:
            DataFrame with combination analysis results
        """
        logger.info(f"Analyzing {len(list(combinations(drug_list, 2)))} drug combinations")
        
        results = []
        
        for drug_a, drug_b in combinations(drug_list, 2):
            combo_score = self._compute_combination_score(drug_a, drug_b)
            
            if combo_score['total_score'] > 0:  # Only keep promising combinations
                results.append({
                    'Drug_A': drug_a,
                    'Drug_B': drug_b,
                    'Combination': f"{drug_a} + {drug_b}",
                    'Total_Score': combo_score['total_score'],
                    'Pathway_Score': combo_score['pathway_score'],
                    'Target_Score': combo_score['target_score'],
                    'Similarity_Score': combo_score['similarity_score'],
                    'Rationale': combo_score['rationale']
                })
        
        df_results = pd.DataFrame(results)
        
        if len(df_results) > 0:
            df_results = df_results.sort_values('Total_Score', ascending=False)
            df_results = df_results.head(top_n).reset_index(drop=True)
            df_results['Rank'] = df_results.index + 1
        
        logger.info(f"Found {len(df_results)} promising combinations")
        return df_results
    
    def _compute_combination_score(self, drug_a: str, drug_b: str) -> Dict:
        """
        Compute comprehensive combination score for a drug pair.
        
        Args:
            drug_a: First drug name
            drug_b: Second drug name
            
        Returns:
            Dictionary with individual scores and rationale
        """
        scores = {
            'pathway_score': 0.0,
            'target_score': 0.0,
            'similarity_score': 0.0,
            'total_score': 0.0,
            'rationale': []
        }
        
        # 1. Pathway complementarity score
        pathway_score = self._compute_pathway_complementarity(drug_a, drug_b)
        scores['pathway_score'] = pathway_score
        
        if pathway_score > 0.6:
            scores['rationale'].append("Highly complementary pathways")
        elif pathway_score > 0.4:
            scores['rationale'].append("Moderately complementary pathways")
        
        # 2. Target diversity score
        target_score = self._compute_target_diversity(drug_a, drug_b)
        scores['target_score'] = target_score
        
        if target_score > 0.7:
            scores['rationale'].append("Distinct molecular targets")
        elif target_score > 0.5:
            scores['rationale'].append("Partially overlapping targets")
        
        # 3. Similarity score (Goldilocks zone)
        similarity_score = self._compute_similarity_score(drug_a, drug_b)
        scores['similarity_score'] = similarity_score
        
        if 0.4 <= similarity_score <= 0.6:
            scores['rationale'].append("Optimal similarity balance")
        
        # 4. Compute total weighted score
        weights = {
            'pathway': 0.4,
            'target': 0.35,
            'similarity': 0.25
        }
        
        scores['total_score'] = (
            weights['pathway'] * pathway_score +
            weights['target'] * target_score +
            weights['similarity'] * similarity_score
        )
        
        scores['rationale'] = '; '.join(scores['rationale']) if scores['rationale'] else "Low synergy potential"
        
        return scores
    
    def _compute_pathway_complementarity(self, drug_a: str, drug_b: str) -> float:
        """
        Calculate pathway complementarity score.
        
        High score means drugs target different but related pathways.
        """
        if self.pathway_data is None or self.pathway_data.empty:
            return 0.5  # Neutral score if no pathway data
        
        # Get pathway enrichment for each drug
        pathways_a = self._get_drug_pathways(drug_a)
        pathways_b = self._get_drug_pathways(drug_b)
        
        if not pathways_a or not pathways_b:
            return 0.5
        
        # Calculate Jaccard distance (1 - Jaccard similarity)
        # High distance = complementary pathways
        intersection = len(pathways_a & pathways_b)
        union = len(pathways_a | pathways_b)
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        complementarity = 1 - jaccard_similarity
        
        # Normalize to favor some overlap (not completely unrelated)
        # Best score when 20-40% overlap
        if 0.2 <= jaccard_similarity <= 0.4:
            return 0.9  # High complementarity with some overlap
        elif jaccard_similarity < 0.2:
            return 0.5 + complementarity * 0.3  # Completely different
        else:
            return complementarity * 0.8  # Too similar
    
    def _get_drug_pathways(self, drug_name: str) -> set:
        """Get set of pathways for a drug."""
        if self.pathway_data is None:
            return set()
        
        drug_pathways = self.pathway_data[
            self.pathway_data['Drug'].str.lower() == drug_name.lower()
        ]
        
        if 'Pathway' in drug_pathways.columns:
            return set(drug_pathways['Pathway'].tolist())
        return set()
    
    def _compute_target_diversity(self, drug_a: str, drug_b: str) -> float:
        """
        Calculate target diversity score.
        
        High score means drugs target different genes/proteins.
        """
        if not self.target_data:
            return 0.5  # Neutral score if no target data
        
        targets_a = set(self.target_data.get(drug_a, []))
        targets_b = set(self.target_data.get(drug_b, []))
        
        if not targets_a or not targets_b:
            return 0.5
        
        # Calculate diversity (1 - overlap coefficient)
        intersection = len(targets_a & targets_b)
        min_size = min(len(targets_a), len(targets_b))
        
        if min_size == 0:
            return 0.0
        
        overlap = intersection / min_size
        diversity = 1 - overlap
        
        # High diversity is good for combination therapy
        return diversity
    
    def _compute_similarity_score(self, drug_a: str, drug_b: str) -> Union[float, Any]:
        """
        Calculate similarity score using the "Goldilocks zone" principle.
        
        Optimal similarity is not too high (redundant) and not too low (unrelated).
        """
        if not self.similarity_matrices:
            return 0.5
        
        # Use average across all similarity methods
        similarities = []
        
        for method_name, sim_matrix in self.similarity_matrices.items():
            if isinstance(sim_matrix, pd.DataFrame):
                if drug_a in sim_matrix.index and drug_b in sim_matrix.columns:
                    sim_value = sim_matrix.loc[drug_a, drug_b]
                    similarities.append(sim_value)
        
        if not similarities:
            return 0.5
        
        avg_similarity = np.mean(similarities)
        
        # Score based on distance from optimal similarity
        distance_from_optimal = abs(avg_similarity - self.optimal_similarity)
        
        # Best score at optimal similarity, decreasing as we move away
        max_distance = self.optimal_similarity  # Max meaningful distance
        score = 1 - (distance_from_optimal / max_distance)
        score = max(0, min(1, score))  # Clamp to [0, 1]
        
        return score
    
    def compute_synergy_matrix(self, drug_list: List[str]) -> pd.DataFrame:
        """
        Create a matrix of pairwise synergy scores.
        
        Args:
            drug_list: List of drugs to analyze
            
        Returns:
            Symmetric matrix of synergy scores
        """
        n_drugs = len(drug_list)
        synergy_matrix = np.zeros((n_drugs, n_drugs))
        
        for i, drug_a in enumerate(drug_list):
            for j, drug_b in enumerate(drug_list):
                if i != j:
                    score = self._compute_combination_score(drug_a, drug_b)
                    synergy_matrix[i, j] = score['total_score']
        
        df_matrix = pd.DataFrame(
            synergy_matrix,
            index=drug_list,
            columns=drug_list
        )
        
        return df_matrix
    
    def export_results(self, 
                      results_df: pd.DataFrame,
                      output_path: Path,
                      include_matrix: bool = False):
        """
        Export combination therapy results.
        
        Args:
            results_df: DataFrame with combination analysis
            output_path: Path to save results
            include_matrix: Whether to also save synergy matrix
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved combination therapy results to {output_path}")
        
        # Optionally save synergy matrix
        if include_matrix and len(results_df) > 0:
            drug_list = list(set(results_df['Drug_A'].tolist() + results_df['Drug_B'].tolist()))
            synergy_matrix = self.compute_synergy_matrix(drug_list)
            
            matrix_path = output_path.parent / f"{output_path.stem}_matrix.csv"
            synergy_matrix.to_csv(matrix_path)
            logger.info(f"Saved synergy matrix to {matrix_path}")
        
        return output_path


def analyze_bliss_independence(
    drug_a_response: np.ndarray,
    drug_b_response: np.ndarray,
    combo_response: np.ndarray
) -> Dict[str, Union[float, str]]:
    """
    Calculate Bliss Independence score for experimental synergy.
    
    Bliss Independence: E(A+B) = EA + EB - EA*EB
    
    Args:
        drug_a_response: Response to drug A alone (0-1, where 1 is max effect)
        drug_b_response: Response to drug B alone
        combo_response: Response to combination
        
    Returns:
        Dictionary with Bliss scores and synergy classification
    """
    # Expected effect under independence
    bliss_expected = drug_a_response + drug_b_response - (drug_a_response * drug_b_response)
    
    # Synergy score (positive = synergy, negative = antagonism)
    bliss_excess = combo_response - bliss_expected
    
    # Classification
    if np.mean(bliss_excess) > 0.1:
        classification = "Synergistic"
    elif np.mean(bliss_excess) < -0.1:
        classification = "Antagonistic"
    else:
        classification = "Additive"
    
    return {
        'bliss_expected': float(np.mean(bliss_expected)),
        'bliss_observed': float(np.mean(combo_response)),
        'bliss_excess': float(np.mean(bliss_excess)),
        'classification': classification
    }

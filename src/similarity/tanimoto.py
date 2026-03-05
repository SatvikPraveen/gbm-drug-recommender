"""
Tanimoto Similarity Analysis Module

Computes chemical similarity using molecular fingerprints and Tanimoto coefficient.

Method:
- Morgan (ECFP) fingerprints by default
- Alternative: MACCS keys (166-bit structural keys)
- Tanimoto coefficient: Intersection / Union of bit sets
- Values range 0-1 (0=completely different, 1=identical)

Fingerprints:
- Morgan: Circular fingerprints capturing local structure
- Radius: 2 (equivalent to ECFP4)
- Bits: 2048 (configurable)

Outputs:
- tanimoto_similarity.csv - Pairwise similarity matrix
- Symmetric matrix with drugs as rows/columns

Usage:
    analyzer = TanimotoSimilarityAnalyzer()
    matrix = analyzer.build_similarity_matrix(smiles_dict)
    analyzer.save_similarity_matrix(matrix)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from ..config import (
    FINGERPRINT_TYPE, FINGERPRINT_RADIUS, FINGERPRINT_BITS,
    TANIMOTO_THRESHOLD, SIMILARITY_RESULTS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TanimotoSimilarityAnalyzer:
    """Calculate Tanimoto similarity between molecular fingerprints"""
    
    def __init__(self, fingerprint_type: str = FINGERPRINT_TYPE):
        """
        Initialize Tanimoto analyzer
        
        Args:
            fingerprint_type: Type of fingerprint to use ('Morgan', 'MACCS', 'RDKit')
        """
        self.fingerprint_type = fingerprint_type
        self.fingerprints_cache = {}
    
    def generate_fingerprint(self, smiles: str):
        """
        Generate molecular fingerprint from SMILES
        
        Args:
            smiles: SMILES string
            
        Returns:
            RDKit fingerprint object
        """
        if smiles in self.fingerprints_cache:
            return self.fingerprints_cache[smiles]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if self.fingerprint_type == 'Morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
                )
            elif self.fingerprint_type == 'MACCS':
                fp = MACCSkeys.GenMACCSKeys(mol)
            else:  # Default to Morgan
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
                )
            
            self.fingerprints_cache[smiles] = fp
            return fp
        
        except Exception as e:
            logger.debug(f"Error generating fingerprint for SMILES {smiles}: {e}")
            return None
    
    def calculate_pairwise_similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate Tanimoto similarity between two SMILES
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Tanimoto similarity score (0-1)
        """
        fp1 = self.generate_fingerprint(smiles1)
        fp2 = self.generate_fingerprint(smiles2)
        
        if fp1 is None or fp2 is None:
            return 0.0
        
        try:
            similarity = TanimotoSimilarity(fp1, fp2)
            return similarity
        except Exception as e:
            logger.debug(f"Error calculating Tanimoto similarity: {e}")
            return 0.0
    
    def build_similarity_matrix(self, drug_smiles_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Build pairwise similarity matrix for all drugs
        
        Args:
            drug_smiles_dict: Dictionary mapping drug names to SMILES
            
        Returns:
            DataFrame with similarity matrix
        """
        logger.info(f"Building Tanimoto similarity matrix for {len(drug_smiles_dict)} drugs")
        
        drug_names = list(drug_smiles_dict.keys())
        n_drugs = len(drug_names)
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n_drugs, n_drugs))
        
        # Calculate pairwise similarities
        for i in tqdm(range(n_drugs), desc="Calculating Tanimoto similarities"):
            for j in range(i, n_drugs):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.calculate_pairwise_similarity(
                        drug_smiles_dict[drug_names[i]],
                        drug_smiles_dict[drug_names[j]]
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # Convert to DataFrame
        sim_df = pd.DataFrame(similarity_matrix, index=drug_names, columns=drug_names)
        
        logger.info("Tanimoto similarity matrix complete")
        return sim_df
    
    def find_similar_drugs(self, target_drug: str, 
                          drug_smiles_dict: Dict[str, str],
                          threshold: float = TANIMOTO_THRESHOLD,
                          top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find drugs similar to a target drug
        
        Args:
            target_drug: Name of target drug
            drug_smiles_dict: Dictionary mapping drug names to SMILES
            threshold: Minimum similarity threshold
            top_n: Number of top similar drugs to return
            
        Returns:
            List of (drug_name, similarity_score) tuples
        """
        if target_drug not in drug_smiles_dict:
            logger.warning(f"Target drug '{target_drug}' not found in database")
            return []
        
        target_smiles = drug_smiles_dict[target_drug]
        similarities = []
        
        for drug_name, smiles in drug_smiles_dict.items():
            if drug_name == target_drug:
                continue
            
            sim = self.calculate_pairwise_similarity(target_smiles, smiles)
            
            if sim >= threshold:
                similarities.append((drug_name, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    def get_drug_pairs_above_threshold(self, similarity_matrix: pd.DataFrame,
                                       threshold: float = TANIMOTO_THRESHOLD) -> pd.DataFrame:
        """
        Get all drug pairs with similarity above threshold
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            threshold: Minimum similarity threshold
            
        Returns:
            DataFrame with drug pairs and their similarities
        """
        pairs = []
        
        drugs = similarity_matrix.index.tolist()
        
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                sim = similarity_matrix.iloc[i, j]
                if sim >= threshold:
                    pairs.append({
                        'drug1': drugs[i],
                        'drug2': drugs[j],
                        'tanimoto_similarity': sim
                    })
        
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values('tanimoto_similarity', ascending=False)
        
        logger.info(f"Found {len(pairs_df)} drug pairs with Tanimoto similarity >= {threshold}")
        
        return pairs_df
    
    def save_similarity_matrix(self, similarity_matrix: pd.DataFrame, 
                              filename: str = "tanimoto_similarity_matrix.csv"):
        """
        Save similarity matrix to file
        
        Args:
            similarity_matrix: Similarity matrix DataFrame
            filename: Output filename
        """
        output_path = SIMILARITY_RESULTS_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        similarity_matrix.to_csv(output_path)
        logger.info(f"Tanimoto similarity matrix saved to {output_path}")
    
    def get_similarity_statistics(self, similarity_matrix: pd.DataFrame) -> Dict:
        """
        Calculate statistics for the similarity matrix
        
        Args:
            similarity_matrix: Similarity matrix DataFrame
            
        Returns:
            Dictionary with statistics
        """
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(similarity_matrix.values, k=1)
        similarities = upper_triangle[upper_triangle > 0]
        
        stats = {
            'mean_similarity': np.mean(similarities),
            'median_similarity': np.median(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'n_comparisons': len(similarities)
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    test_drugs = {
        'Doxorubicin': 'CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O',
        'Gemcitabine': 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F',
        'Temozolomide': 'CN1C(=O)N=C2C(=O)N(C(=N2)C(=O)N)C1=O'
    }
    
    analyzer = TanimotoSimilarityAnalyzer()
    
    # Build similarity matrix
    sim_matrix = analyzer.build_similarity_matrix(test_drugs)
    print("\nTanimoto Similarity Matrix:")
    print(sim_matrix)
    
    # Find similar drugs
    similar = analyzer.find_similar_drugs('Doxorubicin', test_drugs, threshold=0.3)
    print(f"\nDrugs similar to Doxorubicin:")
    for drug, score in similar:
        print(f"  {drug}: {score:.3f}")

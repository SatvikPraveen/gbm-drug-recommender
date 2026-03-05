"""
Maximum Common Substructure (MCS) Similarity Analysis

Identifies largest shared substructure between drug pairs.

Algorithm:
- Finds maximum common substructure using graph matching
- Computes similarity as: MCS_size / average(mol1_size, mol2_size)
- Timeout protection for complex comparisons (5 seconds default)
- Handles disconnected fragments and ring systems

Advantages:
- Captures actual structural overlap
- More interpretable than fingerprint-based methods
- Identifies conserved scaffolds

Limitations:
- Computationally expensive (O(n²) comparisons)
- May timeout on very large/complex molecules

Outputs:
- mcs_similarity.csv - Pairwise MCS-based similarity matrix
- Values 0-1 based on fraction of structure overlap

Usage:
    analyzer = MCSimilarityAnalyzer()
    matrix = analyzer.build_similarity_matrix(smiles_dict)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MCS
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from ..config import MCS_TIMEOUT, MCS_THRESHOLD, SIMILARITY_RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCSimilarityAnalyzer:
    """Calculate Maximum Common Substructure similarity"""
    
    def __init__(self, timeout: float = MCS_TIMEOUT):
        """
        Initialize MCS analyzer
        
        Args:
            timeout: Timeout in seconds for MCS calculation
        """
        self.timeout = timeout
        self.mcs_cache = {}
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES to RDKit molecule
        
        Args:
            smiles: SMILES string
            
        Returns:
            RDKit Mol object or None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception as e:
            logger.debug(f"Error converting SMILES: {e}")
            return None
    
    def calculate_mcs(self, mol1: Chem.Mol, mol2: Chem.Mol) -> Optional[MCS.MCSResult]:
        """
        Calculate Maximum Common Substructure between two molecules
        
        Args:
            mol1: First RDKit molecule
            mol2: Second RDKit molecule
            
        Returns:
            MCS result object or None
        """
        if mol1 is None or mol2 is None:
            return None
        
        try:
            mcs_result = MCS.FindMCS(
                [mol1, mol2],
                timeout=self.timeout,
                atomCompare=MCS.AtomCompare.CompareElements,
                bondCompare=MCS.BondCompare.CompareOrder,
                ringMatchesRingOnly=True
            )
            return mcs_result
        except Exception as e:
            logger.debug(f"Error calculating MCS: {e}")
            return None
    
    def calculate_mcs_similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate MCS-based similarity between two SMILES
        
        Similarity = |MCS| / max(|mol1|, |mol2|)
        where |mol| is the number of atoms in the molecule
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            MCS similarity score (0-1)
        """
        # Check cache
        cache_key = tuple(sorted([smiles1, smiles2]))
        if cache_key in self.mcs_cache:
            return self.mcs_cache[cache_key]
        
        mol1 = self.smiles_to_mol(smiles1)
        mol2 = self.smiles_to_mol(smiles2)
        
        if mol1 is None or mol2 is None:
            self.mcs_cache[cache_key] = 0.0
            return 0.0
        
        # Get molecule sizes
        size1 = mol1.GetNumAtoms()
        size2 = mol2.GetNumAtoms()
        max_size = max(size1, size2)
        
        if max_size == 0:
            self.mcs_cache[cache_key] = 0.0
            return 0.0
        
        # Calculate MCS
        mcs_result = self.calculate_mcs(mol1, mol2)
        
        if mcs_result is None or mcs_result.numAtoms == 0:
            self.mcs_cache[cache_key] = 0.0
            return 0.0
        
        # Calculate similarity
        similarity = mcs_result.numAtoms / max_size
        
        self.mcs_cache[cache_key] = similarity
        return similarity
    
    def build_similarity_matrix(self, drug_smiles_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Build pairwise MCS similarity matrix for all drugs
        
        Args:
            drug_smiles_dict: Dictionary mapping drug names to SMILES
            
        Returns:
            DataFrame with similarity matrix
        """
        logger.info(f"Building MCS similarity matrix for {len(drug_smiles_dict)} drugs")
        logger.info(f"MCS timeout set to {self.timeout} seconds per comparison")
        
        drug_names = list(drug_smiles_dict.keys())
        n_drugs = len(drug_names)
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n_drugs, n_drugs))
        
        # Calculate pairwise similarities
        total_comparisons = (n_drugs * (n_drugs - 1)) // 2
        
        with tqdm(total=total_comparisons, desc="Calculating MCS similarities") as pbar:
            for i in range(n_drugs):
                for j in range(i, n_drugs):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        sim = self.calculate_mcs_similarity(
                            drug_smiles_dict[drug_names[i]],
                            drug_smiles_dict[drug_names[j]]
                        )
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
                        pbar.update(1)
        
        # Convert to DataFrame
        sim_df = pd.DataFrame(similarity_matrix, index=drug_names, columns=drug_names)
        
        logger.info("MCS similarity matrix complete")
        return sim_df
    
    def find_similar_drugs(self, target_drug: str,
                          drug_smiles_dict: Dict[str, str],
                          threshold: float = MCS_THRESHOLD,
                          top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find drugs similar to a target drug based on MCS
        
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
        
        for drug_name, smiles in tqdm(drug_smiles_dict.items(), desc=f"Finding similar to {target_drug}"):
            if drug_name == target_drug:
                continue
            
            sim = self.calculate_mcs_similarity(target_smiles, smiles)
            
            if sim >= threshold:
                similarities.append((drug_name, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    def get_mcs_substructure(self, smiles1: str, smiles2: str) -> Optional[str]:
        """
        Get the SMARTS string of the MCS between two molecules
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            SMARTS string of MCS or None
        """
        mol1 = self.smiles_to_mol(smiles1)
        mol2 = self.smiles_to_mol(smiles2)
        
        if mol1 is None or mol2 is None:
            return None
        
        mcs_result = self.calculate_mcs(mol1, mol2)
        
        if mcs_result is None:
            return None
        
        return mcs_result.smartsString
    
    def get_drug_pairs_above_threshold(self, similarity_matrix: pd.DataFrame,
                                       threshold: float = MCS_THRESHOLD) -> pd.DataFrame:
        """
        Get all drug pairs with MCS similarity above threshold
        
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
                        'mcs_similarity': sim
                    })
        
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values('mcs_similarity', ascending=False)
        
        logger.info(f"Found {len(pairs_df)} drug pairs with MCS similarity >= {threshold}")
        
        return pairs_df
    
    def save_similarity_matrix(self, similarity_matrix: pd.DataFrame,
                              filename: str = "mcs_similarity_matrix.csv"):
        """
        Save similarity matrix to file
        
        Args:
            similarity_matrix: Similarity matrix DataFrame
            filename: Output filename
        """
        output_path = SIMILARITY_RESULTS_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        similarity_matrix.to_csv(output_path)
        logger.info(f"MCS similarity matrix saved to {output_path}")
    
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
            'mean_similarity': np.mean(similarities) if len(similarities) > 0 else 0,
            'median_similarity': np.median(similarities) if len(similarities) > 0 else 0,
            'std_similarity': np.std(similarities) if len(similarities) > 0 else 0,
            'min_similarity': np.min(similarities) if len(similarities) > 0 else 0,
            'max_similarity': np.max(similarities) if len(similarities) > 0 else 0,
            'n_comparisons': len(similarities)
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    test_drugs = {
        'Vincristine': 'COC1=C(C=C2C(=C1)C34CCN5C3CC(C(C4N(C2=O)CC5)(C6=C(C78CCN9C7C(C(C9)(C=CC8=C6)CC)O)CC)O)OC(=O)C)OC',
        'Vinblastine': 'COC1=C(C=C2C(=C1)C34CCN5C3CC(C(C4N(C2=O)CC5)(C6=C(C78CCN9C7C(C(C9)(C=CC8=C6)CC)O)CC)O)C(=O)OC)OC'
    }
    
    analyzer = MCSimilarityAnalyzer(timeout=5.0)
    
    # Calculate MCS similarity
    smiles1 = test_drugs['Vincristine']
    smiles2 = test_drugs['Vinblastine']
    
    sim = analyzer.calculate_mcs_similarity(smiles1, smiles2)
    print(f"\nMCS Similarity between Vincristine and Vinblastine: {sim:.4f}")
    
    # Get MCS substructure
    mcs_smarts = analyzer.get_mcs_substructure(smiles1, smiles2)
    print(f"MCS SMARTS: {mcs_smarts}")

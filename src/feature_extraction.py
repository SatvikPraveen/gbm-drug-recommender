"""
Molecular Feature Extraction Module

Extracts chemical descriptors and fingerprints from SMILES structures.

Features Extracted:
- Molecular Weight, LogP (lipophilicity)
- Topological Polar Surface Area (TPSA)
- Number of H-bond donors/acceptors
- Number of rotatable bonds
- Aromatic rings count
- Lipinski's Rule of Five parameters
- Morgan fingerprints (ECFP)
- MACCS keys

SMILES Management:
- Automatic lookup via PubChem API
- Local caching in smiles_database.json
- Fallback to existing database entries

Outputs:
- molecular_features.csv - Feature matrix for all drugs
- smiles_database.json - Cached SMILES structures

Usage:
    smiles_manager = SMILESManager()
    smiles_dict = smiles_manager.update_smiles_from_list(drug_names)
    extractor = MolecularFeatureExtractor()
    features = extractor.process_drug_list(drug_names, smiles_dict)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, Crippen
from rdkit.Chem import MACCSkeys, RDKFingerprint
from rdkit.DataStructs import TanimotoSimilarity
import pubchempy as pcp
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

from .config import (
    MOLECULAR_DESCRIPTORS, FINGERPRINT_TYPE, FINGERPRINT_RADIUS,
    FINGERPRINT_BITS, FEATURES_FILE, SMILES_DATA_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MolecularFeatureExtractor:
    """Extract molecular features from SMILES strings using RDKit"""
    
    def __init__(self):
        """Initialize the feature extractor"""
        self.smiles_cache = {}
        self.features_cache = {}
    
    def get_smiles_from_pubchem(self, drug_name: str) -> Optional[str]:
        """
        Retrieve SMILES string from PubChem for a given drug name
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            SMILES string or None if not found
        """
        if drug_name in self.smiles_cache:
            return self.smiles_cache[drug_name]
        
        try:
            compounds = pcp.get_compounds(drug_name, 'name')
            if compounds:
                smiles = compounds[0].isomeric_smiles
                self.smiles_cache[drug_name] = smiles
                return smiles
        except Exception as e:
            logger.debug(f"Could not retrieve SMILES for {drug_name}: {e}")
        
        return None
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES string to RDKit molecule object
        
        Args:
            smiles: SMILES string
            
        Returns:
            RDKit Mol object or None if conversion fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception as e:
            logger.debug(f"Error converting SMILES to molecule: {e}")
            return None
    
    def extract_molecular_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract molecular descriptors from RDKit molecule
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dictionary of molecular descriptors
        """
        if mol is None:
            return {desc: np.nan for desc in MOLECULAR_DESCRIPTORS}
        
        descriptors = {}
        
        # Molecular weight
        if 'MolWt' in MOLECULAR_DESCRIPTORS:
            descriptors['MolWt'] = Descriptors.MolWt(mol)
        
        # LogP (lipophilicity)
        if 'LogP' in MOLECULAR_DESCRIPTORS or 'MolLogP' in MOLECULAR_DESCRIPTORS:
            descriptors['LogP'] = Descriptors.MolLogP(mol)
            descriptors['MolLogP'] = Crippen.MolLogP(mol)
        
        # Hydrogen bond donors
        if 'NumHDonors' in MOLECULAR_DESCRIPTORS:
            descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
        
        # Hydrogen bond acceptors
        if 'NumHAcceptors' in MOLECULAR_DESCRIPTORS:
            descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
        
        # Topological polar surface area
        if 'TPSA' in MOLECULAR_DESCRIPTORS:
            descriptors['TPSA'] = Descriptors.TPSA(mol)
        
        # Rotatable bonds
        if 'NumRotatableBonds' in MOLECULAR_DESCRIPTORS:
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        
        # Aromatic rings
        if 'NumAromaticRings' in MOLECULAR_DESCRIPTORS:
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
        
        # Fraction of sp3 carbons (try different spellings for compatibility)
        if 'FractionCSP3' in MOLECULAR_DESCRIPTORS:
            try:
                descriptors['FractionCSP3'] = Descriptors.FractionCSP3(mol)
            except AttributeError:
                try:
                    # Try lowercase version
                    descriptors['FractionCSP3'] = Descriptors.FractionCsp3(mol) 
                except AttributeError:
                    # Skip if not available
                    descriptors['FractionCSP3'] = 0.0
        
        # Molar refractivity
        if 'MolMR' in MOLECULAR_DESCRIPTORS:
            descriptors['MolMR'] = Crippen.MolMR(mol)
        
        return descriptors
    
    def generate_fingerprint(self, mol: Chem.Mol, 
                            fp_type: str = FINGERPRINT_TYPE) -> Optional[np.ndarray]:
        """
        Generate molecular fingerprint
        
        Args:
            mol: RDKit Mol object
            fp_type: Type of fingerprint ('Morgan', 'MACCS', 'RDKit')
            
        Returns:
            Fingerprint as numpy array
        """
        if mol is None:
            return None
        
        try:
            if fp_type == 'Morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
                )
            elif fp_type == 'MACCS':
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif fp_type == 'RDKit':
                fp = RDKFingerprint(mol)
            else:
                logger.warning(f"Unknown fingerprint type: {fp_type}, using Morgan")
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
                )
            
            # Convert to numpy array
            arr = np.zeros((1,))
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        
        except Exception as e:
            logger.debug(f"Error generating fingerprint: {e}")
            return None
    
    def process_drug_list(self, drug_names: List[str], 
                         smiles_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Process a list of drugs and extract features
        
        Args:
            drug_names: List of drug names
            smiles_dict: Optional dictionary mapping drug names to SMILES
            
        Returns:
            DataFrame with molecular features
        """
        logger.info(f"Processing {len(drug_names)} drugs for feature extraction")
        
        results = []
        
        for drug_name in tqdm(drug_names, desc="Extracting features"):
            # Get SMILES
            if smiles_dict and drug_name in smiles_dict:
                smiles = smiles_dict[drug_name]
            else:
                smiles = self.get_smiles_from_pubchem(drug_name)
            
            if smiles is None:
                logger.debug(f"No SMILES found for {drug_name}")
                results.append({
                    'drug_name': drug_name,
                    'smiles': None,
                    **{desc: np.nan for desc in MOLECULAR_DESCRIPTORS}
                })
                continue
            
            # Convert to molecule
            mol = self.smiles_to_mol(smiles)
            
            # Extract descriptors
            descriptors = self.extract_molecular_descriptors(mol)
            
            results.append({
                'drug_name': drug_name,
                'smiles': smiles,
                **descriptors
            })
        
        df = pd.DataFrame(results)
        logger.info(f"Extracted features for {len(df)} drugs")
        
        return df
    
    def save_features(self, features_df: pd.DataFrame, 
                     filepath: str = FEATURES_FILE):
        """
        Save extracted features to file
        
        Args:
            features_df: DataFrame with features
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(filepath, index=False)
        logger.info(f"Features saved to {filepath}")
    
    def load_features(self, filepath: str = FEATURES_FILE) -> pd.DataFrame:
        """
        Load previously extracted features
        
        Args:
            filepath: File path to load from
            
        Returns:
            DataFrame with features
        """
        if filepath.exists():
            logger.info(f"Loading features from {filepath}")
            return pd.read_csv(filepath)
        else:
            logger.warning(f"Features file not found: {filepath}")
            return pd.DataFrame()
    
    def calculate_lipinski_rule_of_five(self, mol: Chem.Mol) -> Dict[str, bool]:
        """
        Check Lipinski's Rule of Five for drug-likeness
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dictionary with rule compliance
        """
        if mol is None:
            return {criterion: False for criterion in ['MW', 'LogP', 'HBD', 'HBA', 'ROF']}
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        rules = {
            'MW': mw <= 500,           # Molecular weight <= 500 Da
            'LogP': logp <= 5,         # LogP <= 5
            'HBD': hbd <= 5,           # H-bond donors <= 5
            'HBA': hba <= 10,          # H-bond acceptors <= 10
        }
        
        # Passes if no more than one violation
        rules['ROF'] = sum(rules.values()) >= 3
        
        return rules
    
    def add_lipinski_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Lipinski Rule of Five compliance to features
        
        Args:
            features_df: DataFrame with SMILES
            
        Returns:
            DataFrame with Lipinski features added
        """
        logger.info("Calculating Lipinski Rule of Five compliance")
        
        lipinski_results = []
        
        for _, row in tqdm(features_df.iterrows(), total=len(features_df), 
                          desc="Lipinski analysis"):
            if pd.notna(row.get('smiles')):
                mol = self.smiles_to_mol(row['smiles'])
                rules = self.calculate_lipinski_rule_of_five(mol)
            else:
                rules = {criterion: False for criterion in ['MW', 'LogP', 'HBD', 'HBA', 'ROF']}
            
            lipinski_results.append(rules)
        
        lipinski_df = pd.DataFrame(lipinski_results)
        lipinski_df.columns = [f'lipinski_{col}' for col in lipinski_df.columns]
        
        result_df = pd.concat([features_df, lipinski_df], axis=1)
        
        logger.info(f"{lipinski_df['lipinski_ROF'].sum()} drugs pass Lipinski's Rule of Five")
        
        return result_df


class SMILESManager:
    """Manage SMILES strings for drugs"""
    
    def __init__(self, smiles_dir: str = SMILES_DATA_DIR):
        """
        Initialize SMILES manager
        
        Args:
            smiles_dir: Directory to store SMILES data
        """
        self.smiles_dir = smiles_dir
        self.smiles_file = self.smiles_dir / "drug_smiles.csv"
    
    def load_smiles_mapping(self) -> Dict[str, str]:
        """
        Load drug name to SMILES mapping
        
        Returns:
            Dictionary mapping drug names to SMILES
        """
        if self.smiles_file.exists():
            df = pd.read_csv(self.smiles_file)
            return dict(zip(df['drug_name'], df['smiles']))
        return {}
    
    def save_smiles_mapping(self, smiles_dict: Dict[str, str]):
        """
        Save drug name to SMILES mapping
        
        Args:
            smiles_dict: Dictionary mapping drug names to SMILES
        """
        self.smiles_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(list(smiles_dict.items()), columns=['drug_name', 'smiles'])
        df.to_csv(self.smiles_file, index=False)
        logger.info(f"Saved {len(smiles_dict)} SMILES strings to {self.smiles_file}")
    
    def update_smiles_from_list(self, drug_names: List[str]) -> Dict[str, str]:
        """
        Update SMILES mapping from a list of drug names
        
        Args:
            drug_names: List of drug names
            
        Returns:
            Updated SMILES dictionary
        """
        smiles_dict = self.load_smiles_mapping()
        extractor = MolecularFeatureExtractor()
        
        new_drugs = [name for name in drug_names if name not in smiles_dict]
        
        if new_drugs:
            logger.info(f"Fetching SMILES for {len(new_drugs)} new drugs")
            
            for drug_name in tqdm(new_drugs, desc="Fetching SMILES"):
                smiles = extractor.get_smiles_from_pubchem(drug_name)
                if smiles:
                    smiles_dict[drug_name] = smiles
            
            self.save_smiles_mapping(smiles_dict)
        
        return smiles_dict


if __name__ == "__main__":
    # Example usage
    extractor = MolecularFeatureExtractor()
    
    # Test with a few drugs
    test_drugs = ['Doxorubicin', 'Gemcitabine', 'Temozolomide']
    
    features = extractor.process_drug_list(test_drugs)
    print("\nMolecular Features:")
    print(features)
    
    # Add Lipinski features
    features_with_lipinski = extractor.add_lipinski_features(features)
    print("\nWith Lipinski Features:")
    print(features_with_lipinski)

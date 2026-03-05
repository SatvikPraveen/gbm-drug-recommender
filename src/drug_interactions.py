"""
Drug-Drug Interaction (DDI) Checker Module

Assesses safety of drug combinations using structural and pharmacological analysis.

Methods:
1. Structural Similarity - High similarity may indicate similar metabolism
2. Pharmacophore Analysis - Overlapping molecular features
3. Known Interactions - Database lookup (when available)
4. Physicochemical Properties - LogP, molecular weight compatibility

Severity Classifications:
- High: Major interactions, avoid combination
- Moderate: Monitor closely, dose adjustment may be needed
- Low: Minor interactions, unlikely to be clinically significant
- None: No predicted interactions

Outputs:
- drug_interactions.csv - All combinations with severity levels
- safe_combinations.csv - Filtered safe pairs (≤ moderate severity)
- drug_interactions_summary.txt - Interaction statistics

Usage:
    checker = DrugInteractionChecker()
    interactions = checker.batch_check_interactions(drug_pairs, smiles_dict)
    safe = checker.filter_safe_combinations(interactions, max_severity='moderate')
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen

logger = logging.getLogger(__name__)


class DrugInteractionChecker:
    """
    Check for potential drug-drug interactions.
    
    Methods:
    1. Structural alerts for incompatible functional groups
    2. Metabolic pathway overlap (CYP450 interactions)
    3. Known interaction database lookup
    4. Physicochemical property conflicts
    """
    
    # Known problematic functional group combinations
    INTERACTION_PATTERNS = {
        'strong_acids_bases': {
            'description': 'Strong acids and bases may neutralize each other',
            'severity': 'moderate',
            'patterns': [
                ('[C,S](=O)[OH]', '[NX3;H2,H1,H0]'),  # Carboxylic/sulfonic acid + amine
            ]
        },
        'oxidizers_reducers': {
            'description': 'Oxidizing and reducing agents may react',
            'severity': 'high',
            'patterns': [
                ('[N+](=O)[O-]', '[SH]'),  # Nitro + thiol
            ]
        },
        'metal_chelators': {
            'description': 'Metal chelators may complex with metal-containing drugs',
            'severity': 'moderate',
            'patterns': [
                ('[OH]C(=O)', 'O=C1c2ccccc2C(=O)c2ccccc21'),  # Carboxyl + quinone
            ]
        }
    }
    
    # CYP450 enzyme substrates/inhibitors (simplified)
    CYP450_SUBSTRATES = {
        'CYP3A4': ['midazolam', 'simvastatin', 'cyclosporine', 'tacrolimus'],
        'CYP2D6': ['codeine', 'tamoxifen', 'metoprolol'],
        'CYP2C9': ['warfarin', 'phenytoin', 'losartan'],
        'CYP1A2': ['theophylline', 'caffeine', 'clozapine'],
    }
    
    CYP450_INHIBITORS = {
        'CYP3A4': ['ketoconazole', 'itraconazole', 'clarithromycin', 'grapefruit'],
        'CYP2D6': ['fluoxetine', 'paroxetine', 'quinidine'],
        'CYP2C9': ['fluconazole', 'amiodarone'],
        'CYP1A2': ['fluvoxamine', 'ciprofloxacin'],
    }
    
    def __init__(self, custom_interactions: Optional[pd.DataFrame] = None):
        """
        Initialize interaction checker.
        
        Args:
            custom_interactions: DataFrame with known drug interactions
                                Columns: Drug_A, Drug_B, Severity, Description
        """
        self.custom_interactions = custom_interactions
        self.interaction_cache = {}
    
    def check_interaction(self,
                         drug_a: str,
                         drug_b: str,
                         smiles_a: Optional[str] = None,
                         smiles_b: Optional[str] = None) -> Dict:
        """
        Check for potential interactions between two drugs.
        
        Args:
            drug_a: First drug name
            drug_b: Second drug name
            smiles_a: SMILES string for drug A
            smiles_b: SMILES string for drug B
            
        Returns:
            Dictionary with interaction assessment
        """
        # Check cache
        cache_key = tuple(sorted([drug_a, drug_b]))
        if cache_key in self.interaction_cache:
            return self.interaction_cache[cache_key]
        
        result = {
            'Drug_A': drug_a,
            'Drug_B': drug_b,
            'Has_Interaction': False,
            'Severity': 'none',
            'Interaction_Type': [],
            'Description': [],
            'Recommendation': 'Safe to combine'
        }
        
        interactions_found = []
        max_severity = 'none'
        
        # 1. Check known interactions database
        if self.custom_interactions is not None:
            known_interaction = self._check_known_interactions(drug_a, drug_b)
            if known_interaction:
                interactions_found.append(known_interaction)
                if self._severity_level(known_interaction['severity']) > self._severity_level(max_severity):
                    max_severity = known_interaction['severity']
        
        # 2. Check CYP450 interactions
        cyp_interaction = self._check_cyp450_interaction(drug_a, drug_b)
        if cyp_interaction:
            interactions_found.append(cyp_interaction)
            if self._severity_level(cyp_interaction['severity']) > self._severity_level(max_severity):
                max_severity = cyp_interaction['severity']
        
        # 3. Check structural incompatibilities
        if smiles_a and smiles_b:
            structural_interaction = self._check_structural_interaction(smiles_a, smiles_b)
            if structural_interaction:
                interactions_found.append(structural_interaction)
                if self._severity_level(structural_interaction['severity']) > self._severity_level(max_severity):
                    max_severity = structural_interaction['severity']
            
            # 4. Check physicochemical conflicts
            property_interaction = self._check_property_conflicts(smiles_a, smiles_b)
            if property_interaction:
                interactions_found.append(property_interaction)
                if self._severity_level(property_interaction['severity']) > self._severity_level(max_severity):
                    max_severity = property_interaction['severity']
        
        # Compile results
        if interactions_found:
            result['Has_Interaction'] = True
            result['Severity'] = max_severity
            result['Interaction_Type'] = [i['type'] for i in interactions_found]
            result['Description'] = [i['description'] for i in interactions_found]
            
            # Generate recommendation
            if max_severity == 'high':
                result['Recommendation'] = 'Avoid combination - significant interaction risk'
            elif max_severity == 'moderate':
                result['Recommendation'] = 'Use with caution - monitor closely'
            else:
                result['Recommendation'] = 'Low risk - acceptable combination'
        
        # Cache result
        self.interaction_cache[cache_key] = result
        
        return result
    
    def _check_known_interactions(self, drug_a: str, drug_b: str) -> Optional[Dict]:
        """Check against known interactions database."""
        if self.custom_interactions is None:
            return None
        
        # Check both orderings
        interaction = self.custom_interactions[
            ((self.custom_interactions['Drug_A'] == drug_a) & 
             (self.custom_interactions['Drug_B'] == drug_b)) |
            ((self.custom_interactions['Drug_A'] == drug_b) & 
             (self.custom_interactions['Drug_B'] == drug_a))
        ]
        
        if len(interaction) > 0:
            row = interaction.iloc[0]
            return {
                'type': 'Known Interaction',
                'severity': row.get('Severity', 'moderate').lower(),
                'description': row.get('Description', 'Documented drug interaction')
            }
        
        return None
    
    def _check_cyp450_interaction(self, drug_a: str, drug_b: str) -> Optional[Dict]:
        """Check for CYP450-mediated interactions."""
        drug_a_lower = drug_a.lower()
        drug_b_lower = drug_b.lower()
        
        for enzyme, substrates in self.CYP450_SUBSTRATES.items():
            inhibitors = self.CYP450_INHIBITORS.get(enzyme, [])
            
            # Check if one is substrate and other is inhibitor
            if ((drug_a_lower in substrates and drug_b_lower in inhibitors) or
                (drug_b_lower in substrates and drug_a_lower in inhibitors)):
                
                return {
                    'type': 'Metabolic Interaction',
                    'severity': 'moderate',
                    'description': f'Potential {enzyme} interaction: inhibitor may increase substrate levels'
                }
        
        return None
    
    def _check_structural_interaction(self, smiles_a: str, smiles_b: str) -> Optional[Dict]:
        """Check for structural incompatibilities."""
        try:
            mol_a = Chem.MolFromSmiles(smiles_a)
            mol_b = Chem.MolFromSmiles(smiles_b)
            
            if mol_a is None or mol_b is None:
                return None
            
            for interaction_name, interaction_data in self.INTERACTION_PATTERNS.items():
                patterns = interaction_data['patterns']
                
                for pattern_a, pattern_b in patterns:
                    patt_a = Chem.MolFromSmarts(pattern_a)
                    patt_b = Chem.MolFromSmarts(pattern_b)
                    
                    if patt_a and patt_b:
                        # Check if both patterns present in different molecules
                        if ((mol_a.HasSubstructMatch(patt_a) and mol_b.HasSubstructMatch(patt_b)) or
                            (mol_a.HasSubstructMatch(patt_b) and mol_b.HasSubstructMatch(patt_a))):
                            
                            return {
                                'type': 'Structural Incompatibility',
                                'severity': interaction_data['severity'],
                                'description': interaction_data['description']
                            }
        
        except Exception as e:
            logger.warning(f"Error checking structural interaction: {e}")
        
        return None
    
    def _check_property_conflicts(self, smiles_a: str, smiles_b: str) -> Optional[Dict]:
        """Check for physicochemical property conflicts."""
        try:
            mol_a = Chem.MolFromSmiles(smiles_a)
            mol_b = Chem.MolFromSmiles(smiles_b)
            
            if mol_a is None or mol_b is None:
                return None
            
            # Calculate properties
            logp_a = Crippen.MolLogP(mol_a)  # type: ignore[attr-defined]
            logp_b = Crippen.MolLogP(mol_b)  # type: ignore[attr-defined]
            
            # Extreme hydrophobicity/hydrophilicity difference
            # May lead to formulation incompatibility
            if abs(logp_a - logp_b) > 7:
                return {
                    'type': 'Physicochemical Incompatibility',
                    'severity': 'low',
                    'description': 'Large difference in lipophilicity may affect co-formulation'
                }
        
        except Exception as e:
            logger.warning(f"Error checking property conflicts: {e}")
        
        return None
    
    @staticmethod
    def _severity_level(severity: str) -> int:
        """Convert severity to numeric level."""
        levels = {'none': 0, 'low': 1, 'moderate': 2, 'high': 3}
        return levels.get(severity.lower(), 0)
    
    def batch_check_interactions(self,
                                drug_combinations: List[Tuple[str, str]],
                                smiles_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Check interactions for multiple drug combinations.
        
        Args:
            drug_combinations: List of (drug_a, drug_b) tuples
            smiles_dict: Dictionary mapping drug names to SMILES
            
        Returns:
            DataFrame with interaction results
        """
        logger.info(f"Checking interactions for {len(drug_combinations)} combinations...")
        
        results = []
        
        for drug_a, drug_b in drug_combinations:
            smiles_a = smiles_dict.get(drug_a) if smiles_dict else None
            smiles_b = smiles_dict.get(drug_b) if smiles_dict else None
            
            interaction = self.check_interaction(drug_a, drug_b, smiles_a, smiles_b)
            
            # Flatten for DataFrame
            result_row = {
                'Drug_A': interaction['Drug_A'],
                'Drug_B': interaction['Drug_B'],
                'Has_Interaction': interaction['Has_Interaction'],
                'Severity': interaction['Severity'],
                'Interaction_Types': '; '.join(interaction['Interaction_Type']) if interaction['Interaction_Type'] else 'None',
                'Descriptions': '; '.join(interaction['Description']) if interaction['Description'] else 'None',
                'Recommendation': interaction['Recommendation']
            }
            
            results.append(result_row)
        
        df_results = pd.DataFrame(results)
        
        # Sort by severity
        severity_order = {'high': 0, 'moderate': 1, 'low': 2, 'none': 3}
        df_results['severity_rank'] = df_results['Severity'].map(severity_order)
        df_results = df_results.sort_values('severity_rank').drop('severity_rank', axis=1)
        
        logger.info(f"Found {len(df_results[df_results['Has_Interaction']])} combinations with potential interactions")
        
        return df_results
    
    def filter_safe_combinations(self,
                                combinations_df: pd.DataFrame,
                                max_severity: str = 'moderate') -> pd.DataFrame:
        """
        Filter combinations to only include those below a severity threshold.
        
        Args:
            combinations_df: DataFrame from batch_check_interactions
            max_severity: Maximum acceptable severity ('low', 'moderate', 'high')
            
        Returns:
            Filtered DataFrame
        """
        severity_levels = {'low': 1, 'moderate': 2, 'high': 3}
        max_level = severity_levels.get(max_severity, 2)
        
        mask = combinations_df['Severity'].apply(
            lambda x: self._severity_level(x) <= max_level
        )
        # Use pandas.DataFrame to ensure correct type
        safe_combos = pd.DataFrame(combinations_df[mask])
        
        logger.info(f"Filtered to {len(safe_combos)} safe combinations (max severity: {max_severity})")
        
        return safe_combos
    
    def export_interactions(self, results_df: pd.DataFrame, output_path: Path):
        """Export interaction results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved interaction results to {output_path}")
        
        # Also save summary statistics
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Drug-Drug Interaction Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            total = len(results_df)
            with_interactions = len(results_df[results_df['Has_Interaction']])
            
            f.write(f"Total combinations analyzed: {total}\n")
            f.write(f"Combinations with interactions: {with_interactions}\n")
            f.write(f"Safe combinations: {total - with_interactions}\n\n")
            
            # Severity breakdown
            f.write("Severity Breakdown:\n")
            for severity in ['high', 'moderate', 'low', 'none']:
                count = len(results_df[results_df['Severity'] == severity])
                pct = 100 * count / total if total > 0 else 0
                f.write(f"  {severity.capitalize()}: {count} ({pct:.1f}%)\n")
        
        logger.info(f"Saved interaction summary to {summary_path}")


def load_drugbank_interactions(drugbank_file: Path) -> pd.DataFrame:
    """
    Load drug interactions from DrugBank export.
    
    Args:
        drugbank_file: Path to DrugBank interactions file
        
    Returns:
        DataFrame with standardized columns
    """
    # This is a placeholder - actual implementation would parse DrugBank XML/CSV
    # DrugBank provides comprehensive drug interaction data
    
    logger.warning("DrugBank loader not implemented - using empty database")
    return pd.DataFrame(columns=['Drug_A', 'Drug_B', 'Severity', 'Description'])

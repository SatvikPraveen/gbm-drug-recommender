"""
Pathway Enrichment Analysis Module

Identifies biological pathways affected by drug candidates using Enrichr API.

Features:
- Drug-to-target mapping via multiple databases
- Pathway enrichment across KEGG, Reactome, BioPlanet, GO
- Statistical significance testing (Fisher's exact test)
- GBM-relevant pathway filtering

Databases:
- KEGG 2021 Human - Canonical signaling pathways
- Reactome 2022 - Curated biological processes
- BioPlanet 2019 - Comprehensive pathway collection
- GO Biological Process & Molecular Function 2021
- Custom GBM-relevant pathway set

Outputs:
- pathway_enrichment_summary.csv - All enriched pathways combined
- pathway_enrichment_<database>.csv - Per-database results
- Includes: term, p-value, adjusted p-value, overlapping genes

Usage:
    mapper = DrugTargetMapper()
    targets = mapper.get_all_targets(drug_list)
    analyzer = PathwayAnalyzer()
    enrichment = analyzer.analyze_drug_targets(targets)
"""

import pandas as pd
import requests
import json
from typing import Dict, List, Optional, Tuple
import logging
import time
from tqdm import tqdm

from .config import (
    ENRICHR_URL, ENRICHR_ENRICH_URL, ENRICHR_LIBRARIES,
    PATHWAY_P_VALUE_THRESHOLD, PATHWAY_ADJUSTED_P_VALUE_THRESHOLD,
    PATHWAY_RESULTS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathwayAnalyzer:
    """Perform pathway enrichment analysis using Enrichr"""
    
    def __init__(self):
        """Initialize pathway analyzer"""
        self.enrichr_url = ENRICHR_URL
        self.enrich_url = ENRICHR_ENRICH_URL
        self.libraries = ENRICHR_LIBRARIES
    
    def submit_gene_list(self, genes: List[str], description: str = "Drug Targets") -> Optional[str]:
        """
        Submit gene list to Enrichr
        
        Args:
            genes: List of gene symbols
            description: Description of the gene list
            
        Returns:
            User list ID or None if failed
        """
        genes_str = '\n'.join(genes)
        
        payload = {
            'list': (None, genes_str),
            'description': (None, description)
        }
        
        try:
            response = requests.post(self.enrichr_url, files=payload)
            
            if response.status_code == 200:
                data = response.json()
                user_list_id = data.get('userListId')
                logger.info(f"Gene list submitted successfully. List ID: {user_list_id}")
                return user_list_id
            else:
                logger.error(f"Error submitting gene list: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Exception during gene list submission: {e}")
            return None
    
    def get_enrichment_results(self, user_list_id: str, 
                               library: str = 'KEGG_2021_Human') -> Optional[pd.DataFrame]:
        """
        Get enrichment results from Enrichr for a specific library
        
        Args:
            user_list_id: User list ID from submission
            library: Gene set library to query
            
        Returns:
            DataFrame with enrichment results or None if failed
        """
        query_string = f'?userListId={user_list_id}&backgroundType={library}'
        
        try:
            response = requests.get(self.enrich_url + query_string)
            
            if response.status_code == 200:
                data = response.json()
                
                if library not in data:
                    logger.warning(f"Library '{library}' not found in results")
                    return None
                
                results = data[library]
                
                if not results:
                    logger.info(f"No results for library '{library}'")
                    return None
                
                # Parse results
                df = pd.DataFrame(results, columns=[
                    'Rank', 'Term', 'P-value', 'Z-score', 'Combined Score',
                    'Overlapping Genes', 'Adjusted P-value', 'Old P-value', 'Old Adjusted P-value'
                ])
                
                return df
            else:
                logger.error(f"Error getting enrichment results: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Exception getting enrichment results: {e}")
            return None
    
    def analyze_drug_targets(self, drug_gene_mapping: Dict[str, List[str]],
                            libraries: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Perform pathway enrichment for drug targets
        
        Args:
            drug_gene_mapping: Dictionary mapping drug names to target gene lists
            libraries: List of Enrichr libraries to query (None = use default)
            
        Returns:
            Dictionary mapping library names to enrichment results DataFrames
        """
        if libraries is None:
            libraries = self.libraries
        
        logger.info(f"Analyzing {len(drug_gene_mapping)} drugs across {len(libraries)} libraries")
        
        # Combine all target genes
        all_genes = set()
        for genes in drug_gene_mapping.values():
            all_genes.update(genes)
        
        all_genes = list(all_genes)
        logger.info(f"Total unique genes: {len(all_genes)}")
        
        # Submit gene list
        user_list_id = self.submit_gene_list(all_genes, "GBM Drug Targets")
        
        if user_list_id is None:
            logger.error("Failed to submit gene list")
            return {}
        
        # Get results for each library
        results = {}
        
        for library in tqdm(libraries, desc="Querying Enrichr libraries"):
            time.sleep(0.5)  # Rate limiting
            
            df = self.get_enrichment_results(user_list_id, library)
            
            if df is not None and len(df) > 0:
                # Filter by p-value
                df = df[df['Adjusted P-value'] < PATHWAY_ADJUSTED_P_VALUE_THRESHOLD]
                
                if len(df) > 0:
                    results[library] = df
                    logger.info(f"{library}: Found {len(df)} significant pathways")
        
        return results
    
    def get_drug_pathway_mapping(self, enrichment_results: Dict[str, pd.DataFrame],
                                 top_n: int = 20) -> pd.DataFrame:
        """
        Create a summary of top pathways across all libraries
        
        Args:
            enrichment_results: Dictionary of enrichment results
            top_n: Number of top pathways to include per library
            
        Returns:
            DataFrame with top pathways
        """
        all_pathways = []
        
        for library, df in enrichment_results.items():
            if df is not None and len(df) > 0:
                top_pathways = df.head(top_n).copy()
                top_pathways['Library'] = library
                all_pathways.append(top_pathways)
        
        if not all_pathways:
            logger.warning("No pathway results to summarize")
            return pd.DataFrame()
        
        summary_df = pd.concat(all_pathways, ignore_index=True)
        summary_df = summary_df.sort_values('Adjusted P-value')
        
        return summary_df
    
    def get_gbm_relevant_pathways(self, enrichment_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Filter for GBM-relevant pathways (EGFR, VEGFR, PDGFR, PI3K, mTOR, etc.)
        
        Args:
            enrichment_results: Dictionary of enrichment results
            
        Returns:
            DataFrame with GBM-relevant pathways
        """
        # GBM-relevant keywords
        gbm_keywords = [
            'EGFR', 'VEGF', 'PDGF', 'PI3K', 'AKT', 'mTOR', 'RAS', 'RAF',
            'glioma', 'glioblastoma', 'brain', 'neural', 'astrocyte',
            'RTK', 'receptor tyrosine kinase', 'angiogenesis', 'cell cycle',
            'apoptosis', 'DNA repair', 'p53', 'RB1', 'cell proliferation'
        ]
        
        relevant_pathways = []
        
        for library, df in enrichment_results.items():
            if df is None or len(df) == 0:
                continue
            
            for keyword in gbm_keywords:
                matching = df[df['Term'].str.contains(keyword, case=False, na=False)]
                
                if len(matching) > 0:
                    matching_copy = matching.copy()
                    matching_copy['Library'] = library
                    matching_copy['Keyword'] = keyword
                    relevant_pathways.append(matching_copy)
        
        if not relevant_pathways:
            logger.info("No GBM-relevant pathways found")
            return pd.DataFrame()
        
        relevant_df = pd.concat(relevant_pathways, ignore_index=True)
        relevant_df = relevant_df.drop_duplicates(subset=['Term', 'Library'])
        relevant_df = relevant_df.sort_values('Adjusted P-value')
        
        logger.info(f"Found {len(relevant_df)} GBM-relevant pathways")
        
        return relevant_df
    
    def save_pathway_results(self, enrichment_results: Dict[str, pd.DataFrame],
                            prefix: str = "pathway_enrichment"):
        """
        Save pathway enrichment results to files
        
        Args:
            enrichment_results: Dictionary of enrichment results
            prefix: Filename prefix
        """
        PATHWAY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save individual library results
        for library, df in enrichment_results.items():
            if df is not None and len(df) > 0:
                safe_library_name = library.replace('_', '-').replace('/', '-')
                filepath = PATHWAY_RESULTS_DIR / f"{prefix}_{safe_library_name}.csv"
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {library} results to {filepath}")
        
        # Save summary
        summary_df = self.get_drug_pathway_mapping(enrichment_results)
        if len(summary_df) > 0:
            summary_path = PATHWAY_RESULTS_DIR / f"{prefix}_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Saved pathway summary to {summary_path}")
        
        # Save GBM-relevant pathways
        gbm_df = self.get_gbm_relevant_pathways(enrichment_results)
        if len(gbm_df) > 0:
            gbm_path = PATHWAY_RESULTS_DIR / f"{prefix}_gbm_relevant.csv"
            gbm_df.to_csv(gbm_path, index=False)
            logger.info(f"Saved GBM-relevant pathways to {gbm_path}")


class DrugTargetMapper:
    """Map drugs to their target genes"""
    
    # Pre-defined drug-target mappings for common GBM drugs
    KNOWN_DRUG_TARGETS = {
        'Afatinib': ['EGFR', 'ERBB2', 'ERBB4'],
        'Gefitinib': ['EGFR'],
        'Erlotinib': ['EGFR'],
        'Lapatinib': ['EGFR', 'ERBB2'],
        'Bevacizumab': ['VEGFA'],
        'Sorafenib': ['VEGFR1', 'VEGFR2', 'VEGFR3', 'PDGFRB', 'RAF1', 'BRAF'],
        'Sunitinib': ['VEGFR1', 'VEGFR2', 'PDGFRA', 'PDGFRB', 'KIT', 'FLT3'],
        'Imatinib': ['BCR-ABL1', 'PDGFRA', 'PDGFRB', 'KIT'],
        'Temozolomide': ['MGMT'],
        'Doxorubicin': ['TOP2A', 'TOP2B'],
        'Gemcitabine': ['RRM1', 'RRM2'],
        'Vincristine': ['TUBB', 'TUBB3'],
        'Vinblastine': ['TUBB', 'TUBB3'],
        'Paclitaxel': ['TUBB', 'BCL2'],
        'Cisplatin': ['TP53'],
        'Carboplatin': ['TP53'],
        'Etoposide': ['TOP2A', 'TOP2B'],
        'Topotecan': ['TOP1'],
        'Irinotecan': ['TOP1'],
        'Camptothecin': ['TOP1'],
        'Vorinostat': ['HDAC1', 'HDAC2', 'HDAC3', 'HDAC6'],
        'Panobinostat': ['HDAC1', 'HDAC2', 'HDAC3', 'HDAC6', 'HDAC8'],
        'Trametinib': ['MAP2K1', 'MAP2K2'],
        'Vemurafenib': ['BRAF'],
        'Dabrafenib': ['BRAF'],
        'Everolimus': ['MTOR'],
        'Temsirolimus': ['MTOR'],
        'Rapamycin': ['MTOR'],
        'Ponatinib': ['BCR-ABL1', 'VEGFR2', 'PDGFRA', 'FGFR1', 'FLT3']
    }
    
    def __init__(self):
        """Initialize drug-target mapper"""
        self.drug_targets = self.KNOWN_DRUG_TARGETS.copy()
    
    def add_drug_targets(self, drug: str, targets: List[str]):
        """
        Add custom drug-target mapping
        
        Args:
            drug: Drug name
            targets: List of target gene symbols
        """
        self.drug_targets[drug] = targets
    
    def get_targets(self, drug: str) -> List[str]:
        """
        Get targets for a drug
        
        Args:
            drug: Drug name
            
        Returns:
            List of target genes
        """
        return self.drug_targets.get(drug, [])
    
    def get_all_targets(self, drugs: List[str]) -> Dict[str, List[str]]:
        """
        Get targets for multiple drugs
        
        Args:
            drugs: List of drug names
            
        Returns:
            Dictionary mapping drugs to targets
        """
        result = {}
        for drug in drugs:
            targets = self.get_targets(drug)
            if targets:
                result[drug] = targets
        
        return result
    
    def create_target_matrix(self, drugs: List[str]) -> pd.DataFrame:
        """
        Create a drug-target binary matrix
        
        Args:
            drugs: List of drug names
            
        Returns:
            DataFrame with drugs as rows and targets as columns
        """
        # Get all unique targets
        all_targets = set()
        drug_target_map = {}
        
        for drug in drugs:
            targets = self.get_targets(drug)
            if targets:
                drug_target_map[drug] = targets
                all_targets.update(targets)
        
        all_targets = sorted(list(all_targets))
        
        # Create binary matrix
        matrix_data = []
        
        for drug in drugs:
            row = [1 if target in drug_target_map.get(drug, []) else 0 
                   for target in all_targets]
            matrix_data.append(row)
        
        df = pd.DataFrame(matrix_data, columns=all_targets, index=drugs)
        
        return df


if __name__ == "__main__":
    # Example usage
    mapper = DrugTargetMapper()
    
    test_drugs = ['Afatinib', 'Gefitinib', 'Temozolomide', 'Bevacizumab']
    
    print("Drug-Target Mapping:")
    for drug in test_drugs:
        targets = mapper.get_targets(drug)
        print(f"{drug}: {targets}")
    
    print("\nTarget Matrix:")
    target_matrix = mapper.create_target_matrix(test_drugs)
    print(target_matrix)

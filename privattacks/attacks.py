import csv
import privattacks
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import itertools as it
from typing import Union, List

class Attack():

    def __init__(self, data:privattacks.Data):
        """
        Initialize an instance of the Attack class.

        Parameters:
            - data (privattacks.Data): An instance of the Data class from the privattacks module. 
            This object represents the dataset to be used for analyzing vulnerabilities 
            to probabilistic re-identification and attribute inference attacks.

        Attributes:
            - data (privattacks.Data): Stores the dataset object, providing access to 
            the dataset's attributes, such as columns (`cols`) and number of rows 
            (`n_rows`).
        """
        self.data = data

    def _check_qids(self, qids:list[str]) -> bool:
        """Check if qids are a subset of columns of the dataset.
        
        Raises:
            ValueError: If there is a qid that is not a column of the dataset.
        """
        for att in qids:
            if att not in self.data.cols:
                raise ValueError(f"The quasi-identifier '{att}' is not in the dataset.")
    
    def _check_sensitive(self, sensitive:Union[str, List[str]]):
        """Check if sensitive attributes are a subset of columns of the dataset.
        
        Raises:
            ValueError: If there is a sensitive attribute that is not a column of the dataset.
        """
        if isinstance(sensitive, str):
            sensitive = [sensitive]
            
        for att in sensitive:
            if att not in self.data.cols:
                raise ValueError(f"The sensitive attribute '{att}' is not in the dataset.")
        return sensitive
                
    def _partial_result_reid(self, qids_subset):
        """Required by multiprocessing package in order to use imap_unordered()."""
        return (qids_subset, self.posterior_reid(qids_subset))

    def prior_reid(self):
        """
        Prior vulnerability of probabilistic re-identification attack.

        Returns:
            float: Prior vulnerability.
        """
        return 1 / self.data.n_rows

    def prior_ai(self, sensitive:Union[str, List[str]]):
        """
        Prior vulnerability of probabilistic attribute inference attack.
        
        Parameters:
            - sensitive (str, list): A single or a list of sensitive attributes.

        Returns:
            - dict[str, float]: Dictionary containing the prior vulnerability for each sensitive attribute (keys are sensitive attribute names and values are posterior vulnerabilities).
        """
        self._check_sensitive(sensitive)
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        posteriors = dict()
        for att in sensitive:
            sensitive_idx = self.data.col2int(att)
            _, counts = np.unique(self.data.dataset[:, sensitive_idx], return_counts=True)
            posteriors[att] = int(counts.max()) / self.data.n_rows
        return posteriors

    def posterior_reid(self, qids:list[str]):
        """
        Posterior vulnerability of probabilistic re-identification attack.

        Parameters:
            - qids (list[str]): List of quasi-identifiers.

        Returns:
            - float: Posterior vulnerability.
        """
        self._check_qids(qids)
        qids_idx = [self.data.col2int(att) for att in qids]
        
        # Groupby by qids
        _, partition_starts = np.unique(self.data.dataset[:, qids_idx], axis=0, return_index=True)
        n_partitions = len(partition_starts)
        
        return n_partitions/self.data.n_rows
    
    def posterior_ai(self, qids:list[str], sensitive:Union[str, List[str]]):
        """
        Posterior vulnerability of probabilistic attribute inference attack.
        Obs: It assumes the dataset is sorted by QID columns + sensitive attribute columns.

        Parameters:
            - qids (list): List of quasi-identifiers. If not provided, all columns will be used.
            - sensitive (str, list): A single or a list of sensitive attributes.

        Returns:
            - dict[str, float]: Dictionary containing the posterior vulnerability for each sensitive attribute (keys are sensitive attribute names and values are posterior vulnerabilities).
        """
        self._check_qids(qids)
        self._check_sensitive(sensitive)
        
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        qids_idx = [self.data.col2int(att) for att in qids] # Qid column indices
        qid_values = self.data.dataset[:, qids_idx] # Partition identifiers
        
        # Find unique qid_values, partition starts (indexes) and partition counts
        _, partition_starts = np.unique(qid_values, axis=0, return_index=True)
        n_partitions = len(partition_starts)

        # Attribute inference
        posteriors = {}
        for sens in sensitive:
            sensitive_idx = self.data.col2int(sens) # Sensitive column index
            sensitive_values = self.data.dataset[:, sensitive_idx] # Sensitive attribute columns

            cur_value = -2
            next_partition, cur_count, max_count, posterior = 0, 0, 0, 0
            # Go through all partitions and find the most frequent element
            for i in np.arange(self.data.n_rows):
                # Check if the current partition has finished
                new_partition = (i == partition_starts[next_partition])

                cur_count += 1
                max_count = max(max_count, cur_count)

                # If we've started a new partition or reach the end of the last partition
                if new_partition or i == self.data.n_rows-1:
                    posterior += max_count

                    # Reset pointers
                    next_partition = min(next_partition+1, n_partitions-1)
                    cur_value = sensitive_values[i]
                    max_count, cur_count = 0, 0
                
                # If the current value has changed or if we've started a new partition
                if cur_value != sensitive_values[i] or new_partition:
                    cur_value = sensitive_values[i]
                    cur_count = 0
                
            posteriors[sens] = posterior/self.data.n_rows

        return posteriors        

    def posterior_reid_ai(self, qids:list[str], sensitive:Union[str, List[str]]):
        """
        Posterior vulnerability of probabilistic re-identification and attribute inference attacks.
        Obs: It assumes the dataset is sorted by QID columns + sensitive attribute columsn.

        Parameters:
            - qids (list, optional): List of quasi-identifiers. If not provided, all columns will be used.
            - sensitive (str, list): A single or a list of sensitive attributes.

        Returns:
            - (float, dict[str, float]): Tuple containing, the posterior vulnerability for re-identification attack and a dictionary containing the posterior vulnerability for each sensitive attribute (keys are sensitive attribute names and values are posterior vulnerabilities).
        """
        self._check_qids(qids)
        self._check_sensitive(sensitive)
        
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        qids_idx = [self.data.col2int(att) for att in qids] # Qid column indices
        qid_values = self.data.dataset[:, qids_idx] # Partition identifiers
        
        # Find unique qid_values, partition starts (indexes) and partition counts
        _, partition_starts = np.unique(qid_values, axis=0, return_index=True)
        n_partitions = len(partition_starts)

        # Re-identification
        posterior_reid = n_partitions/self.data.n_rows

        # Attribute Inference
        posteriors_ai = {}
        for sens in sensitive:
            sensitive_idx = self.data.col2int(sens) # Sensitive column index
            sensitive_values = self.data.dataset[:, sensitive_idx] # Sensitive attribute columns

            cur_value = -2
            next_partition, cur_count, max_count, posterior = 0, 0, 0, 0
            # Go through all partitions and find the most frequent element
            for i in np.arange(self.data.n_rows):
                # Check if the current partition has finished
                new_partition = (i == partition_starts[next_partition])

                cur_count += 1
                max_count = max(max_count, cur_count)

                # If we've started a new partition or reach the end of the last partition
                if new_partition or i == self.data.n_rows-1:
                    posterior += max_count

                    # Reset pointers
                    next_partition = min(next_partition+1, n_partitions-1)
                    cur_value = sensitive_values[i]
                    max_count, cur_count = 0, 0
                
                # If the current value has changed or if we've started a new partition
                if cur_value != sensitive_values[i] or new_partition:
                    cur_value = sensitive_values[i]
                    cur_count = 0
                
            posteriors_ai[sens] = posterior/self.data.n_rows

        return posterior_reid, posteriors_ai

    def posterior_reid_subset(self, qids:list[str], num_min, num_max, save_file=None, n_processes=1, verbose=False):
        """Posterior vulnerability of probabilistic re-identification attack for subsets of qids.
        
        Parameters:
            - qids (list[str]): List of quasi-identifiers.
            - min_size (int): Minimum size of subset of qids.
            - max_size (int): Maximum size of subset of qids.
            - save_file (str, optional): File name to save the results. They will be saved in CSV format.
            - n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            - verbose (bool, optinal): Show the progress. Default is False.

        Returns:
            - float: Posterior vulnerability.
        """
        posteriors = []
        if save_file is not None:
            # Create a new file with the header
            posteriors.to_csv(save_file, index=False)
        
        with multiprocessing.Pool(processes=n_processes) as pool:
            # For qid combinations in the given range run re-identification attack
            for n_qids in tqdm(np.arange(num_min, num_max + 1), desc="Number of qids", disable=(not verbose)):
                partial_result = []
                
                # Run the attack for all combination of 'n_qids' QIDs
                results = pool.imap_unordered(
                    self._partial_result_reid,
                    it.combinations(qids, n_qids)
                )

                # Get results from the pool
                for qids, posterior in results:
                    partial_result.append([n_qids, ", ".join(qids), posterior])
                
                # Save once finished all combinations for 'n_qids'
                posteriors.append()

                # Append to save_file
                with open(save_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(partial_result)

                if save_file is not None:
                    partial_result.to_csv(
                        save_file,
                        index=False,
                        mode="a",     
                        header=False,
                        float_format="%.8f"
                    )
# pd.DataFrame(columns=["n_qids", "qids", "posterior_reid"])
        return posteriors
import csv
import privattacks
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import itertools as it
from typing import Union, List, Tuple, Dict

import privattacks.util

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
                
    def _partial_result_reid(self, qids_subset:list[str]):
        """Required by multiprocessing package in order to use imap_unordered()."""
        return (qids_subset, self.posterior_reid(qids_subset))

    def _partial_result_ai(self, params):
        """Required by multiprocessing package in order to use imap_unordered()."""
        qids_subset, sensitive = params
        return (qids_subset, self.posterior_ai(qids_subset, sensitive))
    
    def _partial_result_reid_ai(self, params):
        """Required by multiprocessing package in order to use imap_unordered()."""
        qids_subset, sensitive = params
        return (qids_subset, self.posterior_reid_ai(qids_subset, sensitive))

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
            - sensitive (str, list[str]): A single or a list of sensitive attributes.

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

    def posterior_reid(self, qids:list[str], histogram=False, bin_size=1):
        """
        Posterior vulnerability of probabilistic re-identification attack.

        Parameters:
            - qids (list[str]): List of quasi-identifiers.
            - histogram (bool, optional): Whether to generate a histogram of individual posterior vulnerabilities. Default is False.
            - bin_size (int, optional): Bin size for the histogram if hist is True. Default is 1.

        Returns:
            - float or (float, dict): If histogram is True, returns a pair with the posterior vulnerability and a dictionary containing the histogram of individual posterior vulnerabilities.
            If histogram is False, returns the posterior vulnerability.
        """
        self._check_qids(qids)
        qids_idx = [self.data.col2int(att) for att in qids]
        
        # Groupby by qids
        _, partition_starts = np.unique(self.data.dataset[:, qids_idx], axis=0, return_index=True)
        n_partitions = len(partition_starts)
        posterior = n_partitions/self.data.n_rows
        
        if histogram:
            # Create an array with the posterior vulnerability of each record
            ind_posteriors = []
            partition_starts = np.append(partition_starts, len(partition_starts))
            for i in np.arange(len(partition_starts)-1):
                partition_size = partition_starts[i+1] - partition_starts[i]
                ind_posteriors += [1/partition_size] * partition_size
            
            hist = privattacks.util.create_histogram(ind_posteriors, bin_size)
            return posterior, hist
        
        return posterior
    
    def posterior_ai(self, qids:list[str], sensitive:Union[str, List[str]], histogram=False, bin_size=1):
        """
        Posterior vulnerability of probabilistic attribute inference attack.
        Obs: It assumes the dataset is sorted by QID columns + sensitive attribute columns.

        Parameters:
            - qids (list): List of quasi-identifiers. If not provided, all columns will be used.
            - sensitive (str, list[str]): A single or a list of sensitive attributes.
            - histogram (bool, optional): Whether to generate a histogram of individual posterior vulnerabilities. Default is False.
            - bin_size (int, optional): Bin size for the histogram if hist is True. Default is 1.

        Returns:
            - dict[str, float] or (dict[str, float], dict): If histogram is True, returns a pair with the posterior vulnerability and a dictionary containing the histogram of individual posterior vulnerabilities for each sensitive attribute.
            If histogram is False, returns a dictionary containing the posterior vulnerability for each sensitive attribute.
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
        if histogram:
            # Create an array with the posterior vulnerability of each record
            ind_posteriors = []
            
        posteriors = {}
        for sens in sensitive:
            sensitive_idx = self.data.col2int(sens) # Sensitive column index
            sensitive_values = self.data.dataset[:, sensitive_idx] # Sensitive attribute columns

            cur_value = -2
            next_partition, cur_count, max_count, posterior = 0, 0, 0, 0

            # Go through all partitions and find the most frequent element
            for i in np.arange(self.data.n_rows):
                # Check if the current partition has finished
                new_partition = (next_partition < n_partitions and i == partition_starts[next_partition])

                if new_partition:
                    next_partition += 1
                    
                    # The most frequent element in the previous partition is in max_count
                    max_count = max(max_count, cur_count)
                    posterior += max_count
                    if histogram and next_partition >= 2:
                        partition_size = partition_starts[next_partition-1] - partition_starts[next_partition-2]
                        ind_posteriors += [max_count/partition_size] * partition_size

                    cur_value = sensitive_values[i]
                    cur_count, max_count = 1, 1
                else:
                    if cur_value == sensitive_values[i]:
                        cur_count += 1
                    else:
                        max_count = max(max_count, cur_count)
                        cur_value = sensitive_values[i]
                        cur_count = 1
                
                # If it's the last element, update the max_count and
                # add the adversary's success for the last partition.
                if i == self.data.n_rows-1:
                    max_count = max(max_count, cur_count)
                    posterior += max_count
                    if histogram:
                        # Last partition
                        partition_size = len(partition_starts) - partition_starts[-1]
                        ind_posteriors += [max_count/partition_size] * partition_size

            posteriors[sens] = posterior/self.data.n_rows

        if histogram:
            hist = privattacks.util.create_histogram(ind_posteriors, bin_size)
            return posteriors, hist
        
        return posteriors

    def posterior_reid_ai(self, qids:list[str], sensitive:Union[str, List[str]]):
        """
        Posterior vulnerability of probabilistic re-identification and attribute inference attacks.
        Obs: It assumes the dataset is sorted by QID columns + sensitive attribute columsn.

        Parameters:
            - qids (list, optional): List of quasi-identifiers. If not provided, all columns will be used.
            - sensitive (str, list[str]): A single or a list of sensitive attributes.

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

        # Attribute inference
        posteriors_ai = {}
        for sens in sensitive:
            sensitive_idx = self.data.col2int(sens) # Sensitive column index
            sensitive_values = self.data.dataset[:, sensitive_idx] # Sensitive attribute columns

            cur_value = -2
            next_partition, cur_count, max_count, posterior = 0, 0, 0, 0
            # Go through all partitions and find the most frequent element
            for i in np.arange(self.data.n_rows):
                # Check if the current partition has finished
                new_partition = (next_partition < n_partitions and i == partition_starts[next_partition])

                if new_partition:
                    next_partition += 1
                    
                    # The most frequent element in the previous partition is in max_count
                    max_count = max(max_count, cur_count)
                    posterior += max_count

                    cur_value = sensitive_values[i]
                    cur_count, max_count = 1, 1
                else:
                    if cur_value == sensitive_values[i]:
                        cur_count += 1
                    else:
                        max_count = max(max_count, cur_count)
                        cur_value = sensitive_values[i]
                        cur_count = 1
                
                # If it's the last element, update the max_count and
                # add the adversary's success for the last partition.
                if i == self.data.n_rows-1:
                    max_count = max(max_count, cur_count)
                    posterior += max_count

            posteriors_ai[sens] = posterior/self.data.n_rows

        return posterior_reid, posteriors_ai

    def posterior_reid_subset(self, qids:list[str], num_min, num_max, save_file=None, n_processes=1, verbose=False) -> pd.DataFrame:
        """Posterior vulnerability of probabilistic re-identification attack for subsets of qids. The attack is run for a subset of the powerset of qids, defined by parameters min_size and max_size. 
        
        Parameters:
            - qids (list[str]): List of quasi-identifiers.
            - min_size (int): Minimum size of subset of qids.
            - max_size (int): Maximum size of subset of qids.
            - save_file (str, optional): File name to save the results. They will be saved in CSV format.
            - n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            - verbose (bool, optional): Show the progress. Default is False.

        Returns:
            - (pandas.DataFrame): A pandas DataFrame containing columns "n_qids", "qids" and "posterior_reid", representing the number of qids in the combination, the actual combination and the posterior vulnerability for the given qid combination, respectively.
        """
        self._check_qids(qids)

        if save_file is not None:
            # Create a new file with the header
            with open(save_file, mode="w") as file:
                file.write("n_qids,qids,posterior_reid\n") # Header
        
        float_format = "{:.8f}"  # For 8 decimal places
        posteriors = []
        with multiprocessing.Pool(processes=n_processes) as pool:
            # For qid combinations in the given range run re-identification attack
            for n_qids in tqdm(np.arange(num_min, num_max + 1), desc="Qids combination size", disable=(not verbose)):
                partial_result = []

                # Run the attack for all combination of 'n_qids' QIDs
                results = pool.imap_unordered(
                    self._partial_result_reid,
                    it.combinations(qids, n_qids)
                )

                # Get results from the pool
                for qids_comb, posterior in results:
                    partial_result.append([int(n_qids), ",".join(qids_comb), float_format.format(posterior)])
                
                # Save once finished all combinations for 'n_qids'
                posteriors.extend(partial_result)

                # Append to save_file
                if save_file is not None:
                    with open(save_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(partial_result)
        
        posteriors = pd.DataFrame(posteriors, columns=["n_qids", "qids", "posterior_reid"])
        return posteriors
        
    def posterior_ai_subset(self, qids:list[str], sensitive:Union[str, List[str]], num_min, num_max, save_file=None, n_processes=1, verbose=False) -> pd.DataFrame:
        """Posterior vulnerability of probabilistic attribute inference attack for subsets of qids. The attack is run for a subset of the powerset of qids, defined by parameters min_size and max_size.
        
        Parameters:
            - qids (list[str]): List of quasi-identifiers.
            - sensitive (str, list[str]): A single or a list of sensitive attributes.
            - min_size (int): Minimum size of subset of qids.
            - max_size (int): Maximum size of subset of qids.
            - save_file (str, optional): File name to save the results. They will be saved in CSV format.
            - n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            - verbose (bool, optional): Show the progress. Default is False.

        Returns:
            - (pandas.DataFrame): A pandas DataFrame containing columns "n_qids", "qids" and one column "posterior_S" for every sensitive attribute S, representing, respectively, the number of qids in the combination, the actual combination and the posterior vulnerability for each sensitive attribute.
        """
        self._check_qids(qids)
        self._check_sensitive(sensitive)
        
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        posterior_cols = [f"posterior_{sens}" for sens in sensitive]

        if save_file is not None:
            # Create a new file with the header
            with open(save_file, mode="w") as file:
                file.write(",".join(["n_qids", "qids"] + posterior_cols) + "\n") # Header
        
        float_format = "{:.8f}"  # For 8 decimal places
        posteriors = []
        with multiprocessing.Pool(processes=n_processes) as pool:
            # For qid combinations in the given range run re-identification attack
            for n_qids in tqdm(np.arange(num_min, num_max + 1), desc="Qids combination size", disable=(not verbose)):
                partial_result = []

                # Run the attack for all combination of 'n_qids' QIDs
                results = pool.imap_unordered(
                    self._partial_result_ai,
                    ((comb,sensitive) for comb in it.combinations(qids, n_qids))
                )

                # Get results from the pool
                for qids_comb, posterior in results:
                    posteriors_partial = [float_format.format(posterior[sens]) for sens in sensitive]
                    partial_result.append([int(n_qids), ",".join(qids_comb)] + posteriors_partial)
                
                # Save once finished all combinations for 'n_qids'
                posteriors.extend(partial_result)
                
                # Append to save_file
                if save_file is not None:
                    float_format = "{:.8f}"  # For 8 decimal places
                    with open(save_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(partial_result)
        
        posteriors = pd.DataFrame(posteriors, columns=["n_qids", "qids"] + posterior_cols)
        return posteriors
    
    def posterior_reid_ai_subset(self, qids:list[str], sensitive:Union[str, List[str]], num_min, num_max, save_file=None, n_processes=1, verbose=False) -> pd.DataFrame:
        """Posterior vulnerability of probabilistic re-identification and attribute inference attack for subsets of qids. The attack is run for a subset of the powerset of qids, defined by parameters min_size and max_size.
        
        Parameters:
            - qids (list[str]): List of quasi-identifiers.
            - sensitive (str, list[str]): A single or a list of sensitive attributes.
            - min_size (int): Minimum size of subset of qids.
            - max_size (int): Maximum size of subset of qids.
            - save_file (str, optional): File name to save the results. They will be saved in CSV format.
            - n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            - verbose (bool, optional): Show the progress. Default is False.

        Returns:
            - (pandas.DataFrame): A pandas DataFrame containing columns "n_qids", "qids", "posterior_reid", and one column "posterior_S" for every sensitive attribute S, representing, respectively, the number of qids in the combination, the actual combination, the posterior vulnerability for re-identification and the posterior vulnerability for each sensitive attribute.
        """
        self._check_qids(qids)
        self._check_sensitive(sensitive)
        
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        posterior_cols = [f"posterior_{sens}" for sens in sensitive]

        if save_file is not None:
            # Create a new file with the header
            with open(save_file, mode="w") as file:
                file.write(",".join(["n_qids", "qids", "posterior_reid"] + posterior_cols) + "\n") # Header
        
        float_format = "{:.8f}"  # For 8 decimal places
        posteriors = []
        with multiprocessing.Pool(processes=n_processes) as pool:
            # For qid combinations in the given range run re-identification attack
            for n_qids in tqdm(np.arange(num_min, num_max + 1), desc="Qids combination size", disable=(not verbose)):
                partial_result = []

                # Run the attack for all combination of 'n_qids' QIDs
                results = pool.imap_unordered(
                    self._partial_result_reid_ai,
                    ((comb,sensitive) for comb in it.combinations(qids, n_qids))
                )

                # Get results from the pool
                for qids_comb, posterior in results:
                    posterior_reid, posterior_ai = posterior
                    posterior_partial_reid = float_format.format(posterior_reid)
                    posteriors_partial_ai = [float_format.format(posterior_ai[sens]) for sens in sensitive]
                    partial_result.append(
                        [int(n_qids), ",".join(qids_comb)] + [posterior_partial_reid] + posteriors_partial_ai
                    )
                
                # Save once finished all combinations for 'n_qids'
                posteriors.extend(partial_result)
                
                # Append to save_file
                if save_file is not None:
                    float_format = "{:.8f}"  # For 8 decimal places
                    with open(save_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(partial_result)
        
        posteriors = pd.DataFrame(posteriors, columns=["n_qids", "qids", "posterior_reid"] + posterior_cols)
        return posteriors
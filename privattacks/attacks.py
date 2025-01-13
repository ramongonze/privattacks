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
            data (privattacks.Data): An instance of the Data class from the privattacks module. 
            This object represents the dataset to be used for analyzing vulnerabilities 
            to probabilistic re-identification and attribute inference attacks.

        Attributes:
            data (privattacks.Data): Stores the dataset object, providing access to 
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
                
    def _sort_dataset(self, qids:list[str]):
        """Sort dataset by qid values. It's an assumption for attack methods in class Attack.
        """
        qids_idx = [self.data.cols.index(qid) for qid in qids][::-1]
        
        # Sort in ascending order (lexicographical sort) 
        # The order must be reversed to use numpy.lexsort (order of priority)
        keys = tuple(self.data.dataset[:, i] for i in qids_idx)
        sorted_indices = np.lexsort(keys)

        # Use the indices to sort the array
        self.data.dataset = self.data.dataset[sorted_indices]

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
            sensitive (str, list[str]): A single or a list of sensitive attributes.

        Returns:
            dict[str, float]: Dictionary containing the prior vulnerability for each sensitive attribute (keys are sensitive attribute names and values are posterior vulnerabilities).
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
            qids (list[str]): List of quasi-identifiers.
            histogram (bool, optional): Whether to generate a histogram of individual posterior vulnerabilities. Default is False.
            bin_size (int, optional): Bin size for the histogram if hist is True. Default is 1.

        Returns:
            float or (float, dict): If histogram is False, returns the posterior vulnerability.
                If histogram is True, returns a pair (<posterior vulnerability>, <histogram>).
                Example of output when histogram is False::
                    
                    0.75

                Example of output when histogram is True::
                
                    (0.75, {'[0,0.50)':7, '[0.50,1]':13})
        """
        self._check_qids(qids)
        self._sort_dataset(qids)

        qids_idx = [self.data.col2int(att) for att in qids]
        
        # Groupby by qids
        _, partition_starts = np.unique(self.data.dataset[:, qids_idx], axis=0, return_index=True)
        partition_starts.sort()
        n_partitions = len(partition_starts)
        posterior = n_partitions/self.data.n_rows
        
        if histogram:
            # Create an array with the posterior vulnerability of each record
            ind_posteriors = []
            partition_starts = np.append(partition_starts, self.data.n_rows)
            for i in np.arange(len(partition_starts)-1):
                partition_size = partition_starts[i+1] - partition_starts[i]
                ind_posteriors += [1/partition_size] * partition_size

            hist = privattacks.util.create_histogram(ind_posteriors, bin_size)
            return posterior, hist
        
        return posterior
    
    def posterior_ai(self, qids:list[str], sensitive:Union[str, List[str]], histogram=False, bin_size=1):
        """
        Posterior vulnerability of probabilistic attribute inference attack.

        Parameters:
            qids (list): List of quasi-identifiers. If not provided, all columns will be used.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            histogram (bool, optional): Whether to generate a histogram of individual posterior vulnerabilities. Default is False.
            bin_size (int, optional): Bin size for the histogram if hist is True. Default is 1.

        Returns:
            dict[str, float] or (dict[str, float], dict):
                If histogram is False, returns a dictionary containing the posterior vulnerability for each sensitive attribute.
                If histogram is True, returns a pair ``(<posterior vulnerability>, <histogram for each sensitive attribute>)``.
                Example of output when histogram is False::
                
                    {'disease': 0.3455, 'income':0.7}

                Example of ouput when histogram is True::
                
                    ({'disease': 0.3455, 'income':0.7},
                     {'disease': {'[0,0.50)':15, '[0.50,1]':5},
                      'income':{'[0,0.50)':12, '[0.50,1]':8}})
        """
        self._check_qids(qids)
        self._check_sensitive(sensitive)
        self._sort_dataset(qids)
        
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        qids_idx = [self.data.col2int(att) for att in qids] # Qid column indices
        qid_values = self.data.dataset[:, qids_idx] # Partition identifiers
        

        # Find unique qid_values, partition starts (indexes) and partition counts
        _, partition_starts = np.unique(qid_values, axis=0, return_index=True)
        partition_starts.sort()
        n_partitions = len(partition_starts)

        # Attribute inference
        if histogram:
            # Create an array with the posterior vulnerability of each record
            ind_posteriors = {sens:[] for sens in sensitive}
        
        posteriors = {}
        for sens in sensitive:
            sensitive_idx = self.data.col2int(sens) # Sensitive column index
            sensitive_values = self.data.dataset[:, sensitive_idx] # Sensitive attribute columns

            posterior = 0
            for i in np.arange(len(partition_starts)):
                start = partition_starts[i]

                # Get the index the partition ends
                if start == partition_starts[-1]:
                    end = self.data.n_rows-1
                else:
                    end = partition_starts[i+1]-1

                # Count the number of times each sensitive value appears in the current partition
                values, counts = np.unique(sensitive_values[start:end+1], return_counts=True)
                max_freq = counts.max()
                posterior += max_freq # Number of times the most frequent element appears

                if histogram:
                    partition_size = end-start+1
                    ind_posteriors[sens] += [max_freq/partition_size] * partition_size

            posteriors[sens] = posterior/self.data.n_rows

        if histogram:
            hist = {sens:privattacks.util.create_histogram(ind_posteriors[sens], bin_size) for sens in sensitive}
            return posteriors, hist
        
        return posteriors

    def posterior_reid_ai(self, qids:list[str], sensitive:Union[str, List[str]], histogram=False, bin_size=1):  
        """
        Posterior vulnerability of probabilistic re-identification and attribute inference attacks.

        Parameters:
            qids (list, optional): List of quasi-identifiers. If not provided, all columns will be used.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            histogram (bool, optional): Whether to generate a histogram of individual posterior vulnerabilities. Default is False.
            bin_size (int, optional): Bin size for the histogram if hist is True. Default is 1.

        Returns:
            (float, dict[str, float]) or ((float, dict), (dict[str, float], dict[str, dict[str, int]])):
            If histogram is False, returns a pair ``(<posterior re-identification>, <posterior attribute inference for each sensitive attribute (dictionary)>)``. If histogram is True, returns a pair containing the results for re-identification and attribute inference. The re-identification results is a pair ``(<posterior vulnerability>, <histogram>)`` and attribute inference results is a pair ``(<posterior vulnerability>, <histogram for each sensitive attribute>)``.
            Example of output when histogram is False::
            
                (0.75, {'disease': 0.3455, 'income':0.7})

            Example of ouput when histogram is True::
            
                ((0.75, {'[0,0.50)':7, '[0.50,1]':13}),
                 ({'disease': 0.3455, 'income':0.7},
                  {'disease': {'[0,0.50)':15, '[0.50,1]':5}, 'income':{'[0,0.50)':12, '[0.50,1]':8}}))
        """
        self._check_qids(qids)
        self._check_sensitive(sensitive)
        self._sort_dataset(qids)
        
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        qids_idx = [self.data.col2int(att) for att in qids] # Qid column indices
        qid_values = self.data.dataset[:, qids_idx] # Partition identifiers
        
        # Find unique qid_values, partition starts (indexes) and partition counts
        _, partition_starts = np.unique(qid_values, axis=0, return_index=True)
        partition_starts.sort()
        n_partitions = len(partition_starts)

        # Re-identification
        posterior_reid = n_partitions/self.data.n_rows

        if histogram:
            # Create an array with the posterior vulnerability of each record
            ind_posteriors = []
            partition_starts = np.append(partition_starts, self.data.n_rows)
            for i in np.arange(len(partition_starts)-1):
                partition_size = partition_starts[i+1] - partition_starts[i]
                ind_posteriors += [1/partition_size] * partition_size
            
            hist_reid = privattacks.util.create_histogram(ind_posteriors, bin_size)

            # Reset the array for attribute inference's histogram
            partition_starts = partition_starts[:-1]
            ind_posteriors = {sens:[] for sens in sensitive}

        # Attribute inference
        posteriors_ai = {}
        for sens in sensitive:
            sensitive_idx = self.data.col2int(sens) # Sensitive column index
            sensitive_values = self.data.dataset[:, sensitive_idx] # Sensitive attribute columns

            posterior = 0
            for i in np.arange(len(partition_starts)):
                start = partition_starts[i]

                # Get the index the partition ends
                if start == partition_starts[-1]:
                    end = self.data.n_rows-1
                else:
                    end = partition_starts[i+1]-1

                # Count the number of times each sensitive value appears in the current partition
                _, counts = np.unique(sensitive_values[start:end+1], return_counts=True)
                max_freq = counts.max()
                posterior += max_freq # Number of times the most frequent element appears

                if histogram:
                    partition_size = end-start+1
                    ind_posteriors[sens] += [max_freq/partition_size] * partition_size

            posteriors_ai[sens] = posterior/self.data.n_rows

        if histogram:
            hist_ai = {sens:privattacks.util.create_histogram(ind_posteriors[sens], bin_size) for sens in sensitive}
            return (posterior_reid, hist_reid), (posteriors_ai, hist_ai)
        
        return posterior_reid, posteriors_ai

    def posterior_reid_subset(self, qids:list[str], num_min, num_max, save_file=None, n_processes=1, verbose=False) -> pd.DataFrame:
        """Posterior vulnerability of probabilistic re-identification attack for subsets of qids. The attack is run for a subset of the powerset of qids, defined by parameters min_size and max_size. 
        
        Parameters:
            qids (list[str]): List of quasi-identifiers.
            min_size (int): Minimum size of subset of qids.
            max_size (int): Maximum size of subset of qids.
            save_file (str, optional): File name to save the results. They will be saved in CSV format.
            n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            verbose (bool, optional): Show the progress. Default is False.

        Returns:
            (pandas.DataFrame): A pandas DataFrame containing columns "n_qids", "qids" and "posterior_reid", representing the number of qids in the combination, the actual combination and the posterior vulnerability for the given qid combination, respectively.
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
            qids (list[str]): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            min_size (int): Minimum size of subset of qids.
            max_size (int): Maximum size of subset of qids.
            save_file (str, optional): File name to save the results. They will be saved in CSV format.
            n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            verbose (bool, optional): Show the progress. Default is False.

        Returns:
            (pandas.DataFrame): A pandas DataFrame containing columns "n_qids", "qids" and one column "posterior_S" for every sensitive attribute S, representing, respectively, the number of qids in the combination, the actual combination and the posterior vulnerability for each sensitive attribute.
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
            qids (list[str]): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            min_size (int): Minimum size of subset of qids.
            max_size (int): Maximum size of subset of qids.
            save_file (str, optional): File name to save the results. They will be saved in CSV format.
            n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            verbose (bool, optional): Show the progress. Default is False.

        Returns:
            (pandas.DataFrame): A pandas DataFrame containing columns "n_qids", "qids", "posterior_reid", and one column "posterior_S" for every sensitive attribute S, representing, respectively, the number of qids in the combination, the actual combination, the posterior vulnerability for re-identification and the posterior vulnerability for each sensitive attribute.
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
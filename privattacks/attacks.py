import csv
import math
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

    def _check_cols(self, cols:list[str]) -> bool:
        """Check if columns are a subset of columns of the dataset.
        
        Raises:
            ValueError: If there is a qid that is not a column of the dataset.
        """
        for col in cols:
            if col not in self.data.cols:
                raise ValueError(f"Column '{col}' is not in the dataset.")
                
    def _sort_dataset(self, cols:list[str]):
        """Sort dataset by a given set of columns. Returns a sorted copy of the dataset (only the given columns)."""
        cols_idx = [self.data.col2int(col) for col in cols]
        
        # Sort in ascending order (lexicographical sort) 
        # The order must be reversed to use numpy.lexsort (order of priority)
        keys = tuple(self.data.dataset[:, i] for i in cols_idx[::-1])
        sorted_indices = np.lexsort(keys)

        # Use the indices to sort the array
        return self.data.dataset[sorted_indices][:,cols_idx].copy()

    def _partial_result_reid(self, qids_subset:list[str]):
        """Required by multiprocessing package in order to use imap_unordered()."""
        return (list(qids_subset), self.posterior_reid(list(qids_subset)))

    def _partial_result_ai(self, params):
        """Required by multiprocessing package in order to use imap_unordered()."""
        qids_subset, sensitive = params
        return (list(qids_subset), self.posterior_ai(list(qids_subset), sensitive))
    
    def _partial_result_reid_ai(self, params):
        """Required by multiprocessing package in order to use imap_unordered()."""
        qids_subset, sensitive = params
        return (list(qids_subset), self.posterior_reid_ai(list(qids_subset), sensitive))

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
        if isinstance(sensitive, str):
            sensitive = [sensitive]
        
        self._check_cols(sensitive)

        priors = dict()
        for sens in sensitive:
            priors[sens] = 1/len(self.data.domains[sens])
        return priors

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
        self._check_cols(qids)
        cols = qids
        dataset = self._sort_dataset(qids).copy()
        qids_idx = [cols.index(qid) for qid in qids]
        
        # Groupby by qids
        _, partition_starts = np.unique(dataset[:, qids_idx], axis=0, return_index=True)
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
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(qids + sensitive)

        cols = qids + sensitive
        dataset = self._sort_dataset(cols)
        
        qids_idx = [cols.index(qid) for qid in qids] # Qid column indices
        qid_values = dataset[:, qids_idx] # Partition identifiers
        
        # Find unique qid_values, partition starts (indexes) and partition counts
        _, partition_starts = np.unique(qid_values, axis=0, return_index=True)
        partition_starts.sort()

        # Attribute inference
        if histogram:
            # Create an array with the posterior vulnerability of each record
            ind_posteriors = {sens:[] for sens in sensitive}
        
        posteriors = {}
        for sens in sensitive:
            sensitive_idx = cols.index(sens) # Sensitive column index
            sensitive_values = dataset[:, sensitive_idx] # Sensitive attribute columns

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
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(qids + sensitive)
        
        cols = qids + sensitive
        dataset = self._sort_dataset(cols)
        qids_idx = [cols.index(qid) for qid in qids] # Qid column indices

        qid_values = dataset[:, qids_idx] # Partition identifiers

        # Find unique , partition starts (indexes) and partition counts
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
            sensitive_idx = cols.index(sens) # Sensitive column index
            sensitive_values = dataset[:, sensitive_idx] # Sensitive attribute columns

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
        self._check_cols(qids)

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
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(qids + sensitive)

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
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(qids + sensitive)

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
    
    def posterior_reid_krr_individual(self, qids:list[str], data_san:privattacks.Data, epsilons:dict[str,float]):
        """Posterior vulnerability of re-identification in a dataset sanitized by k-RR individually on each column.
        The dataset used in the constructor will be considered the original dataset.
        
        Parameters:
            qids (list[str]): List of quasi-identifiers.
            data_san (privattacks.Data): Sanitized version of the dataset.
            epsilons (dict[str, float]): Privacy parameter for each column.
            domain_sizes (dict[str, int]): Column domain sizes.

        Returns:
            float: Posterior re-identification vulnerability.
        """
        self._check_cols(qids)

        # Transform into numpy arrays
        domain_sizes = np.array([len(self.data.domains[qid]) for qid in qids])
        epsilons = np.array([epsilons[qid] for qid in qids])
        qid_idxs = [self.data.col2int(qid) for qid in qids]

        dataset_ori = self.data.dataset[:, qid_idxs]
        dataset_san = data_san.dataset[:, qid_idxs]

        # p = probability to keep the original value
        p = math.e**epsilons / (math.e**epsilons + domain_sizes - 1)
        p_any_other = (1 - p) / (domain_sizes - 1)

        # For a given target, calculates the probability of each record in the 
        # sanitized dataset to be the sanitized version of the target
        prob = lambda target, dataset_san: np.sum(p * (dataset_san == target) + p_any_other * (dataset_san != target), axis=-1)

        isclose = np.vectorize(math.isclose)
        posterior = 0
        for idx_target, target in enumerate(dataset_ori):
            probs = prob(target, dataset_san)     
            max_prob = np.max(probs)
            
            # Check if the target is in the list of candidates
            if math.isclose(max_prob, probs[idx_target]):
                n_candidates = sum(isclose(probs, max_prob))
                posterior += 1/n_candidates
        
        # Divide by the prior probability of each target
        posterior /= self.data.n_rows

        return posterior
    
    def posterior_ai_krr_individual(self, qids:list[str], sensitive:Union[str, List[str]], data_san:privattacks.Data, epsilons:dict[str,float]):
        """Posterior vulnerability of attribute inference in a dataset sanitized by k-RR individually on each column.
        The dataset used in the constructor will be considered the original dataset.
        
        Parameters:
            qids (list[str]): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            data_san (privattacks.Data): Sanitized version of the dataset.
            epsilons (dict[str, float]): Privacy parameter for each column.
            domain_sizes (dict[str, int]): Column domain sizes.

        Returns:
            dict[str, float]: Dictionary containing the posterior vulnerability for each sensitive attribute.
        """
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(qids + sensitive)

        # Transform into numpy arrays
        domain_sizes = np.array([len(self.data.domains[qid]) for qid in qids])
        epsilons = np.array([epsilons[qid] for qid in qids])
        qid_idxs = [self.data.col2int(qid) for qid in qids]

        dataset_ori = self.data.dataset[:, qid_idxs]
        dataset_san = data_san.dataset[:, qid_idxs]

        # p = probability to keep the original value
        p = math.e**epsilons / (math.e**epsilons + domain_sizes - 1)
        p_any_other = (1 - p) / (domain_sizes - 1)

        # For a given target, calculates the probability of each record in the 
        # sanitized dataset to be the sanitized version of the target
        prob = lambda target, dataset_san: np.sum(p * (dataset_san == target) + p_any_other * (dataset_san != target), axis=-1)

        isclose = np.vectorize(math.isclose)
        posteriors = {sens:0 for sens in sensitive}
        for idx_target, target in enumerate(dataset_ori):
            probs = prob(target, dataset_san)     
            max_prob = np.max(probs)
            candidates = isclose(probs, max_prob)

            for sens in sensitive:
                sens_idx = self.data.col2int(sens)
                cand_sensitive = data_san.dataset[candidates, sens_idx] # Sensitive values of candidates

                # Get the most frequent elements
                values, counts = np.unique(cand_sensitive, return_counts=True)
                max_freq = np.max(counts)
                max_values = values[counts == max_freq]

                target_ori_value = self.data.dataset[idx_target, sens_idx]
                # Check if the target's original value is in the list of most frequent sensitive values
                if target_ori_value in max_values:
                    posteriors[sens] += 1/len(max_values)

        # Divide by the prior probability of each target
        for sens in sensitive:
            posteriors[sens] /= self.data.n_rows

        return posteriors
    
    def posterior_reid_ai_krr_individual(self, qids:list[str], sensitive:Union[str, List[str]], data_san:privattacks.Data, epsilons:dict[str,float]):
        """Posterior vulnerability of re-identification and attribute inference in a dataset sanitized by k-RR individually on each column.
        The dataset used in the constructor will be considered the original dataset.
        
        Parameters:
            qids (list[str]): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            data_san (privattacks.Data): Sanitized version of the dataset.
            epsilons (dict[str, float]): Privacy parameter for each column.
            domain_sizes (dict[str, int]): Column domain sizes.

        Returns:
            (float, dict[str, float]): A pair where the first element is the posterior vulnerability of re-identificadtion and the second is a dictionary containing the posterior vulnerability for each sensitive attribute.
        """
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(qids + sensitive)

        # Transform into numpy arrays
        domain_sizes = np.array([len(self.data.domains[qid]) for qid in qids])
        epsilons = np.array([epsilons[qid] for qid in qids])
        qid_idxs = [self.data.col2int(qid) for qid in qids]

        dataset_ori = self.data.dataset[:, qid_idxs]
        dataset_san = data_san.dataset[:, qid_idxs]

        # p = probability to keep the original value
        p = math.e**epsilons / (math.e**epsilons + domain_sizes - 1)
        p_any_other = (1 - p) / (domain_sizes - 1)
        
        # For a given target, calculates the probability of each record in the 
        # sanitized dataset to be the sanitized version of the target
        prob = lambda target, dataset_san: np.sum(p * (dataset_san == target) + p_any_other * (dataset_san != target), axis=-1)

        isclose = np.vectorize(math.isclose)
        posterior_reid = 0
        posteriors_ai = {sens:0 for sens in sensitive}
        for idx_target, target in enumerate(dataset_ori):
            probs = prob(target, dataset_san)     
            max_prob = np.max(probs)
            candidates = isclose(probs, max_prob)

            # Re-identification
            # Check if the target is in the list of candidates
            if math.isclose(max_prob, probs[idx_target]):
                posterior_reid += 1/sum(candidates)
            
            # Attribute inference
            for sens in sensitive:
                sens_idx = self.data.col2int(sens)
                cand_sensitive = data_san.dataset[candidates, sens_idx] # Sensitive values of candidates

                # Get the most frequent elements
                values, counts = np.unique(cand_sensitive, return_counts=True)
                max_freq = np.max(counts)
                max_values = values[counts == max_freq]

                target_ori_value = self.data.dataset[idx_target, sens_idx]
                # Check if the target's original value is in the list of most frequent sensitive values
                if target_ori_value in max_values:
                    posteriors_ai[sens] += 1/len(max_values)

        # Divide by the prior probability of each target
        posterior_reid /= self.data.n_rows
        for sens in sensitive:
            posteriors_ai[sens] /= self.data.n_rows

        return posterior_reid, posteriors_ai

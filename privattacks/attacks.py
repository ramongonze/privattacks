import os
import csv
import math
import privattacks
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import itertools as it
from typing import Union, List
import zipfile
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
    
    def _partial_result_ai_record(self, params):
        """Required by multiprocessing package in order to use imap_unordered().
        It also returns the posterior vulnerability per record."""
        qids_subset, sensitive = params
        return (list(qids_subset), self.posterior_ai(list(qids_subset), sensitive, distribution=True))

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

    def posterior_reid(
            self,
            qids:list[str],
            distribution=False
        ):
        """
        Posterior vulnerability of probabilistic re-identification attack.

        Parameters:
            qids (list[str]): List of quasi-identifiers.
            distribution (bool, optional): Whether to return the distribution of posterior vulnerability per record. Default is False.

        Returns:
            float or (float, list): If distribution is False, returns the posterior vulnerability.
                If distribution is True, returns a pair (<posterior vulnerability>, <distribution>).
                Example of output when distribution is False::
                    
                    0.75

                Example of output when distribution is True::
                
                    (0.75, [0.5, 0.5, 1.0, 1.0, 0.75])
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
        
        if distribution:
            # Create an array with the posterior vulnerability of each record
            posteriors_record = []
            partition_starts = np.append(partition_starts, self.data.n_rows)
            for i in np.arange(len(partition_starts)-1):
                partition_size = partition_starts[i+1] - partition_starts[i]
                posteriors_record += [1/partition_size] * partition_size

            return posterior, np.array(posteriors_record)
        
        return posterior
    
    def posterior_ai(
            self,
            qids:list[str],
            sensitive:Union[str,List[str]],
            distribution=False
        ):
        """
        Posterior vulnerability of probabilistic attribute inference attack.

        Parameters:
            qids (list): List of quasi-identifiers. If not provided, all columns will be used.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            distribution (bool, optional): Whether to return the distribution of posterior vulnerability per record. Default is False.

        Returns:
            dict[str, float] or (dict[str, list]):
                If distribution is False, returns a dictionary containing the posterior vulnerability for each sensitive attribute.
                If distribution is True, returns a pair ``(<posterior vulnerability>, <distribution for each sensitive attribute>)``.
                Example of output when distribution is False::
                
                    {'disease': 0.3455, 'income':0.7}

                Example of ouput when distribution is True::
                
                    ({'disease': 0.3455, 'income':0.7},
                     {'disease': [0.1, 0.1, 0.3, 0.4, 0.8275],
                      'income': [0.6, 0.7, 0.7, 0.7, 0.8]})
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
        if distribution:
            # Create an array with the posterior vulnerability of each record
            posteriors_record = {sens:[] for sens in sensitive}
        
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

                if distribution:
                    partition_size = end-start+1
                    posteriors_record[sens] += [max_freq/partition_size] * partition_size

            posteriors[sens] = posterior/self.data.n_rows

        if distribution:
            return posteriors, {sens:np.array(posteriors_record[sens]) for sens in sensitive}
        
        return posteriors

    def posterior_reid_ai(
            self,
            qids:list[str],
            sensitive:Union[str,List[str]],
            distribution=False
        ):  
        """
        Posterior vulnerability of probabilistic re-identification and attribute inference attacks.

        Parameters:
            qids (list, optional): List of quasi-identifiers. If not provided, all columns will be used.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            distribution (bool, optional): Whether to return the distribution of posterior vulnerability per record. Default is False.

        Returns:
            (float, dict[str, float]) or ((float, list), (dict[str, float], dict[str, list])):
            If distribution is False, returns a pair ``(<posterior re-identification>, <posterior attribute inference for each sensitive attribute (dictionary)>)``. If distribution is True, returns a pair containing the results for re-identification and attribute inference. The re-identification results is a pair ``(<posterior vulnerability>, <distribution>)`` and attribute inference results is a pair ``(<posterior vulnerability>, <distribution for each sensitive attribute>)``.
            Example of output when distribution is False::
            
                (0.75, {'disease': 0.3455, 'income':0.7})

            Example of ouput when distribution is True::
            
                ((0.75, [0.5, 0.5, 1.0, 1.0, 0.75]),
                 ({'disease': 0.3455, 'income':0.7},
                  {'disease': [0.1, 0.1, 0.3, 0.4, 0.8275],
                   'income': [0.6, 0.7, 0.7, 0.7, 0.8]}))
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

        if distribution:
            # Create an array with the posterior vulnerability of each record
            posteriors_reid_record = []
            partition_starts = np.append(partition_starts, self.data.n_rows)
            for i in np.arange(len(partition_starts)-1):
                partition_size = partition_starts[i+1] - partition_starts[i]
                posteriors_reid_record += [1/partition_size] * partition_size

            # Reset the array for attribute inference's distribution
            partition_starts = partition_starts[:-1]
            posteriors_ai_record = {sens:[] for sens in sensitive}

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

                if distribution:
                    partition_size = end-start+1
                    posteriors_ai_record[sens] += [max_freq/partition_size] * partition_size

            posteriors_ai[sens] = posterior/self.data.n_rows

        if distribution:
            return (posterior_reid, np.array(posteriors_reid_record)), (posteriors_ai, {sens:np.array(posteriors_ai_record[sens]) for sens in sensitive})
        
        return posterior_reid, posteriors_ai

    def posterior_reid_subset(
            self,
            qids:list[str],
            num_min,
            num_max,
            save_file=None,
            n_processes=1,
            verbose=False
        ) -> pd.DataFrame:
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

        if save_file:
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
                if save_file:
                    with open(save_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(partial_result)
        
        posteriors = pd.DataFrame(posteriors, columns=["n_qids", "qids", "posterior_reid"])
        return posteriors
        
    def posterior_ai_subset(
            self,
            qids:list[str],
            sensitive:Union[str,List[str]],
            num_min,
            num_max,
            save_file=None,
            zip_save=False,
            n_processes=1,
            distribution=False,
            return_results=True,
            verbose=False
        ):
        """Posterior vulnerability of probabilistic attribute inference attack for subsets of qids. The attack is run for a subset of the powerset of qids, defined by parameters min_size and max_size.
        
        Parameters:
            qids (list[str]): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            min_size (int): Minimum size of subset of qids.
            max_size (int): Maximum size of subset of qids.
            save_file (str, optional): File name to save the results. They will be saved in CSV format.
            zip_save (bool, optional): Save the results in a zip file insteade of csv. Default is False.
            n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            distribution (bool, optional): Whether to return the distribution of posterior vulnerability per record. Default is False.
            return_results (bool, optional): Whether to return the results or not. Default is True.
            verbose (bool, optional): Show the progress. Default is False.

        Returns:
            (pandas.DataFrame): A pandas DataFrame containing columns "n_qids", "qids" and one column "posterior_S" for every sensitive attribute S, representing, respectively, the number of qids in the combination, the actual combination and the posterior vulnerability for each sensitive attribute.
        """
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(qids + sensitive)

        posterior_cols = [f"posterior_{sens}" for sens in sensitive]
        if distribution:
            posterior_cols += [f"posterior_{sens}_record" for sens in sensitive]
            partial_method = self._partial_result_ai_record
        else:
            partial_method = self._partial_result_ai

        if save_file:
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
                    partial_method,
                    ((comb,sensitive) for comb in it.combinations(qids, n_qids))
                )

                # Get results from the pool
                for qids_comb, posterior in results:
                    if distribution:    
                        posterior_vul, posterior_vul_record = posterior
                        posterior_vul_record = [[float_format.format(p) for p in posterior_vul_record[sens]] for sens in sensitive]
                    else:
                        posterior_vul = posterior
                        posterior_vul_record = []
                    
                    posteriors_partial = [float_format.format(posterior_vul[sens]) for sens in sensitive]
                    
                    if save_file:
                        # Append to save_file
                        with open(save_file, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerows([[int(n_qids), ",".join(qids_comb)] + posteriors_partial + posterior_vul_record])
                            
                    if return_results:
                        partial_result.append([int(n_qids), ",".join(qids_comb)] + posteriors_partial + posterior_vul_record)
                
                if return_results:
                    # Save once finished all combinations for 'n_qids'
                    posteriors.extend(partial_result)
        
        if zip_save:
            # Create zip and add the csv inside it
            zip_path = save_file.replace(".csv", ".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(save_file, arcname=file)
            
            # Remove csv file
            os.remove(save_file)

        if return_results:
            posteriors = pd.DataFrame(posteriors, columns=["n_qids", "qids"] + posterior_cols)
            return posteriors
    
    def posterior_reid_ai_subset(
            self,
            qids:list[str],
            sensitive:Union[str,List[str]],
            num_min,
            num_max,
            save_file=None,
            n_processes=1,
            verbose=False
        ) -> pd.DataFrame:
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

        if save_file:
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
                if save_file:
                    float_format = "{:.8f}"  # For 8 decimal places
                    with open(save_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(partial_result)
        
        posteriors = pd.DataFrame(posteriors, columns=["n_qids", "qids", "posterior_reid"] + posterior_cols)
        return posteriors
    
    def posterior_reid_krr_individual(
            self,
            qids:list[str],
            data_san:privattacks.Data,
            epsilons:dict[str,float]
        ):
        """Posterior vulnerability of re-identification in a dataset sanitized by k-RR individually on each column.
        The dataset used in the constructor will be considered the original dataset.
        
        Parameters:
            qids (list[str]): List of quasi-identifiers.
            data_san (privattacks.Data): Sanitized version of the dataset.
            epsilons (dict[str,float]): Privacy parameter for each column.

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
    
    def posterior_ai_krr_individual(
            self,
            qids:list[str],
            sensitive:Union[str,List[str]],
            data_san:privattacks.Data,
            epsilons:dict[str,float]
        ):
        """Posterior vulnerability of attribute inference in a dataset sanitized by k-RR individually on each column.
        The dataset used in the constructor will be considered the original dataset.
        
        Parameters:
            qids (list[str]): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            data_san (privattacks.Data): Sanitized version of the dataset.
            epsilons (dict[str, float]): Privacy parameter for each column.

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
    
    def posterior_reid_ai_krr_individual(
            self,
            qids:list[str],
            sensitive:Union[str,List[str]],
            data_san:privattacks.Data,
            epsilons:dict[str,float]
        ):
        """Posterior vulnerability of re-identification and attribute inference in a dataset sanitized by k-RR individually on each column.
        The dataset used in the constructor will be considered the original dataset.
        
        Parameters:
            qids (list[str]): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            data_san (privattacks.Data): Sanitized version of the dataset.
            epsilons (dict[str, float]): Privacy parameter for each column.

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

    def posterior_reid_krr_combined(
            self,
            qids:str,
            data_san:privattacks.Data,
            epsilon:float
        ):
        """Posterior vulnerability of re-identificadtion in a dataset sanitized by k-RR in all columns combined (as if it was a single column). The dataset used in the constructor will be considered the original dataset.
        
        Parameters:
            qids (str): List of quasi-identifiers.
            data_san (privattacks.Data): Sanitized version of the dataset.
            epsilon (float): Privacy parameter.

        Returns:
            dict[str, float]: Dictionary containing the posterior vulnerability for each sensitive attribute.
        """
        self._check_cols(qids)

        domain_size_combined_qids = np.array([len(self.data.domains[qid]) for qid in qids]).prod()
        qid_idxs = [self.data.col2int(qid) for qid in qids]

        dataset_ori = self.data.dataset[:, qid_idxs]
        dataset_san = data_san.dataset[:, qid_idxs]

        # p = probability to keep the original value
        p = math.e**epsilon / (math.e**epsilon + domain_size_combined_qids - 1)

        # For a given target, calculates the probability of each record in the 
        # sanitized dataset to be the sanitized version of the target. 
        # prob = True if the probability is p or False if it's (1-p)/(k-1)
        prob = lambda target, dataset_san: np.all(dataset_san == target, axis=1)

        posterior = 0
        for idx_target, target in enumerate(dataset_ori):
            # Number of records with probability p (the same values as the target)
            probs = prob(target, dataset_san)
            if p >= 1/2:
                # The probability to keep the original tuple is higher than flipping it
                if probs[idx_target]:
                    posterior += 1/probs.sum()
                elif probs.sum() == 0:
                    posterior += 1/self.data.n_rows                    
            else:
                # The probability to flip the tuple is higher than keep the original
                if not probs[idx_target]:
                    posterior += 1/(self.data.n_rows - probs.sum())
                elif probs.sum() == self.data.n_rows:
                    posterior += 1/self.data.n_rows
        
        # Divide by the prior probability of each target
        posterior /= self.data.n_rows

        return posterior
        
    def posterior_ai_krr_combined(
            self,
            qids:str,
            sensitive:Union[str,List[str]],
            data_san:privattacks.Data,
            epsilon:float
        ):
        """Posterior vulnerability of attribute inference in a dataset sanitized by k-RR in all columns together (as if it was a single column). The dataset used in the constructor will be considered the original dataset.
        
        Parameters:
            qids (str): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            data_san (privattacks.Data): Sanitized version of the dataset.
            epsilon (float): Privacy parameter.

        Returns:
            dict[str, float]: Dictionary containing the posterior vulnerability for each sensitive attribute.
        """
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(qids + sensitive)

        domain_size_combined_qids = np.array([len(self.data.domains[qid]) for qid in qids]).prod()
        qid_idxs = [self.data.col2int(qid) for qid in qids]

        dataset_ori = self.data.dataset[:, qid_idxs]
        dataset_san = data_san.dataset[:, qid_idxs]

        # p = probability to keep the original value
        p = math.e**epsilon / (math.e**epsilon + domain_size_combined_qids - 1)

        # For a given target, calculates the probability of each record in the 
        # sanitized dataset to be the sanitized version of the target. 
        # prob = True if the probability is p or False if it's (1-p)/(k-1)
        prob = lambda target, dataset_san: np.all(dataset_san == target, axis=1)

        posteriors = {sens:0 for sens in sensitive}
        for idx_target, target in enumerate(dataset_ori):
            # Number of records with probability p (the same values as the target)
            probs = prob(target, dataset_san)
            if p >= 1/2:
                candidates = np.where(probs)[0]
            else:
                candidates = np.where(~probs)[0]
            
            if len(candidates) == 0:
                candidates = list(range(self.data.n_rows))

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

        for sens in sensitive:            
            # Divide by the prior probability of each target
            posteriors[sens] /= self.data.n_rows

        return posteriors
    
    def posterior_reid_ai_krr_combined(
            self,
            qids:str,
            sensitive:Union[str,List[str]],
            data_san:privattacks.Data,
            epsilon:float
        ):
        """Posterior vulnerability of re-identification and attribute inference in a dataset sanitized by k-RR in all columns together (as if it was a single column). The dataset used in the constructor will be considered the original dataset.
        
        Parameters:
            qids (str): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            data_san (privattacks.Data): Sanitized version of the dataset.
            epsilon (float): Privacy parameter.

        Returns:
            (float, dict[str, float]): A pair where the first element is the posterior vulnerability of re-identificadtion and the second is a dictionary containing the posterior vulnerability for each sensitive attribute.
        """
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(qids + sensitive)

        domain_size_combined_qids = np.array([len(self.data.domains[qid]) for qid in qids]).prod()
        qid_idxs = [self.data.col2int(qid) for qid in qids]

        dataset_ori = self.data.dataset[:, qid_idxs]
        dataset_san = data_san.dataset[:, qid_idxs]

        # p = probability to keep the original value
        p = math.e**epsilon / (math.e**epsilon + domain_size_combined_qids - 1)

        # For a given target, calculates the probability of each record in the 
        # sanitized dataset to be the sanitized version of the target. 
        # prob = True if the probability is p or False if it's (1-p)/(k-1)
        prob = lambda target, dataset_san: np.all(dataset_san == target, axis=1)

        posterior_reid = 0
        posteriors_ai = {sens:0 for sens in sensitive}
        for idx_target, target in enumerate(dataset_ori):
            # Number of records with probability p (the same values as the target)
            probs = prob(target, dataset_san)
    
            if p >= 1/2:
                # Re-identification
                # The probability to keep the original tuple is higher than flipping it
                if probs[idx_target]:
                    posterior_reid += 1/probs.sum()
                elif probs.sum() == 0:
                    posterior_reid += 1/self.data.n_rows

                # Attribute inference
                candidates = np.where(probs)[0]
            else:
                # Re-identification
                # The probability to flip the tuple is higher than keep the original
                if not probs[idx_target]:
                    posterior_reid += 1/(self.data.n_rows - probs.sum())
                elif probs.sum() == self.data.n_rows:
                    posterior_reid += 1/self.data.n_rows

                # Attribute inference
                candidates = np.where(~probs)[0]
            
            if len(candidates) == 0:
                candidates = list(range(self.data.n_rows))

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

        posterior_reid /= self.data.n_rows
        for sens in sensitive:            
            # Divide by the prior probability of each target
            posteriors_ai[sens] /= self.data.n_rows

        return posterior_reid, posteriors_ai
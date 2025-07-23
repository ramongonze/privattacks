import os
import csv
import privattacks
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import itertools as it
import zipfile

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

    def prior_vulnerability(
            self,
            atk,
            sensitive=[]
        ):
        """Prior vulnerability.

        Parameters:
            atk (str): Either 'ai' for attribute inference attack, 'reid' for re-identification or 'all' for both attacks. Default is [].
            sensitive (str or Sequence[str], optional): A single or a list of sensitive attributes for attribute inference attack.

        Returns:
            if atk == 'reid':
                float: Prior vulnerability.

            if atk == 'ai':
                dict[str, float]: Dictionary containing the prior vulnerability for each sensitive attribute (keys are sensitive attribute names and values are posterior vulnerabilities).

            if atk == 'all':
                dict: Dictionary with values 'reid' and 'ai' and their respective prior vulnerabilities.
        """
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(sensitive)

        if atk == "reid":
            return self._prior_reid()
        elif atk == "ai":
            return self._prior_ai(sensitive)
        elif atk == "all":
            return {"reid": self._prior_reid(), "ai":self._prior_ai(sensitive)}
        else:
            raise ValueError("Parameter atk must be 'ai', 'reid', or 'all'.")
        
    def posterior_vulnerability(
            self,
            atk,
            qids,
            sensitive=[],
            distribution=False,
            combinations:list[int]=None,
            save_file=None,
            zip_save=False,
            n_processes=1,
            return_results=True,
            verbose=False
        ):
        """Posterior vulnerability. 
        
        Parameters:
            atk (str): Either 'ai' for attribute inference attack, 'reid' for re-identification or 'all' for both attacks.
            qids (list[str]): List of quasi-identifiers.
            sensitive (str or Sequence[str], optional): A single or a list of sensitive attributes for attribute inference attack. Default is [].
            distribution (bool, optional): Whether to return the distribution of posterior vulnerability per record. Default is False.
            combinations (list[int]): Whether to run the attack for different subset of QIDs (instead of only the list of QIDs given in the parameter 'qids'). It must be provided a list of subset sizes of QIDs. The attack will be run for all subset of QIDs of sizes present in the list.
            zip_save (bool, optional): Save the results in a zip file insteade of csv. Default is False.
            save_file (str, optional): File name to save the results. They will be saved in CSV format. Works only when 'combinations' is given.
            n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1. Works only when 'combinations' is given.
            return_results (bool, optional): Whether to return the results or not. Default is True. Works only when 'combinations' is given.
            verbose (bool, optional): Show the progress. Default is False. Works only when 'combinations' is given.

        Returns:
            if atk == 'reid':
                float or (float, list): If distribution is False, returns the posterior vulnerability.
                If distribution is True, returns a pair (<posterior vulnerability>, <distribution>).
                Example of output when distribution is False::
                    
                    0.75

                Example of output when distribution is True::
                
                    (0.75, [0.5, 0.5, 1.0, 1.0, 0.75])

            if atk == 'ai':
                dict[str, float] or (dict[str, list]):
                If distribution is False, returns a dictionary containing the posterior vulnerability for each sensitive attribute.
                If distribution is True, returns a pair ``(<posterior vulnerability>, <distribution for each sensitive attribute>)``.
                Example of output when distribution is False::
                
                    {'disease': 0.3455, 'income':0.7}

                Example of ouput when distribution is True::
                
                    ({'disease': 0.3455, 'income':0.7},
                     {'disease': [0.1, 0.1, 0.3, 0.4, 0.8275],
                      'income': [0.6, 0.7, 0.7, 0.7, 0.8]})

            if atk == 'all':
                dict: Dictionary with values 'reid' and 'ai' and their respective posterior vulnerabilities.

            if combinations:
                vulnerabilities: Pandas DataFrame with posterior vulnerabilities for all combination of n QIDs, where is the sizes provided in the parameter 'combinations'.
        """
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(qids + sensitive)
        
        if atk == "reid":
            if combinations:
                return self._posterior_reid_subset(
                    qids, combinations, save_file, zip_save, n_processes, distribution, return_results, verbose
                )
            else:
                return self._posterior_reid(qids, distribution)
            
        elif atk == "ai":
            if combinations:
                return self._posterior_ai_subset(
                    qids, sensitive, combinations, save_file, zip_save, n_processes, distribution, return_results, verbose
                )
            else:
                return self._posterior_ai(qids, sensitive, distribution)
        elif atk == "all":
            if combinations:
                return self._posterior_reid_ai_subset(
                    qids, sensitive, combinations, save_file, zip_save, n_processes, distribution, return_results, verbose
                )
            else:
                post_reid, post_ai = self._posterior_reid_ai(qids, sensitive, distribution)
                return {"reid": post_reid, "ai": post_ai}
        else:
            raise ValueError("Parameter atk must be 'ai', 'reid', or 'all'.")
 
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
        cols_idx = self.data.col2int(cols)
        
        # Sort in ascending order (lexicographical sort) 
        # The order must be reversed to use numpy.lexsort (order of priority)
        keys = tuple(self.data.dataset[:, i] for i in cols_idx[::-1])
        sorted_indices = np.lexsort(keys)

        # Use the indices to sort the array
        return self.data.dataset[sorted_indices][:,cols_idx].copy()

    def _prior_reid(self):
        """
        Prior vulnerability of probabilistic re-identification attack.

        Returns:
            float: Prior vulnerability.
        """
        return 1 / self.data.n_rows

    def _prior_ai(self, sensitive):
        """
        Prior vulnerability of probabilistic attribute inference attack.
        
        Parameters:
            sensitive: List of sensitive attributes.

        Returns:
            dict[str, float]: Dictionary containing the prior vulnerability for each sensitive attribute (keys are sensitive attribute names and values are posterior vulnerabilities).
        """
        priors = dict()
        for sens in sensitive:
            priors[sens] = 1/len(self.data.domains[sens])
        return priors

    def _posterior_reid(
            self,
            qids,
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
    
    def _posterior_ai(
            self,
            qids,
            sensitive,
            distribution=False
        ):
        """
        Posterior vulnerability of probabilistic attribute inference attack.

        Parameters:
            qids (list[str]): List of quasi-identifiers.
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

    def _posterior_reid_ai(
            self,
            qids,
            sensitive,
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

    def _posterior_reid_subset(
            self,
            qids,
            n_qids:list[int],
            save_file=None,
            zip_save=False,
            n_processes=1,
            distribution=False,
            return_results=True,
            verbose=False
        ):
        """Posterior vulnerability of probabilistic re-identification attack for subsets of qids. The posterior vulnerability will be calculated for combionations of all sizes of QIDs present in the sizes provided in parameter 'n_qids'.
        
        Parameters:
            qids (list[str]): List of QIDs.
            n_qids (list[int]): List of subset sizes of QIDs. The posterior vulnerability will be calculated for combinations of all sizes of QIDs present in the list.
            save_file (str, optional): File name to save the results. They will be saved in CSV format.
            zip_save (bool, optional): Save the results in a zip file insteade of csv. Default is False.
            n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            distribution (bool, optional): Whether to return the distribution of posterior vulnerability per record. Default is False.
            return_results (bool, optional): Whether to return the results or not. Default is True.
            verbose (bool, optional): Show the progress. Default is False.

        Returns:
            (pandas.DataFrame): A pandas DataFrame containing columns "n_qids", "qids" and "posterior_reid", representing the number of qids in the combination, the actual combination and the posterior vulnerability for the given qid combination, respectively.
        """
        posterior_cols = ["posterior_reid"]
        if distribution:
            posterior_cols += ["posterior_reid_record"]
            partial_method = self._partial_result_reid_record
        else:
            partial_method = self._partial_result_reid

        if save_file:
            # Create a new file with the header
            with open(save_file, mode="w") as file:
                file.write(",".join(["n_qids","qids"] + posterior_cols) + "\n") # Header
        
        float_format = "{:.8f}"  # For 8 decimal places
        posteriors = []
        with multiprocessing.Pool(processes=n_processes) as pool:
            # For qid combinations in the given range run re-identification attack
            for n in tqdm(n_qids, desc="Qids combination size", disable=(not verbose)):

                # Run the attack for all combination of 'n' QIDs
                results = pool.imap_unordered(
                    partial_method,
                    it.combinations(qids, n)
                )

                # Get results from the pool
                partial_result = []
                for qids_comb, posterior in results:
                    if distribution:
                        posterior_vul, posterior_vul_record = posterior
                        posterior_vul_record = [float_format.format(p) for p in posterior_vul_record]
                    else:
                        posterior_vul = posterior
                        posterior_vul_record = []    
                    
                    posterior_partial = [float_format.format(posterior_vul)]

                    if save_file:
                        # Append to save_file
                        with open(save_file, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerows([[int(n), ",".join(qids_comb)] + posterior_partial + posterior_vul_record])
                            
                    if return_results:
                        partial_result.append([int(n), ",".join(qids_comb)] + posterior_partial + posterior_vul_record)
                
                if return_results:
                    # Save once finished all combinations for 'n'
                    posteriors.extend(partial_result)

        if zip_save:
            # Create zip and add the csv inside it
            zip_path = save_file.replace(".csv", ".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(save_file, arcname=os.path.basename(save_file))
            
            # Remove csv file
            os.remove(save_file)

        if return_results:
            posteriors = pd.DataFrame(posteriors, columns=["n_qids", "qids"] + posterior_cols)
            return posteriors
        
    def _posterior_ai_subset(
            self,
            qids,
            sensitive,
            n_qids:list[int],
            save_file=None,
            zip_save=False,
            n_processes=1,
            distribution=False,
            return_results=True,
            verbose=False
        ):
        """Posterior vulnerability of probabilistic attribute inference attack for subsets of qids. The posterior vulnerability will be calculated for combionations of all sizes of QIDs present in the sizes provided in parameter 'n_qids'.
        
        Parameters:
            qids (list[str]): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            n_qids (list[int]): List of subset sizes of QIDs. The posterior vulnerability will be calculated for combinations of all sizes of QIDs present in the list.
            save_file (str, optional): File name to save the results. They will be saved in CSV format.
            zip_save (bool, optional): Save the results in a zip file insteade of csv. Default is False.
            n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            distribution (bool, optional): Whether to return the distribution of posterior vulnerability per record. Default is False.
            return_results (bool, optional): Whether to return the results or not. Default is True.
            verbose (bool, optional): Show the progress. Default is False.

        Returns:
            (pandas.DataFrame): A pandas DataFrame containing columns "n_qids", "qids" and one column "posterior_S" for every sensitive attribute S, representing, respectively, the number of qids in the combination, the actual combination and the posterior vulnerability for each sensitive attribute.
        """
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
            for n in tqdm(n_qids, desc="Qids combination size", disable=(not verbose)):

                # Run the attack for all combination of 'n' QIDs
                results = pool.imap_unordered(
                    partial_method,
                    ((comb,sensitive) for comb in it.combinations(qids, n))
                )

                # Get results from the pool
                partial_result = []
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
                            writer.writerows([[int(n), ",".join(qids_comb)] + posteriors_partial + posterior_vul_record])
                            
                    if return_results:
                        partial_result.append([int(n), ",".join(qids_comb)] + posteriors_partial + posterior_vul_record)
                
                if return_results:
                    # Save once finished all combinations for 'n'
                    posteriors.extend(partial_result)
        
        if zip_save:
            # Create zip and add the csv inside it
            zip_path = save_file.replace(".csv", ".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(save_file, arcname=os.path.basename(save_file))
            
            # Remove csv file
            os.remove(save_file)

        if return_results:
            posteriors = pd.DataFrame(posteriors, columns=["n_qids", "qids"] + posterior_cols)
            return posteriors
    
    def _posterior_reid_ai_subset(
            self,
            qids,
            sensitive,
            n_qids:list[int],
            save_file=None,
            zip_save=False,
            n_processes=1,
            distribution=False,
            return_results=True,
            verbose=False
        ):
        """Posterior vulnerability of probabilistic re-identification and attribute inference attack for subsets of qids. The posterior vulnerability will be calculated for combionations of all sizes of QIDs present in the sizes provided in parameter 'n_qids'.
        
        Parameters:
            qids (list[str]): List of quasi-identifiers.
            sensitive (str, list[str]): A single or a list of sensitive attributes.
            n_qids (list[int]): List of subset sizes of QIDs. The posterior vulnerability will be calculated for combinations of all sizes of QIDs present in the list.
            save_file (str, optional): File name to save the results. They will be saved in CSV format.
            zip_save (bool, optional): Save the results in a zip file insteade of csv. Default is False.
            n_processes (int, optional): Number of processes to run the method in parallel using multiprocessing package. Default is 1.
            distribution (bool, optional): Whether to return the distribution of posterior vulnerability per record. Default is False.
            return_results (bool, optional): Whether to return the results or not. Default is True.
            verbose (bool, optional): Show the progress. Default is False.

        Returns:
            (pandas.DataFrame): A pandas DataFrame containing columns "n_qids", "qids", "posterior_reid", and one column "posterior_S" for every sensitive attribute S, representing, respectively, the number of qids in the combination, the actual combination, the posterior vulnerability for re-identification and the posterior vulnerability for each sensitive attribute.
        """
        posterior_cols = ["posterior_reid"]
        if distribution:
            posterior_cols += ["posterior_reid_record"]
            posterior_cols += [f"posterior_{sens}" for sens in sensitive]
            posterior_cols += [f"posterior_{sens}_record" for sens in sensitive]
            partial_method = self._partial_result_reid_ai_record
        else:
            posterior_cols += [f"posterior_{sens}" for sens in sensitive]
            partial_method = self._partial_result_reid_ai

        if save_file:
            # Create a new file with the header
            with open(save_file, mode="w") as file:
                file.write(",".join(["n_qids", "qids"] + posterior_cols) + "\n") # Header
        
        float_format = "{:.8f}"  # For 8 decimal places
        posteriors = []
        with multiprocessing.Pool(processes=n_processes) as pool:
            # For qid combinations in the given range run re-identification attack
            for n in tqdm(n_qids, desc="Qids combination size", disable=(not verbose)):

                # Run the attack for all combination of 'n' QIDs
                results = pool.imap_unordered(
                    partial_method,
                    ((comb,sensitive) for comb in it.combinations(qids, n))
                )

                # Get results from the pool
                partial_result = []
                for qids_comb, posterior in results:
                    if distribution:
                        posterior_reid, posterior_reid_record, posterior_ai, posterior_ai_record = posterior
                        posterior_reid_record = [float_format.format(p) for p in posterior_reid_record]
                        posterior_ai_record = [[float_format.format(p) for p in posterior_ai_record[sens]] for sens in sensitive]
                    else:
                        posterior_reid, posterior_ai = posterior
                        posterior_reid_record, posterior_ai_record = [], []
                    
                    posterior_partial_reid = [float_format.format(posterior_reid)]
                    posteriors_partial_ai = [float_format.format(posterior_ai[sens]) for sens in sensitive]
                    
                    if save_file:
                        # Append to save_file
                        with open(save_file, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerows(
                                [[int(n), ",".join(qids_comb)] +
                                posterior_partial_reid + posterior_reid_record + 
                                posteriors_partial_ai + posterior_ai_record]
                            )
                    
                    if return_results:
                        partial_result.append(
                            [int(n), ",".join(qids_comb)] +
                            posterior_partial_reid + posterior_reid_record + 
                            posteriors_partial_ai + posterior_ai_record
                        )
                    
                # Save once finished all combinations for 'n'
                posteriors.extend(partial_result)
                
        if zip_save:
            # Create zip and add the csv inside it
            zip_path = save_file.replace(".csv", ".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(save_file, arcname=os.path.basename(save_file))
            
            # Remove csv file
            os.remove(save_file)

        if return_results:
            posteriors = pd.DataFrame(posteriors, columns=["n_qids", "qids"] + posterior_cols)
            return posteriors
   
    def _partial_result_reid(self, qids_subset:list[str]):
        """Required by multiprocessing package in order to use imap_unordered()."""
        return (list(qids_subset), self._posterior_reid(list(qids_subset)))

    def _partial_result_reid_record(self, qids_subset:list[str]):
        """Required by multiprocessing package in order to use imap_unordered().
        It also returns the posterior vulnerability per record."""
        return (list(qids_subset), self._posterior_reid(list(qids_subset), distribution=True))

    def _partial_result_ai(self, params):
        """Required by multiprocessing package in order to use imap_unordered()."""
        qids_subset, sensitive = params
        return (list(qids_subset), self._posterior_ai(list(qids_subset), sensitive))
    
    def _partial_result_ai_record(self, params):
        """Required by multiprocessing package in order to use imap_unordered().
        It also returns the posterior vulnerability per record."""
        qids_subset, sensitive = params
        return (list(qids_subset), self._posterior_ai(list(qids_subset), sensitive, distribution=True))

    def _partial_result_reid_ai(self, params):
        """Required by multiprocessing package in order to use imap_unordered()."""
        qids_subset, sensitive = params
        return (list(qids_subset), self._posterior_reid_ai(list(qids_subset), sensitive))
    
    def _partial_result_reid_ai_record(self, params):
        """Required by multiprocessing package in order to use imap_unordered()."""
        qids_subset, sensitive = params
        return (list(qids_subset), self._posterior_reid_ai(list(qids_subset), sensitive, distribution=True))

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
    """
    Initialize an instance of the Attack class.

    Parameters:
        data (privattacks.Data): An instance of the Data class from the privattacks module. This object represents the dataset to be used for analyzing vulnerabilities to probabilistic re-identification and attribute inference attacks.

    Attributes:
        data (privattacks.Data): Stores the dataset object, providing access to the dataset's attributes, such as columns (`cols`) and number of rows (`n_rows`).
    """

    def __init__(self, data:privattacks.Data):
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
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        self._check_cols(sensitive)

        priors = dict()
        for sens in sensitive:
            # Select the most frequent sensitive value
            sens_idx = self.data.col2int(sens)
            _, counts = np.unique(self.data.dataset[:, sens_idx], return_counts=True)
            priors[sens] = max(counts)/self.data.n_rows
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


class NoisyAttack():
    """
    Initialize an instance of the NoisyAttack class.

    This class analyzes the posterior vulnerability of a probabilistic attribute inference
    attack against a sensitive attribute that has been perturbed by a single-attribute local
    differential privacy (LDP) mechanism: GRR (Generalized Randomized Response), UE (Unary
    Encoding), LH (Local Hashing), SS (Subset Selection), or THE (Threshold with Histogram
    Encoding).

    Unlike Attack, which operates on a single (unnoised) dataset, NoisyAttack requires both
    the original/true dataset and the noised dataset, aligned by row index (row i of `data`
    and row i of `data_noised` must describe the same individual). Quasi-identifiers are
    assumed to come from `data` and to be unaffected by noise, since all supported mechanisms
    only perturb the sensitive attribute. Equivalence-class partitions are therefore built
    from the qids in `data`; within each partition the adversary's guess is derived from the
    noised report(s) in `data_noised`, but a guess counts as correct only when it matches the
    individual's true (unnoised) sensitive value in `data`.

    Only attribute inference attacks are supported: there is no prior_vulnerability method and
    no re-identification attack, since none of the target mechanisms perturb the qids.

    Note:
        `Data` encodes each column's values to integer codes independently, in first-seen
        order. Code 0 in `data.domains[sens]` is not guaranteed to mean the same real value as
        code 0 in `data_noised.domains[col]` unless `data_noised` is built with an explicit
        `domains` argument that reuses the true column's domain list (e.g.
        ``Data(dataframe=..., domains={"income_report": data.domains["income"]})``). Mechanism
        strategies must resolve through `.domains` before comparing values.

    Parameters:
        data (privattacks.Data): The original (unnoised) dataset. Supplies both the qids and
            the ground-truth sensitive attribute values.
        data_noised (privattacks.Data): The noised dataset, containing the LDP mechanism's
            report column(s) for the sensitive attribute(s). Must have the same number of rows
            as `data`, in the same row order (same individuals).
        mechanism (str): Identifier of the LDP mechanism used to produce `data_noised`. One of
            NoisyAttack.MECHANISMS: 'grr' (Generalized Randomized Response), 'ue' (Unary
            Encoding), 'lh' (Local Hashing), 'ss' (Subset Selection), 'the' (Threshold with
            Histogram Encoding). Case-insensitive.

    Attributes:
        data (privattacks.Data): The original (unnoised) dataset.
        data_noised (privattacks.Data): The noised dataset.
        mechanism (str): Normalized (lowercase) mechanism identifier.
    """

    MECHANISMS = ("grr", "ue", "lh", "ss", "the")

    def __init__(self, data:privattacks.Data, data_noised:privattacks.Data, mechanism:str):
        if not isinstance(mechanism, str) or mechanism.lower() not in self.MECHANISMS:
            raise ValueError(
                f"Parameter 'mechanism' must be one of {self.MECHANISMS} (case-insensitive). "
                f"Got: {mechanism!r}."
            )

        if data.n_rows != data_noised.n_rows:
            raise ValueError(
                "'data' and 'data_noised' must have the same number of rows (same "
                f"individuals, same row order). Got data.n_rows={data.n_rows} and "
                f"data_noised.n_rows={data_noised.n_rows}."
            )

        self.data = data
        self.data_noised = data_noised
        self.mechanism = mechanism.lower()

        # Dispatch table: mechanism identifier -> bound strategy method.
        self._strategies = {
            "grr": self._guess_grr,
            "ue": self._guess_ue,
            "lh": self._guess_lh,
            "ss": self._guess_ss,
            "the": self._guess_the,
        }

    def posterior_vulnerability(
            self,
            qids,
            sensitive,
            distribution=False
        ):
        """Posterior vulnerability of attribute inference attack against a
        sensitive attribute noised by a local differentially private mechanism.

        Equivalence-class partitions are formed from `qids` in the true dataset (`self.data`).
        For each partition, the mechanism-specific adversary strategy (selected by
        `self.mechanism`) receives the noised reports and the true sensitive values of the
        individuals in that partition, and returns the expected number of correct guesses for
        the partition. This mirrors Attack's posterior_vulnerability for attribute inference,
        which accumulates one "most frequent value" count per partition and divides the total
        by `self.data.n_rows`.

        Parameters:
            qids (list[str]): List of quasi-identifiers. Must be columns of `self.data`
                (assumed not noised).
            sensitive (dict[str, list[str]]): Maps each sensitive attribute's column name in
                `self.data` to the list of column names in `self.data_noised` that encode its
                noised report (a single column for scalar-report mechanisms like GRR; multiple
                columns for vector-valued encodings like UE's per-domain-position indicator
                columns).
            distribution (bool, optional): Whether to return the distribution of posterior
                vulnerability per record. Default is False.

        Returns:
            dict[str, float] or (dict[str, float], dict[str, numpy.ndarray]):
                If distribution is False, returns a dictionary containing the posterior
                vulnerability for each sensitive attribute (keys are sensitive attribute names
                from `self.data`).
                If distribution is True, returns a pair
                ``(<posterior vulnerability>, <distribution for each sensitive attribute>)``.
                Example of output when distribution is False::

                    {'income': 0.625}

                Example of output when distribution is True::

                    ({'income': 0.625},
                     {'income': [1.0, 0.5, 0.5, 1.0, 1.0, 0.333, 0.333, 0.333]})
        """
        self._check_cols(qids, sensitive)

        # Row permutation computed ONCE from self.data (qids are assumed unnoised). The SAME
        # permutation must be applied to both self.data and self.data_noised so they stay
        # row-aligned -- they must NOT be lexsorted independently.
        sorted_indices = self._lexsort_indices(qids)
        true_sorted = self.data.dataset[sorted_indices]
        noised_sorted = self.data_noised.dataset[sorted_indices]

        qids_idx = self.data.col2int(qids)
        qid_values = true_sorted[:, qids_idx]

        # Find unique qid_values, partition starts (indexes) and partition counts
        _, partition_starts = np.unique(qid_values, axis=0, return_index=True)
        partition_starts.sort()

        strategy = self._strategies[self.mechanism]

        if distribution:
            # Create an array with the posterior vulnerability of each record
            posteriors_record = {sens:[] for sens in sensitive}

        posteriors = {}
        for sens, noised_cols in sensitive.items():
            sens_idx = self.data.col2int(sens) # True sensitive column index
            noised_cols_idx = self.data_noised.col2int(noised_cols) # Noised report column indices

            true_values = true_sorted[:, sens_idx] # True sensitive attribute column
            noised_values = noised_sorted[:, noised_cols_idx] # Noised report columns

            posterior = 0.0
            for i in np.arange(len(partition_starts)):
                start = partition_starts[i]

                # Get the index the partition ends
                if start == partition_starts[-1]:
                    end = self.data.n_rows-1
                else:
                    end = partition_starts[i+1]-1

                true_partition = true_values[start:end+1]
                noised_partition = noised_values[start:end+1]

                # Expected number of correct guesses in the current partition
                expected_correct = strategy(noised_partition, true_partition)
                posterior += expected_correct

                if distribution:
                    partition_size = end-start+1
                    posteriors_record[sens] += [expected_correct/partition_size] * partition_size

            posteriors[sens] = posterior/self.data.n_rows

        if distribution:
            return posteriors, {sens:np.array(posteriors_record[sens]) for sens in sensitive}

        return posteriors

    def _check_cols(self, qids:list[str], sensitive:dict) -> bool:
        """Check that qids and sensitive-attribute columns are valid.

        Parameters:
            qids (list[str]): Quasi-identifiers, expected to be columns of `self.data`.
            sensitive (dict[str, list[str]]): Maps a sensitive attribute's column name in
                `self.data` to a list of column names in `self.data_noised` that encode its
                noised report.

        Raises:
            TypeError: If `sensitive` is not a dict, or a value in `sensitive` is not a list
                (e.g. a bare string was passed instead of a single-element list).
            ValueError: If a qid is not a column of `self.data`, a sensitive attribute key is
                not a column of `self.data`, or a noised column is not a column of
                `self.data_noised`.
        """
        for qid in qids:
            if qid not in self.data.cols:
                raise ValueError(f"Column '{qid}' is not in the (true) dataset.")

        if not isinstance(sensitive, dict):
            raise TypeError(
                "Parameter 'sensitive' must be a dict[str, list[str]] mapping each sensitive "
                "attribute's column name in the true dataset to the list of its noised-report "
                "column names in the noised dataset."
            )

        for sens, noised_cols in sensitive.items():
            if sens not in self.data.cols:
                raise ValueError(f"Sensitive attribute '{sens}' is not in the (true) dataset.")

            if not isinstance(noised_cols, list):
                raise TypeError(
                    f"sensitive['{sens}'] must be a list of column names (even if there is a "
                    f"single noised column), got {noised_cols!r}."
                )

            if len(noised_cols) == 0:
                raise ValueError(f"sensitive['{sens}'] must contain at least one noised column.")

            for col in noised_cols:
                if col not in self.data_noised.cols:
                    raise ValueError(
                        f"Noised column '{col}' (for sensitive attribute '{sens}') is not in "
                        "the noised dataset."
                    )

    def _lexsort_indices(self, qids:list[str]):
        """Row permutation that lexicographically sorts `self.data` by the given columns,
        without reordering or slicing the dataset itself.

        Unlike Attack._sort_dataset (which both permutes rows and slices columns, returning a
        new array), this returns only the permutation. That permutation can then be applied
        independently to both `self.data.dataset` and `self.data_noised.dataset`, keeping the
        two datasets row-aligned.

        Parameters:
            qids (list[str]): Column names in `self.data.cols` to sort by.

        Returns:
            numpy.ndarray: Array of shape (self.data.n_rows,) of row indices such that
                `self.data.dataset[sorted_indices]` is sorted ascending, lexicographically, by
                `qids`.
        """
        cols_idx = self.data.col2int(qids)

        # Sort in ascending order (lexicographical sort)
        # The order must be reversed to use numpy.lexsort (order of priority)
        keys = tuple(self.data.dataset[:, i] for i in cols_idx[::-1])
        return np.lexsort(keys)

    def _guess_grr(self, noised_partition, true_partition) -> float:
        """Adversary strategy for Generalized Randomized Response (GRR).

        Parameters:
            noised_partition (numpy.ndarray): Shape (partition_size, n_noised_cols). Raw
                noised-report codes (indices into self.data_noised.domains) for every
                individual in one equivalence-class partition, row-aligned with
                `true_partition`.
            true_partition (numpy.ndarray): Shape (partition_size,). True sensitive-attribute
                codes (indices into self.data.domains) for the same individuals, in the same
                row order as `noised_partition`.

        Returns:
            float: Expected number of correct guesses within this partition (a value in
                [0, partition_size]) -- not a per-row array. Dividing by partition_size gives
                the per-record posterior vulnerability shared by every individual in the
                partition.
        """
        raise NotImplementedError(
            "NoisyAttack._guess_grr: the GRR (Generalized Randomized Response) adversary "
            "strategy is not implemented yet."
        )

    def _guess_ue(self, noised_partition, true_partition) -> float:
        """Adversary strategy for Unary Encoding (UE).

        Parameters:
            noised_partition (numpy.ndarray): Shape (partition_size, n_noised_cols), where
                n_noised_cols is the size of the sensitive attribute's domain -- each column
                is the noised binary indicator for one domain position. Row-aligned with
                `true_partition`.
            true_partition (numpy.ndarray): Shape (partition_size,). True sensitive-attribute
                codes for the same individuals, same row order.

        Returns:
            float: Expected number of correct guesses within this partition. See
                `_guess_grr` for the full contract.
        """
        raise NotImplementedError(
            "NoisyAttack._guess_ue: the UE (Unary Encoding) adversary strategy is not "
            "implemented yet."
        )

    def _guess_lh(self, noised_partition, true_partition) -> float:
        """Adversary strategy for Local Hashing (LH).

        Parameters:
            noised_partition (numpy.ndarray): Shape (partition_size, n_noised_cols). Raw
                noised-report codes for every individual in the partition (may include
                per-user hashing metadata, e.g. hash seed/index, as extra columns),
                row-aligned with `true_partition`.
            true_partition (numpy.ndarray): Shape (partition_size,). True sensitive-attribute
                codes for the same individuals, same row order.

        Returns:
            float: Expected number of correct guesses within this partition. See
                `_guess_grr` for the full contract.
        """
        raise NotImplementedError(
            "NoisyAttack._guess_lh: the LH (Local Hashing) adversary strategy is not "
            "implemented yet."
        )

    def _guess_ss(self, noised_partition, true_partition) -> float:
        """Adversary strategy for Subset Selection (SS).

        Parameters:
            noised_partition (numpy.ndarray): Shape (partition_size, n_noised_cols). Raw
                noised-report codes (e.g. indicator columns for the reported subset) for every
                individual in the partition, row-aligned with `true_partition`.
            true_partition (numpy.ndarray): Shape (partition_size,). True sensitive-attribute
                codes for the same individuals, same row order.

        Returns:
            float: Expected number of correct guesses within this partition. See
                `_guess_grr` for the full contract.
        """
        raise NotImplementedError(
            "NoisyAttack._guess_ss: the SS (Subset Selection) adversary strategy is not "
            "implemented yet."
        )

    def _guess_the(self, noised_partition, true_partition) -> float:
        """Adversary strategy for Threshold with Histogram Encoding (THE).

        Parameters:
            noised_partition (numpy.ndarray): Shape (partition_size, n_noised_cols). Raw
                noised histogram-encoding report codes for every individual in the partition,
                row-aligned with `true_partition`.
            true_partition (numpy.ndarray): Shape (partition_size,). True sensitive-attribute
                codes for the same individuals, same row order.

        Returns:
            float: Expected number of correct guesses within this partition. See
                `_guess_grr` for the full contract.
        """
        raise NotImplementedError(
            "NoisyAttack._guess_the: the THE (Threshold with Histogram Encoding) adversary "
            "strategy is not implemented yet."
        )

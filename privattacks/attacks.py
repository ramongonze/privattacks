import privattacks
import numpy as np
from typing import Union, List

class Attack():

    def __init__(self, data:privattacks.Data):
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
                
    def prior_reid(self):
        """
        Calculates the prior vulnerability of probabilistic re-identification attack.

        Returns:
            float: Prior vulnerability.
        """
        return 1 / self.data.num_rows

    def prior_ai(self, sensitive:Union[str, List[str]]):
        """
        Calculates the prior vulnerability of probabilistic attribute inference attack.

        Returns:
            dict[str, float]: A dictionary containing the prior vulnerability for each sensitive attribute. Keys are attribute names and values are prior vulnerabilities.
        """
        self._check_sensitive(sensitive)
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        posteriors = dict()
        for att in sensitive:
            sensitive_idx = self.data.col2int(att)
            _, counts = np.unique(self.data.dataset[:, sensitive_idx], return_counts=True)
            posteriors[att] = int(counts.max()) / self.data.num_rows
        return posteriors

    def posterior_reid(self, qids:list[str], hist:bool=False, bin_size:int=5):
        """
        Calculate the expected posterior vulnerability of probabilistic re-identification attack for a given dataset. If hist is True, it provides also the histogram of individual posterior vulnerabilities (i.e., the posterior of each person in the dataset).

        Parameters:
            - qids (list[str]): List of quasi-identifiers.
            - hist (bool, optional): Whether to generate the histogram of posterior vulnerabilities. Default is False.
            - bin_size (int, optional): Histogram bin size. The parameter hist must be True. Default is 5.

        Returns:
            float or tuple: If hist is False, returns the expected posterior vulnerability of Probabilistic Re-identification attack.
            If hist is True, returns a list containing the expected posterior probability and a histogram of individual posterior vulnerabilities. The histogram is an array of size 100/bin_size where the ith element is the count for the ith bin.
        """
        self._check_qids(qids)
        qids_idx = [self.data.col2int(att) for att in qids]
        
        # Groupby by qids
        _, counts = np.unique(self.data.dataset[:, qids_idx], axis=0, return_counts=True)
        posterior = len(counts)/self.data.num_rows
    
        if hist:
            return privattacks.util.hist(counts, bin_size), posterior
        
        return posterior
    
    def posterior_ai(self, qids:list[str], sensitive:Union[str, List[str]], hist:bool=False, bin_size:int=5):
        """
        Calculate the expected posterior vulnerability of probabilistic attribute inference attack for a given dataset. If hist is True, it provides also the histogram of individual posterior vulnerabilities (i.e., the posterior of each person in the dataset).

        Obs: It assumes the dataset is sorted by QID columns + sensitive attribute columsn.

        Parameters:
            - qids (list, optional): List of quasi-identifiers. If not provided, all columns will be used.
            - hist (bool, optional): Whether to generate the histogram of posterior vulnerabilities. Default is False.
            - bin_size (int, optional): Bin size for the histogram if hist is True. Default is 5.

        Returns:
            dict[str, float] or (dict[str, float], tuple): If hist is False, returns a dictionary with the expected posterior vulnerability of probabilistic attribute inference attack for each sensitive attribute.
            If hist is True, returns a tuple containing a dictionary with the expected posterior vulnerability of probabilistic attribute inference attack for each sensitive attribute and the histogram of individual posterior vulnerabilities.
        """
        self._check_qids(qids)
        self._check_sensitive(sensitive)
        
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        qids_idx = [self.data.col2int(att) for att in qids] # Qid column indices
        qid_values = self.data.dataset[:, qids_idx] # Partition identifiers
        
        posteriors = {}
        for sens in sensitive:
            sensitive_idx = self.data.col2int(sens) # Sensitive column index
            sensitive_values = self.data.dataset[:, sensitive_idx] # Sensitive attribute columns

            # Find unique qid_values, partition starts (indexes) and partition counts
            _, partition_starts, partition_sizes = np.unique(qid_values, axis=0, return_index=True, return_counts=True)
            
            # Calculate the adversary's success for each partition
            posterior = 0
            for i, start_idx in enumerate(partition_starts):
                if i == len(partition_starts)-1:
                    end_idx = len(sensitive_values)
                else:
                    end_idx = partition_starts[i+1]

                partition_size = partition_sizes[i]
                # Find the most frequent element in the partition
                _, counts = np.unique(sensitive_values[start_idx:end_idx], return_counts=True)
                
                # The adversary's success for the current partition is the most frequent element / partition size
                posterior += counts.max()
            
            posteriors[sens] = posterior / self.data.num_rows
        
        return posteriors

    def posterior_reid_ai(self, qids:list[str], sensitive:Union[str, List[str]], hist:bool=False, bin_size:int=5):
        """
        Calculate the expected posterior vulnerability of probabilistic re-identification and attribute inference attack for a given dataset. If hist is True, it provides also the histogram of individual posterior vulnerabilities (i.e., the posterior of each person in the dataset).

        Parameters:
            - qids (list, optional): List of quasi-identifiers. If not provided, all columns will be used.


            - hist (bool, optional): Whether to generate the histogram of posterior vulnerabilities. Default is False.
            - bin_size (int, optional): Bin size for the histogram if hist is True. Default is 5.

        Returns:
            (float, float): If hist is False, return the posterior vulnerability for re-identification and attribute inference, respectively.
            ((float, tuple), (float, tuple)): If hist is True, return two pairs (float, tuple), containing the posterior vulnerability and the histogram for re-identification and attribute inference, respectively. The ith element in the histogram is the # people with vulnerability in the bin [i * 100/bin_size, (i+1)*bin_size).
        """
        self._check_qids(qids)
        self._check_sensitive(sensitive)
        
        if isinstance(sensitive, str):
            sensitive = [sensitive]

        qids_idx = [self.data.col2int(att) for att in qids] # Qid column indices
        qid_values = self.data.dataset[:, qids_idx] # Partition identifiers
        
        # Find unique qid_values, partition starts (indexes) and partition counts
        _, partition_starts, partition_sizes = np.unique(qid_values, axis=0, return_index=True, return_counts=True)

        # Re-identification
        posterior_reid = len(partition_sizes)/self.data.num_rows

        # Attribute inference
        posteriors_ai = {}
        for sens in sensitive:
            sensitive_idx = self.data.col2int(sens) # Sensitive column index
            sensitive_values = self.data.dataset[:, sensitive_idx] # Sensitive attribute columns

            # Calculate the adversary's success for each partition
            posterior = 0
            for i, start_idx in enumerate(partition_starts):
                if i == len(partition_starts)-1:
                    end_idx = len(sensitive_values)
                else:
                    end_idx = partition_starts[i+1]

                partition_size = partition_sizes[i]
                # Find the most frequent element in the partition
                _, counts = np.unique(sensitive_values[start_idx:end_idx], return_counts=True)
                
                # The adversary's success for the current partition is the most frequent element / partition size
                posterior += counts.max()
            
            posteriors_ai[sens] = posterior / self.data.num_rows
        
        return posterior_reid, posteriors_ai    

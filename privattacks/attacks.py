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
            sensitive_idx = self.col2int(att)
            _, counts = np.unique(self.data.dataset[:, sensitive_idx], return_counts=True)
            posteriors[att] = int(counts.max()) / self.data.num_rows
        return posteriors

    def posterior_reid(self, qids, hist=False, bin_size=5):
        """
        Calculate the expected posterior vulnerability of probabilistic re-identification attack for a given dataset. If hist is True, it provides also the histogram of individual posterior vulnerabilities (i.e., the posterior of each person in the dataset).

        Parameters:
            - qids (list[str]): List of quasi-identifiers.
            - hist (bool, optional): Whether to generate the histogram of posterior vulnerabilities. Default is False.
            - bin_size (int, optional): Bin size for the histogram if hist is True. Default is 5.

        Returns:
            float or tuple: If hist is False, returns the expected posterior vulnerability of Probabilistic Re-identification attack.
            If hist is True, returns a tuple containing the expected posterior probability and a histogram of individual posterior vulnerabilities.
        """
        self._check_qids(qids)
        qids_idx = [self.data.col2int(att) for att in qids]
        _, counts = np.unique(self.data.dataset[:, qids_idx], return_counts=True)
        posterior = len(counts)/self.data.num_rows
    
        if hist:
            return privattacks.util.hist(counts, bin_size), posterior
        
        return posterior
    
    def posterior_ai(self):
        pass

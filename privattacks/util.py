import numpy as np

def hist(partitions, bin_size) -> np.ndarray:
    """Generate a histogram of posterior vulnerabilities given partition sizes.

    Parameters:
        - partitions (list or np.ndarray): Partitions of a dataset. It's the result of a groupby of qids (in the case of re-identification) or the result of a groupby of qids + sensitive attribute (in the case of attribute inference).
        - bin_size (int): Histogram bin size.

    Returns:
        np.ndarray: An array representing the histogram counts, where the ith position is the number of counts in the ith bin of the histograms (that varies with bin_size). For instance, if bin_size=5 then bin 0 = [0, 0.05), bin 2 = [0.05, 0.1), ..., bin 19 = [0.95, 1].
    """
    if bin_size < 1 or bin_size > 100:
        raise ValueError("Invalid bin_size. It must be an int between 1 and 100.")
    
    num_bins = 100 // bin_size

    hist = np.zeros(num_bins)
    for i in np.arange(len(partitions)):
        # Ex.: If bin_size = 5, bin 0: [0, 0.05), bin 2: [0.05, 0.1), ..., bin 19: [0.95, 1]
        bin_number = min(int(100/partitions[i] / bin_size), num_bins - 1)
        hist[bin_number] += partitions[i]

    return hist
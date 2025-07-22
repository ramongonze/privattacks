import numpy as np

def create_histogram(ind_posteriors, bin_size=1) -> dict:
    """Generate a histogram of posterior vulnerabilities given partition sizes.

    Parameters:
        - ind_posteriors (list or np.ndarray): Individual posterior vulnerabilties for all records in the dataset.
        - bin_size (int, optional): Histogram bin size. For instance, if bin_size=5 then bin 0 = [0, 0.05), bin 2 = [0.05, 0.1), ..., bin 19 = [0.95, 1]. Default is 5.

    Returns:
        hist (dict): A dictionary containing the histogram. Keys are strings (e.g., '[0, 0.05)', '[0.95,1]') and values are the counts of the respective bins.
    """
    if bin_size < 1 or bin_size > 100:
        raise ValueError("Invalid bin_size. It must be an int between 1 and 100.")
    
    num_bins = int(100/bin_size)
    labels = []
    hist = dict()
    for i in np.arange(num_bins):
        if i < num_bins-1:
            label = f"[{i * (bin_size / 100):.2f},{(i + 1) * (bin_size / 100):.2f})"
        else:
            label = f"[{i * (bin_size / 100):.2f},1]"
        labels.append(label)
        hist[label] = 0
    
    for prob in ind_posteriors:
        # Ex.: If bin_size = 5, bin 0: [0, 0.05), bin 2: [0.05, 0.1), ..., bin 19: [0.95, 1]
        bin_number = min(int(100*prob/bin_size), num_bins - 1)
        hist[labels[bin_number]] += 1
    
    return hist
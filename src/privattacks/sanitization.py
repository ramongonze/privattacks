import copy
import privattacks
import numpy as np
import pandas as pd
import multiprocessing

def check_cols(data:privattacks.Data, cols:list[str]):
    """Check if columns are in the list of columns of the dataset."""
    for col in cols:
        if col not in data.cols:
            raise ValueError(f"Column {col} not in the dataset.")

def krr(value, domain_size, epsilon):
    """It's assumed the domain values are in [0, domain_size-1]."""
    # Probability to keep the original value
    p = np.exp(epsilon) / (np.exp(epsilon) + domain_size - 1)

    # Bernoulli experiment
    if np.random.binomial(n=1, p=p) == 1:
        return value
    
    # Return a value different of the given 'value'
    new_value = np.random.randint(0,domain_size-1)
    return new_value if new_value < value else new_value + 1

def add_noise_krr_ind(params):
    data_ori, cols, domain_sizes, epsilons, seed, vectorized_krr = params
    
    # Re-seed inside the worker so each instance is unique
    np.random.seed(seed)

    data_san = copy.deepcopy(data_ori)
    for col in cols:
        col_idx = data_san.col2int(col)
        data_san.dataset[:, col_idx] = vectorized_krr(
            data_san.dataset[:, col_idx],
            domain_sizes[col],
            epsilons[col]
        )

    return data_san

def params_krr_ind(data_ori, cols, domain_sizes, epsilons, n_instances, child_seeds, vectorized_krr):
    i = 0
    while i < n_instances:
        yield (data_ori, cols, domain_sizes, epsilons, child_seeds[i], vectorized_krr)
        i += 1

def krr_individual(
        data:privattacks.Data,
        domain_sizes:dict,
        epsilons:dict[str, float],
        n_instances=1,
        n_processes=1,
        seed=None
    ) -> list:
    """k-Randomized-Response (k-RR) mechanism. Adds noise individually to each column.

    Parameters:
        data (privattacks.Data): Original data.
        domain_sizes (dict[str, int]): Column domain sizes. Keys are names of columns and values are integers.
        epsilons (dict[str, float]): Privacy parameter. Keys are names of columns and values are float. All values must be greater than zero.
        n_instances (int, optional): Number of sanitized instances. Default is 1.
        n_processes (int, optional): Number of cores to run in parallel. Default is 1.
        seed (int, optional): Seed for randomness.

    Returns:
        data (list[privattacks.Data]): List of sanitized dataset.
    """
    check_cols(data, set(domain_sizes.keys()))
    check_cols(data, set(epsilons.keys()))

    # Check if the set of variables is the same for parameters domain_sizes and epsilons
    if set(domain_sizes.keys()) != set(epsilons.keys()):
        raise ValueError("The set of columns of parameters domain_sizes and epsilons are different.")

    cols = list(domain_sizes.keys())

    # Check if all epsilons are greater than 0
    assert (np.array(list(epsilons.values())) > 0).sum() == len(epsilons), "All epsilons must be greater than zero"

    # If user provided a seed, use it to initialize the random module
    if seed is not None:
        np.random.seed(seed)
    
     # Generate one child seed per instance
    child_seeds = [np.random.randint(0, 2**32 - 1) for _ in range(n_instances)]

    # Create sanitized datasets
    vectorized_krr = np.vectorize(krr)
    data_san = []
    with multiprocessing.Pool(processes=n_processes) as pool:
        # Run the attack for all combination of 'n_qids' QIDs
        results = pool.imap_unordered(
            add_noise_krr_ind,
            params_krr_ind(data, cols, domain_sizes, epsilons, n_instances, child_seeds, vectorized_krr)
        )

        # Get results from the pool
        for data_san_i in results:
            data_san.append(data_san_i)
        
    return data_san

def getidx(domains_arr, tup, mult):
    # Merge quasi-identifier columns into a single column
    # Calculate the index
    idx = 0
    for i, val in enumerate(tup):
        position = domains_arr[i].index(val)  # Get the idx of the value in the domain
        idx += position * mult[i]
    
    return idx

def gettup(index, domains_arr, domain_sizes):
    """Get a tuple (original domains) given a index in the single domain (merged quasi-identifiers)."""
    indices = []
    while domain_sizes.size > 0:
        indices.append(index % domain_sizes[-1])
        index //= domain_sizes[-1]
        domain_sizes = domain_sizes[:-1]

    return [domains_arr[i][value] for i, value in enumerate(indices[::-1])]

def multipliers(domain_sizes):
    """Multipliers used to calculate the index of a tuple for the domain of all attributes together.""" 
    # The multiplier of last column is 1
    mult = [1] * len(domain_sizes)
    
    for i in range(len(domain_sizes) - 2, -1, -1):
        mult[i] = mult[i+1] * domain_sizes[i+1]
    
    return mult

def add_noise_krr_comb(params):    
    new_dataset, qids, epsilon, new_domain_size, qid_domains_arr, qid_domain_sizes, seed = params

    # Re-seed inside the worker so each instance is unique
    np.random.seed(seed)

    dataset_san = new_dataset.copy()
    # Generate the noised tuple and convert back to original domains
    dataset_san[qids] = new_dataset["qids_combined"].apply(
        lambda record: gettup(krr(record, new_domain_size, epsilon), qid_domains_arr, qid_domain_sizes)
    ).tolist()

    return dataset_san.drop("qids_combined", axis=1)

def params_krr_comb(dataset_ori, qids, domains, epsilon, n_instances, child_seeds):
    """Parameters for adding noise to dataset with krr_combined."""
    # New new values are integers from 0 to new_domain_size-1
    qid_domains_arr = [domains[qid] for qid in qids]
    qid_domain_sizes = np.array([len(domains[qid]) for qid in qids])
    new_domain_size = qid_domain_sizes.prod()
    mult = multipliers(qid_domain_sizes)

    new_dataset = dataset_ori.copy()
    new_dataset["qids_combined"] = dataset_ori[qids].apply(lambda x : getidx(qid_domains_arr, x.values, mult), axis=1)
    
    i = 0
    while i < n_instances:
        yield (new_dataset, qids, epsilon, new_domain_size, qid_domains_arr, qid_domain_sizes, child_seeds[i])
        i += 1

def krr_combined(
        dataset_ori:pd.DataFrame,
        qids:list[str],
        domains:dict[str, list],
        epsilon:float,
        n_instances=1,
        n_processes=1,
        seed=None
    ) -> list[privattacks.Data]:
    """k-Randomized-Response (k-RR) mechanism. Adds noise to a record considering all quasi-identifiers as one single attribute.

    Parameters:
        dataset_ori (pd.DataFrame): Original data.
        qids (list[str]): List of quasi-identifiers. The other attributes will be considered sensitive.
        domains (dict[str, list], optional): Domain of columns.
        epsilon float: Privacy parameter.
        n_instances (int, optional): Number of sanitized instances. Default is 1.
        n_processes (int, optional): Number of cores to run in parallel. Default is 1.
        seed (int, optional): Seed for randomness.

    Returns:
        san_dataset (list[privattacks.Data]): List of sanitized datasets.
    """
    # If user provided a seed, use it to initialize the random module
    if seed is not None:
        np.random.seed(seed)
    
     # Generate one child seed per instance
    child_seeds = [np.random.randint(0, 2**32 - 1) for _ in range(n_instances)]

    data_san = []
    # Create sanitized datasets. The new domain of the single attribute (merged quasi-identifiers)
    # will be the union of all values presented in the original and in all sanitized instances.
    with multiprocessing.Pool(processes=n_processes) as pool:
        # Run the attack for all combination of 'n_qids' QIDs
        results = pool.imap_unordered(
            add_noise_krr_comb,
            params_krr_comb(dataset_ori, qids, domains, epsilon, n_instances, child_seeds)
        )

        # Get results from the pool
        for data_san_i in results:
            data_san.append(privattacks.Data(dataframe=data_san_i.copy(), domains=domains))
        
    return data_san

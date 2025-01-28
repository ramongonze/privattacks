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

def krr_individual(data:privattacks.Data, domain_sizes:dict, epsilons:dict[str, float]):
    """k-Randomized-Response (k-RR) mechanism. Adds noise individually to each column.

    Parameters:
        data (privattacks.Data): Original data.
        domain_sizes (dict[str, int]): Column domain sizes. Keys are names of columns and values are integers.
        epsilons (dict[str, float]): Privacy parameter. Keys are names of columns and values are float. All values must be greater than zero.

    Returns:
        data (privattacks.Data): Sanitized dataset.
    """
    check_cols(data, set(domain_sizes.keys()))
    check_cols(data, set(epsilons.keys()))

    # Check if the set of variables is the same for parameters domain_sizes and epsilons
    if set(domain_sizes.keys()) != set(epsilons.keys()):
        raise ValueError("The set of columns of parameters domain_sizes and epsilons are different.")

    cols = list(domain_sizes.keys())

    # Check if all epsilons are greater than 0
    assert (np.array(list(epsilons.values())) > 0).sum() == len(epsilons), "All epsilons must be greater than zero"

    # Sanitize dataset column by column
    vectorized_krr = np.vectorize(krr)
    data_san = copy.deepcopy(data)
    for col in cols:
        col_idx = data_san.col2int(col)
        data_san.dataset[:, col_idx] = vectorized_krr(
            data_san.dataset[:, col_idx],
            domain_sizes[col],
            epsilons[col]
        )

    return data_san

# Merge quasi-identifier columns into a single column
def getidx(domains_arr, tup, mult):
    # Calculate the index
    idx = 0
    for i, val in enumerate(tup):
        position = domains_arr[i].index(val)  # Get the idx of the value in the domain
        idx += position * mult[i]
    
    return idx

def gettup(index, domain_sizes):
    indices = []    
    while domain_sizes.size > 0:
        indices.append(index % domain_sizes[-1])
        index //= domain_sizes[-1]
        domain_sizes = domain_sizes[:-1]
    
    return tuple(indices[::-1])

def multipliers(domain_sizes):
    """Multipliers used to calculate the index of a tuple for the domain of all attributes together.""" 
    # The multiplier of last column is 1
    mult = [1] * len(domain_sizes)
    
    for i in range(len(domain_sizes) - 2, -1, -1):
        mult[i] = mult[i+1] * domain_sizes[i+1]
    
    return mult

# Sanitized datasets
def generate_data(params):
    i, dataframe, cols, domains = params
    return (
        i,
        privattacks.Data(
            dataframe=dataframe,
            cols=cols,
            cols_to_ignore=["qids_combined"],
            domains=domains
        )
    )

def params(san_data, cols, domains, n_instances):
    i = 0
    while i < n_instances:
        yield (i, san_data[i], cols, domains)
        i += 1

def krr_combined(
        dataset:pd.DataFrame,
        qids:list[str],
        domains:dict[str, list],
        epsilon:float,
        n_instances:int,
        n_processes=1
    ) -> list[privattacks.Data]:
    """k-Randomized-Response (k-RR) mechanism. Adds noise to a record considering all quasi-identifiers as one single attribute.

    Parameters:
        dataset (pd.DataFrame): Original data.
        qids (list[str]): List of quasi-identifiers. The other attributes will be considered sensitive.
        domains (dict[str, list], optional): Domain of columns.
        epsilon float: Privacy parameter.
        n_instances (int): Number of sanitized instances.
        n_processes (int, optional): Number of cores to run in parallel. Default is 1.

    Returns:
        ori_data, san_dataset (list[privattacks.Data]): Pair original dataset with the new single attribute and a list of sanitized datasets with the new single attribute (merged quasi-identifiers).
    """
    sensitive = list(set(dataset.columns) - set(qids))

    # New new values are integers from 0 to new_domain_size-1
    qid_domains_arr = [domains[qid] for qid in qids]
    qid_domain_sizes = np.array([len(domains[qid]) for qid in qids])
    new_domain_size = qid_domain_sizes.prod()
    mult = multipliers(qid_domain_sizes)

    new_dataset = dataset.copy()
    new_dataset["qids_combined"] = dataset[qids].apply(lambda x : getidx(qid_domains_arr, x.values, mult), axis=1)
    new_dataset = new_dataset[["qids_combined"] + sensitive]

    san_data = []
    domain_qids_combined = set(new_dataset["qids_combined"].values)
    # Create sanitized datasets. The new domain of the single attribute (merged quasi-identifiers)
    # will be the union of all values presented in the original and in all sanitized instances.
    for _ in np.arange(n_instances):
        df = new_dataset.copy()
        # Generate the noised tuple and convert back to original domains
        df["qids_combined"] = df["qids_combined"].apply(lambda record: krr(record, new_domain_size, epsilon)).tolist()
        domain_qids_combined.update(df["qids_combined"].values)
        san_data.append(df)

    # Generate privattacks.Data data with sanitized pandas dataframes
    new_domain = {"qids_combined":list(domain_qids_combined)} | {sens:domains[sens] for sens in sensitive}
    with multiprocessing.Pool(processes=n_processes) as pool:
        # Run the attack for all combination of 'n_qids' QIDs
        results = pool.imap_unordered(
            generate_data,
            params(san_data, ["qids_combined"] + sensitive, new_domain, n_instances)
        )

        # Get results from the pool. result[i] = (i, san_data[i])
        for i, data_san_i in results:
            san_data[i] = data_san_i

    ori_data = privattacks.Data(
        dataframe=new_dataset,
        cols=["qids_combined"] + sensitive,
        cols_to_ignore=["qids_combined"],
        domains=new_domain
    )

    return ori_data, san_data

import copy
import privattacks
import numpy as np
from typing import Union, List, Tuple, Dict

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
        dataset (np.ndarray): Dataset of interest. All column domains must be integers from 0 to domain_size-1.
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
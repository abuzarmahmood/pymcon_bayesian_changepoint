"""
Utility for returning data for specific examples
"""

import numpy as np
import pandas as pd



def return_simple_data():
    """
    Returns data for the simple poisson changepoint example

    Input:
        None

    Output:
        counts: pd.Series of counts
        years: np.array of years
    """

    disaster_data = pd.Series(
            [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
             3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
             2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
             1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
             0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
             3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
            )
    years = np.arange(1851, 1962)

    na_bool = np.logical_not(np.isnan(disaster_data))
    disaster_data = disaster_data.loc[na_bool]
    years = years[na_bool]

    return disaster_data, years

def return_bernoulli_mixture_data(
        length = 100,
        n_states = 3,
        n_trials = 30,
        n_components = 2,
        mixture_probs = None,
        ):
    """
    Returns data for the bernoulli mixture example

    Input:
        length: length of data
        n_states: number of states
        n_trials: number of trials
        n_components: number of mixutre components

    Output:
        true_lambda: true firing rate values
        data_vec: spike_array
        component_inds: indices of mixture components each trial is drawn from
        idx: time indices
    """
    if mixture_probs is None: 
        mixture_probs = np.random.random(n_components)
        mixture_probs /= np.sum(mixture_probs)
    else:
        assert len(mixture_probs) == n_components, "mixture_probs must be of length n_components" 
        assert np.sum(mixture_probs) == 1, "mixture_probs must sum to 1"

    idx = np.arange(length)

    # Generate transition times
    true_tau = np.cumsum(np.random.random((n_trials,n_states)),axis=-1)
    true_tau /= np.max(true_tau,axis=-1)[:,np.newaxis]
    true_tau *= length
    true_tau = np.vectorize(int)(true_tau)

    # Generate state boundaries using transition times
    state_inds = np.concatenate([np.zeros((n_trials,1)),true_tau],axis=-1)
    state_inds = np.vectorize(int)(state_inds)
    true_tau = true_tau[:-1]

    # Generate emission values
    true_lambda = np.random.random((n_components,n_states))

    # Generate "trials" from n different mixture 
    # components with uneven mixture probabilities
    component_inds = np.random.choice(range(n_components), n_trials, p = [0.3,0.7])
    true_r = np.zeros((n_trials,length))
    for trial in range(n_trials):
        for state in range(n_states):
            true_r[trial, state_inds[trial,state]:state_inds[trial,state+1]] = \
                    true_lambda[component_inds[trial],state]

    # Final spiking data
    data_vec = np.random.random(true_r.shape) < true_r

    return true_r, data_vec, component_inds, idx, true_lambda

def return_dirichlet_process_data(
        length = 100,
        n_states = 3,
        n_neurons = 10,
        seesaw_emissions = True,
        ):
    """
    Returns data for the dirichlet process example

    Input:
        length: length of data
        n_states: number of states
        n_neurons: number of neurons

    Output:
        true_r: true firing rate values
        true_tau: true state boundaries
        true_lambda: true emission rates

    """

    true_tau = np.cumsum(np.random.random(n_states))
    true_tau /= np.max(true_tau)
    true_tau *= length
    true_tau = np.vectorize(int)(true_tau)
    state_inds = np.concatenate([np.zeros((1)),true_tau])
    state_inds = np.vectorize(int)(state_inds)
    true_tau = true_tau[:-1]

    lambda_multipliers = np.random.random(n_states) * 10
    if seesaw_emissions:
        true_lambda = np.array(
                [np.ones(n_neurons) * (i % 2) for i in range(n_states)]
                ).T
    else:
        true_lambda = np.random.random((n_neurons,n_states))
        true_lambda = true_lambda * lambda_multipliers[np.newaxis,:]

    true_r = np.zeros((n_neurons,length))
    for num, val in enumerate(true_lambda.T):
        true_r[:,state_inds[num]:state_inds[num+1]] = val[:,np.newaxis]

    return true_r, true_tau, true_lambda

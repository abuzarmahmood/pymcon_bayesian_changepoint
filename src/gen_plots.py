"""
Utility for generating plots for specific examples 
"""

import numpy as np
import pylab as plt
import seaborn as sns
import pymc as pm
from scipy import stats
import pandas as pd

def gen_simple_plots(
        years,
        disaster_data,
        disaster_model,
        disaster_trace,
        figsize = (7,7),
        ):
    with disaster_model:
      disaster_ppc = pm.sample_posterior_predictive(disaster_trace)

    ppc_values = disaster_ppc['posterior_predictive']['disasters']
    mean_ppc, std_ppc = np.mean(ppc_values,axis=(0,1)),np.std(ppc_values,axis=(0,1))

    disaster_switch_inferred = disaster_trace["posterior"]["switchpoint"]
    mean_switch,std_switch = disaster_switch_inferred.mean(), disaster_switch_inferred.std()

    fig, ax = plt.subplots(2,1,figsize=figsize)
    ax[0].scatter(years, disaster_data, marker = ".", s = 200)
    ax[0].set_ylabel("Number of accidents") 
    ax[0].set_xlabel("Year") 

    ax[0].vlines(mean_switch, disaster_data.min(), disaster_data.max(),
               color="k", label = 'Mean switchpoint')

    ax[0].vlines(disaster_switch_inferred.min(), disaster_data.min(), disaster_data.max(),
               color="k", linestyles = 'dashed')
    ax[0].vlines(disaster_switch_inferred.max(), disaster_data.min(), disaster_data.max(),
               color="k", label = 'Switchpoint bounds', linestyles = 'dashed')
    ax[0].axvspan(mean_switch-3*std_switch, mean_switch+3*std_switch, alpha = 0.5,
                color='r', label = '+/- 3 std Switchpoint')
    ax[0].plot(years, mean_ppc, label = 'Average disasters', c='red', lw=5)
    ax[0].legend()

    ax[1].scatter(years, disaster_data, marker = ".", s = 200)
    ax[1].set_ylabel("Number of accidents") 
    ax[1].set_xlabel("Year") 
    ax[1].fill_between(years, mean_ppc-std_ppc, mean_ppc+std_ppc,
                       label = '+/- 1 std mean rate', color='red', alpha = 0.5)
    ax[1].vlines(disaster_switch_inferred.min(), disaster_data.min(), disaster_data.max(),
               color="k", linestyles = 'dashed')
    ax[1].vlines(disaster_switch_inferred.max(), disaster_data.min(), disaster_data.max(),
               color="k", label = 'Switchpoint bounds', linestyles = 'dashed')
    ax[1].legend()
    fig.suptitle('Inference Outputs')

def gen_bernoulli_plots(
    trace,
    component_inds,
    n_trials,
    true_r,
    figsize = (7,3),
    ):

    tau_samples = trace['posterior']['tau']
    int_tau = np.vectorize(int)(tau_samples)
    mode_tau = stats.mode(int_tau,axis=0)[0][0]
    hist_tau = np.array([np.histogram(trial, bins = np.arange(20))[0] for trial in int_tau.swapaxes(0,1)])
    
    w_samples = trace['posterior']['w'].values
    w_samples_long = np.reshape(w_samples,(-1,w_samples.shape[-1]))
    mean_w = np.mean(w_samples,axis=(0,1))
    categorical_w = np.argmax(mean_w,axis=-1)
    
    cat_accuracy_list = [np.sum(categorical_w==component_inds),
                          np.sum((1-categorical_w)==component_inds)]
    model_ind = int(np.argmax(cat_accuracy_list))
    cat_accuracy = cat_accuracy_list[model_ind]
    
    
    if np.mean(categorical_w == component_inds) < 0.5:
        plot_inferred_inds = 1-categorical_w
    else:
        plot_inferred_inds = categorical_w
    
    fig,ax = plt.subplots(1,3,figsize=figsize)
    ax[0].set_title('True rates with transitions')
    ax[0].imshow(true_r,aspect='auto',origin='lower',vmin=0,vmax=1)
    ax[1].plot(component_inds, np.arange(n_trials),'-o',label='Actual', alpha = 0.5)
    ax[1].plot(plot_inferred_inds, np.arange(n_trials),'-x',label='Inferred', alpha = 0.5)
    ax[1].set_title(f'Accuracy : {cat_accuracy}/{categorical_w.shape[0]}')
    ax[1].legend()
    for trial,val in enumerate(mode_tau):
        ax[0].vlines(val, trial-0.5,trial+0.5, linewidth = 5, color='red')
    ax[2].imshow(mean_w,aspect='auto',origin='lower');
    ax[2].set_title('Mixture weights')
    ax[1].set_ylabel('Trial #')
    ax[1].set_xlabel('Mixture #')
    ax[2].set_ylabel('Trial #')
    ax[2].set_xlabel('Mixture #')
    plt.tight_layout()

def gen_dirichlet_plots(
    dpp_trace,
    true_r,
    n_chains,
    true_tau,
    length,
    n_states,
    max_states,
    figsize = (7,15),
    ):

    w_latent_samples = dpp_trace['posterior']['w_latent'].values
    cat_w_latent_samples = np.concatenate(w_latent_samples)
    sorted_lens = np.sort(cat_w_latent_samples,axis=-1)[:,::-1]

    sorted_w_latent = np.stack(np.array_split(np.sort(w_latent_samples,axis=-1)[...,::-1],n_chains,axis=0))
    sorted_w_latent = np.squeeze(sorted_w_latent)
    mean_sorted = np.mean(sorted_w_latent, axis = 1)

    all_state_edges = np.concatenate([[0],true_tau,[length]])
    state_durations = np.abs(np.diff(all_state_edges))
    sorted_state_durations = np.sort(state_durations / length)[::-1]
    shortest_state = sorted_state_durations[-1]

    inds = np.array(list(np.ndindex(mean_sorted.shape)))
    state_frame = pd.DataFrame(
                            dict(
                                chains = inds[:,0],
                                states = inds[:,1]+1,
                                dur = mean_sorted.flatten()
                            )
                        )

    fig,ax = plt.subplots(5,1, figsize = figsize)

    sns.stripplot(
        data = state_frame,
        x = 'states',
        y = 'dur',
        color = 'k',
        ax = ax[0]
    )
    ax[0].plot(mean_sorted.T, alpha = 0.7, color = 'grey')

    ax[0].axvline(n_states-1, zorder = -1, color = 'black', label = 'Actual')
    ax[0].legend()
    ax[0].axhline(shortest_state, color = 'red', linestyle = '--')
    ax[0].text(0, shortest_state, 'Shortest state')
    ax[0].axhline(0.01, color = 'red', linestyle = '--')
    ax[0].text(0, 0.01, '0.01')
    ax[0].set_xlabel('State #')
    ax[0].set_ylabel('Fractional Duration')

    corrected_transitions = np.cumsum(sorted_w_latent,axis=-1)
    tau_samples = dpp_trace['posterior']['tau'].values
    ax[1].imshow(true_r,aspect='auto', interpolation = 'nearest')
    ax[1].set_title('True rates')
    ax[1].set_ylabel('Neuron #')
    ax[1].set_xlabel('Time')
    ax[2].hist(tau_samples.flatten(), bins = np.arange(length), color = 'grey')
    ax[2].sharex(ax[1])
    ax[2].set_title('Tau samples')
    ax[2].set_ylabel('Count')
    ax[2].set_xlabel('Time')

    im1 = ax[3].imshow(sorted_lens, interpolation='nearest', aspect= 'auto')
    ax[3].set_title('Sorted latent_w')
    ax[3].set_ylabel('Sample #')
    ax[3].set_xlabel('State #')
    fig.colorbar(im1, ax=ax[3], label = 'State Duration')
    plt.tight_layout()

    max_state_per_chain = state_frame.loc[state_frame.dur > 0.01].groupby('chains').max()
    max_state_counts = max_state_per_chain.groupby('states').count()
    state_vec = np.arange(1,max_states+1)
    counts = [max_state_counts.loc[x].values[0] if x in max_state_counts.index else 0 for x in state_vec ]
    ax[4].axvline(n_states, zorder = 2, color = 'black', label = 'Actual')
    ax[4].bar(state_vec, counts, label = 'Inferred')
    ax[4].set_xlabel("States")
    ax[4].set_ylabel('Count')
    ax[4].set_title('Number of states')
    ax[4].legend()

    plt.tight_layout()


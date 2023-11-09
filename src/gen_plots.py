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
        poisson_data,
        simple_model,
        poisson_trace,
        figsize = (7,7),
        ):
    with simple_model:
      poisson_ppc = pm.sample_posterior_predictive(poisson_trace)

    ppc_values = poisson_ppc['posterior_predictive']['obs']
    mean_ppc, std_ppc = np.mean(ppc_values,axis=(0,1)),np.std(ppc_values,axis=(0,1))

    poisson_switch_inferred = poisson_trace["posterior"]["switchpoint"]
    mean_switch,std_switch = poisson_switch_inferred.mean(), poisson_switch_inferred.std()

    fig, ax = plt.subplots(2,1,figsize=figsize)
    ax[0].scatter(years, poisson_data, marker = ".", s = 200)
    ax[0].set_ylabel("Number of Counts") 
    ax[0].set_xlabel("Time") 

    ax[0].vlines(mean_switch, poisson_data.min(), poisson_data.max(),
               color="k", label = 'Mean switchpoint')

    ax[0].vlines(poisson_switch_inferred.min(), poisson_data.min(), poisson_data.max(),
               color="k", linestyles = 'dashed')
    ax[0].vlines(poisson_switch_inferred.max(), poisson_data.min(), poisson_data.max(),
               color="k", label = 'Switchpoint bounds', linestyles = 'dashed')
    ax[0].axvspan(mean_switch-3*std_switch, mean_switch+3*std_switch, alpha = 0.5,
                color='r', label = '+/- 3 std Switchpoint')
    ax[0].plot(years, mean_ppc, label = 'Average poissons', c='red', lw=5)
    ax[0].legend()

    ax[1].scatter(years, poisson_data, marker = ".", s = 200)
    ax[1].set_ylabel("Number of Counts") 
    ax[1].set_xlabel("Time") 
    ax[1].fill_between(years, mean_ppc-std_ppc, mean_ppc+std_ppc,
                       label = '+/- 1 std mean rate', color='red', alpha = 0.5)
    ax[1].vlines(poisson_switch_inferred.min(), poisson_data.min(), poisson_data.max(),
               color="k", linestyles = 'dashed')
    ax[1].vlines(poisson_switch_inferred.max(), poisson_data.min(), poisson_data.max(),
               color="k", label = 'Switchpoint bounds', linestyles = 'dashed')
    ax[1].legend()
    fig.suptitle('Inference Outputs')

def gen_bernoulli_main_plot(
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


    plot_components = np.zeros((n_trials, 2))
    if np.mean(categorical_w == component_inds) < 0.5:
        flip_emissions = True
        plot_inferred_inds = 1-categorical_w
        plot_components[np.where(component_inds)[0],0] = 1
        plot_components[np.where(1-component_inds)[0],1] = 1
    else:
        flip_emissions = False
        plot_inferred_inds = categorical_w
        plot_components[np.where(1-component_inds)[0],0] = 1
        plot_components[np.where(component_inds)[0],1] = 1

    fig,ax = plt.subplots(1,4,figsize=figsize, sharey=True)
    ax[0].set_title('True rates with transitions')
    ax[0].imshow(true_r,aspect='auto',origin='lower',vmin=0,vmax=1)
    ax[1].plot(component_inds, np.arange(n_trials),'-o',label='Actual', alpha = 0.5)
    ax[1].plot(plot_inferred_inds, np.arange(n_trials),'-x',label='Inferred', alpha = 0.5)
    ax[1].set_title(f'Accuracy : {cat_accuracy}/{categorical_w.shape[0]}')
    ax[1].legend()
    for trial,val in enumerate(mode_tau):
        ax[0].vlines(val, trial-0.5,trial+0.5, linewidth = 5, color='red')
    ax[2].imshow(mean_w,aspect='auto',origin='lower');
    ax[3].imshow(plot_components,aspect='auto',origin='lower');
    ax[3].set_title('Actual Mixture Components')
    ax[3].set_xlabel('Mixture #')
    ax[2].set_title('Mixture weights')
    ax[1].set_xlabel('Mixture #')
    ax[0].set_ylabel('Trial #')
    ax[2].set_xlabel('Mixture #')
    plt.tight_layout()

    return flip_emissions

def gen_bernoulli_emissions_plot(
    trace,
    n_states,
    n_components,
    flip_emissions,
    true_lambda = None,
    figsize = (7,3),
):
    if true_lambda is not None:
        fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize = figsize)
    else:
        fig, ax = plt.subplots(1,1, 
                               sharex=True, sharey=True, 
                               figsize = (figsize[0]/2, figsize[1]))
        ax = [ax]
        
    state_names = np.stack(
            [
                [f'Comp{i}_State{j}' for j in range(n_states)] \
                        for i in range(n_components)
                        ]
            )

    emission_values = trace['posterior']['lambdas'].values
    if flip_emissions:
        emission_values = emission_values[:,:,::-1]
    mean_emissions = emission_values.mean(axis=(0,1))
    std_emissions = emission_values.std(axis=(0,1))

    if true_lambda is not None:
        ax[0].bar(state_names.flatten(), true_lambda.flatten());
        ax[0].set_xticks(
                np.arange(state_names.size), 
                state_names.flatten(), rotation = 45, 
                ha='right', rotation_mode='anchor');
        ax[0].set_title('Actual Emissions')
        ax[0].set_xlabel('Variable Name');
        ax[0].set_ylabel('Emission Rate')
    
    ax[-1].errorbar(state_names.flatten(), mean_emissions.flatten(), 
                   yerr = std_emissions.flatten(), fmt = 'o', c = 'k',
                   label = 'St.Dev.');
    ax[-1].legend()
    ax[-1].bar(state_names.flatten(), mean_emissions.flatten());
    ax[-1].set_title('Inferred Emissions');
    ax[-1].set_xlabel('Variable Name');
    ax[-1].set_xticks(
            np.arange(state_names.size), state_names.flatten(), 
            rotation = 45, ha='right', rotation_mode='anchor');

def gen_dirichlet_plots(
		dpp_trace,
		true_r,
		n_chains,
		length,
		max_states,
		n_states = None,
		dur_thresh = 0.01,
		figsize = (7,15),
		):

	w_latent_samples = dpp_trace['posterior']['w_latent'].values
	cat_w_latent_samples = np.concatenate(w_latent_samples)
	sorted_lens = np.sort(cat_w_latent_samples,axis=-1)[:,::-1]

	sorted_w_latent = np.stack(np.array_split(np.sort(w_latent_samples,axis=-1)[...,::-1],n_chains,axis=0))
	sorted_w_latent = np.squeeze(sorted_w_latent)
	mean_sorted = np.mean(sorted_w_latent, axis = 1)

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

	if n_states is not None:
		ax[0].axvline(n_states-1, zorder = -1, color = 'black', label = 'Actual')
		ax[0].legend()
	ax[0].axhline(dur_thresh, color = 'red', linestyle = '--')
	ax[0].text(max_states//2, dur_thresh*1.5, f'Duration Threshold = {dur_thresh}')
	ax[0].set_xlabel('State #')
	ax[0].set_ylabel('Fractional Duration')
	ax[0].set_title('Inferred Durations of States')

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

	max_state_per_chain = state_frame.loc[state_frame.dur > dur_thresh].groupby('chains').max()
	max_state_counts = max_state_per_chain.groupby('states').count()
	state_vec = np.arange(1,max_states+1)
	counts = [max_state_counts.loc[x].values[0] if x in max_state_counts.index else 0 for x in state_vec ]
	if n_states is not None:
		ax[4].axvline(n_states, zorder = 2, color = 'black', label = 'Actual')
	ax[4].bar(state_vec, counts, label = 'Inferred')
	ax[4].set_xlabel("States")
	ax[4].set_ylabel('Count')
	ax[4].set_title('Number of states')
	ax[4].legend()

	plt.tight_layout()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

colors = sns.color_palette("bright")

def scatter_country_performance(metrics_by_country, 
                                country_names,
                                plot_key,
                                metric_indexes_per_result= [1,2],
                                name_per_result = ['corrected with true rural',
                                                   'corrected with predicted rural'],
                                title='TITLE ME',
                                legend=True,
                                ax=None):
    
    targeting_metric = 'alloc' in plot_key #in ['group_alloc_diff','group_alloc_proxy']
    bias_metric = plot_key in ['bias_urban','bias_rural']
     
    n = len(metrics_by_country[0]['r2_all'][0])
    std_to_errbar = 2.0 / np.sqrt(n)
    mins, maxs = 1,0
    
    if ax is None:
        fig, ax = plt.subplots()
        
    for c, metrics in enumerate(metrics_by_country):
     #   if country_names[c] == 'dhs/colombia' and plot_key == 'r2_all' and spatial: 
       #        continue

        msize = 60
        if country_names[c] in ["us", "mexico"]: mstyle = 's'
        else: mstyle = 'o'
            
        # can have na in recall
        not_na_standard = ~np.isnan(np.array(metrics[plot_key][0]))
        perf_standard = np.array(metrics[plot_key][0])[not_na_standard]
        perf_by_method = []
        for metric_idx in metric_indexes_per_result:
            
            # can have na in recall
            not_na = ~np.isnan(np.array(metrics[plot_key][metric_idx]))
            perf_by_method.append(np.array(metrics[plot_key][metric_idx])[not_na])
            
        for m,metric_idx in enumerate(metric_indexes_per_result):
            if m == 0: 
                facecolor = colors[c]
                label= country_names[c].split('/')[-1]
            else:
                facecolor= 'white'
                label=None
                                
            # corrected by strategy 2
            ax.scatter(perf_standard.mean(),
                       perf_by_method[m].mean(), 
                       color=facecolor,
                       edgecolor = colors[c],
                       s = msize,
                       marker = mstyle,
                       label=label,
                       zorder=10)

            ax.errorbar(perf_standard.mean(), 
                        perf_by_method[m].mean(), 
                        perf_standard.std() * std_to_errbar, 
                        perf_by_method[m].std() * std_to_errbar,  
                        alpha=0.4,
                        color = colors[c])

            mins = np.min([mins, 
                           perf_standard.mean(), 
                           perf_by_method[m].mean()])

            maxs = np.max([maxs, 
                           perf_standard.mean(), 
                           perf_by_method[m].mean()
                        #   perf_corrected_true_rural.mean(), 
                       #    perf_corrected_pred_rural.mean()
                          ])

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    ax.scatter([0], [0],
                color='black',
                label=f'{name_per_result[0]}')
    if len(metric_indexes_per_result) >1:
        ax.scatter([0], [0],
                    color='white',
                    edgecolor='black',
                    label=f'{name_per_result[1]}')

  #  print(mins, maxs)
    ax.plot([mins,maxs],[mins,maxs], label='y=x', color='lightgrey')
    if targeting_metric: ax.plot([mins,maxs],[-mins,-maxs], label='y=-x', color='lightgrey')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
   # ax.set_ylim([np.minimum(np.maximum(x,0),1) for x in ylim])
    
    if targeting_metric: ax_descriptor = 'difference in allocation (to rural)'
    elif bias_metric: ax_descriptor = 'bias'
    else: ax_descriptor = 'performance'
    ax.set_xlabel(f'{ax_descriptor} of uncorrected predictions')
    ax.set_ylabel(f'{ax_descriptor} of "corrected" predictions')
    if legend: plt.legend(loc='lower right')
    
    ax.set_title(f'{title}')
    ax.set_aspect('equal')
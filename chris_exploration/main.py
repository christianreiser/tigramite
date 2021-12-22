import numpy as np
import pandas as pd
import scipy.optimize
from matplotlib import pyplot as plt

from chris_exploration.preprocessing import remove_nan_seq_from_top_and_bot, non_contemporary_tie_series_generation
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI


def parabola(x, a, b, c):
    x = np.array(x)
    return a * x ** 2 + b * x + c


def arg_closest(lst,x):
    lst = np.subtract(lst, x)
    return np.where(lst == min(lst, key=abs))[0][0]


def reduce_tau_max(correlations):
    # 3d-> 2D via reshape, 2D->1D via amax, abs
    abs_max_corr_coeff = np.absolute(np.amax(correlations.reshape(df.shape[1] ** 2, -1), axis=0))

    abs_max_corr_coeff = np.delete(abs_max_corr_coeff, 0)    # remove idx 0. idk what it was for
    time_lag = list(range(0, len(abs_max_corr_coeff)))  # array of time lags
    parabola_params, _ = scipy.optimize.curve_fit(parabola, time_lag, abs_max_corr_coeff) # parabola_params
    y_parabola = parabola(time_lag, *parabola_params) # y values of fitted parabola
    parabola_first_half = y_parabola[:np.argmin(y_parabola)] # keep only part of parabola which is before argmin
    tau_max = arg_closest(parabola_first_half, corr_threshold)

    # plotting
    plt.plot(abs_max_corr_coeff, label='max correlation coefficient')
    plt.plot(time_lag, y_parabola, label='quadratic fit')
    # plt.axhline(y=corr_threshold, label='corr_threshold')
    plt.axvline(tau_max, 0, 30, label='tau_max')
    plt.fill_between([0,len(abs_max_corr_coeff)], 0, corr_threshold,
                    facecolor='red', alpha=0.3,label='below corr threshold')
    plt.title('Computation of tau_max='+str(tau_max))
    plt.ylabel('max correlation coefficient')
    plt.ylabel('time lag')
    plt.xlim([0, len(abs_max_corr_coeff)])
    plt.ylim([0, max(abs_max_corr_coeff)])
    plt.legend(loc='best')
    plt.show()
    return tau_max

df = pd.read_csv('./data/daily_summaries_all (copy).csv', sep=",")
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# select columns
df = df[['Date', 'Mood', 'HumidInMax()']]

df = remove_nan_seq_from_top_and_bot(df)
# df = non_contemporary_tie_series_generation(df)
df = df.drop(['Date'], axis=1)  # drop date col
#
# # standardize data
df -= df.mean(axis=0)
df /= df.std(axis=0)

tau_max = 32
alpha_level = 0.01
corr_threshold = 0.07
verbosity = 0
var_names = df.columns
dataframe = pp.DataFrame(df.values, datatime=np.arange(len(df)),
                         var_names=var_names)

# tp.plot_timeseries(dataframe)
# plt.show()

parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
    verbosity=verbosity)

correlations = pcmci.get_lagged_dependencies(tau_max=tau_max, val_only=True)['val_matrix']
lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names': var_names,                                                             'x_base': 5, 'y_base': .5})
plt.show()

tau_max = reduce_tau_max(correlations)



pcmci.verbosity = verbosity
results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)

q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=tau_max, fdr_method='fdr_bh')
pcmci.print_significant_links(
    p_matrix=results['p_matrix'],
    q_matrix=q_matrix,
    val_matrix=results['val_matrix'],
    alpha_level=alpha_level)

link_matrix = pcmci.return_significant_links(pq_matrix=q_matrix,
                                             val_matrix=results['val_matrix'], alpha_level=alpha_level)['link_matrix']
tp.plot_graph(
    val_matrix=results['val_matrix'],
    link_matrix=link_matrix,
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    figsize=(10, 6),
)
plt.show()

# Plot time series graph
tp.plot_time_series_graph(
    figsize=(12, 8),
    val_matrix=results['val_matrix'],
    link_matrix=link_matrix,
    var_names=var_names,
    link_colorbar_label='MCI',
)
plt.show()

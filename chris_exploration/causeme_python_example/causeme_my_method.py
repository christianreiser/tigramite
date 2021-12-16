"""
This file must contain a function called my_method that triggers all the steps 
required in order to obtain

 *val_matrix: mandatory, (N, N) matrix of scores for links
 *p_matrix: optional, (N, N) matrix of p-values for links; if not available, 
            None must be returned
 *lag_matrix: optional, (N, N) matrix of time lags for links; if not available, 
              None must be returned

Zip this file (together with other necessary files if you have further handmade 
packages) to upload as a code.zip. You do NOT need to upload files for packages 
that can be imported via pip or conda repositories. Once you upload your code, 
we are able to validate results including runtime estimates on the same machine.
These results are then marked as "Validated" and users can use filters to only 
show validated results.

Shown here is a vector-autoregressive model estimator as a simple method.
"""

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
# Your method must be called 'my_method'
# Describe all parameters (except for 'data') in the method registration on CauseMe
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI


def my_method(data, maxlags, correct_pvalues=True):
    data = pd.DataFrame(data)
    var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']

    # Input data is of shape (time, variables)
    T, N = data.shape

    # Standardize data
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    data = pp.DataFrame(data.values, datatime=np.arange(len(data)),
                        var_names=var_names)

    # Fit VAR model and get coefficients and p-values
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(
        dataframe=data,
        cond_ind_test=parcorr,
        verbosity=0)
    correlations = pcmci.get_lagged_dependencies(tau_max=maxlags, val_only=True)['val_matrix']
    results = pcmci.run_pcmci(tau_max=maxlags, pc_alpha=None)
    pvalues = results['p_matrix']
    values = results['val_matrix']

    # CauseMe requires to upload a score matrix and
    # optionally a matrix of p-values and time lags where
    # the links occur

    # In val_matrix an entry [i, j] denotes the score for the link i --> j and
    # must be a non-negative real number with higher values denoting a higher
    # confidence for a link.
    # Fitting a VAR model results in several lagged coefficients for a
    # dependency of j on i.
    # Here we pick the absolute value of the coefficient corresponding to the
    # lag with the smallest p-value.
    val_matrix = np.zeros((N, N), dtype='float32')

    # Matrix of p-values
    p_matrix = np.ones((N, N), dtype='float32')

    # Matrix of time lags
    lag_matrix = np.zeros((N, N), dtype='uint8')

    for j in range(N):
        for i in range(N):
            # Store only values at lag with minimum p-value
            tmp2 = np.arange(1, maxlags + 1)
            tmp3 = (tmp2 - 1) * N + i
            tmp1 = pvalues[i, j, :]
            tau_min_pval = np.argmin(tmp1)
            p_matrix[i, j] = pvalues[i, j, tau_min_pval - 1]

            # Store absolute coefficient value as score
            tmp4 = values[j, i, tau_min_pval - 1]
            val_matrix[i, j] = np.abs(tmp4)

            # Store lag
            lag_matrix[i, j] = tau_min_pval

    # Optionally adjust p-values since we took the minimum over all lags 
    # [1..maxlags] for each i-->j; should lead to an expected false positive
    # rate of 0.05 when thresholding the (N, N) p-value matrix at alpha=0.05
    # You can, of course, use different ways or none. This will only affect
    # evaluation metrics that are based on the p-values, see Details on CauseMe
    if correct_pvalues:
        p_matrix *= float(maxlags)
        p_matrix[p_matrix > 1.] = 1.

    return val_matrix, p_matrix, lag_matrix

import numpy as np
import pandas as pd
from scipy import stats


def NPC(X, y, n_perm=1000, method='Fisher', return_report=False):

    # 1) Test each hypothesis separately using permutations that are performed synchronously across features
    # 2) Statistics for each permutation are stored. Provides a null dist for each one.
    # 3) Calculate global statistics for each permutation and compare to real data for final statistic

    # X: dataframe with samples as rows and features as columns
    # y: group labels (only for two group comparisons)
    # n_trails: number of permutations

    # Convert data types
    if isinstance(y, pd.Series):
        y = y.values
    if y is dict:
        y = np.array([y[i] for i in y])
    if y is list:
        y = np.array(y)

    # label names
    labels = np.unique(y)

    # Real data statistics
    pvals = pd.Series(index=X.columns, dtype='float')

    group1, group2 = (y == labels[0]).tolist(), (y == labels[1]).tolist()
    for column_name in X.columns:
        t, p = stats.mannwhitneyu(X.loc[group1, column_name], X.loc[group2, column_name],
                                  alternative='two-sided')
        pvals.loc[column_name] = p

    # Permutation statistics
    pvals_perm = pd.DataFrame(index=range(n_perm), columns=X.columns, dtype='float')

    for i in range(n_perm):

        perm_i = np.random.permutation(y)
        group1, group2 = (perm_i == labels[0]).tolist(), (perm_i == labels[1]).tolist()

        for column_name in X.columns:

            t_perm, p_perm = stats.mannwhitneyu(X.loc[group1, column_name], X.loc[group2, column_name],
                                                alternative='two-sided')
            pvals_perm.loc[i, column_name] = p_perm

    # Calculate global statistics
    if method == 'Fisher':
        global_stat = -2*(np.log(pvals).sum())
        perm_global_stats = -2*(np.log(pvals_perm).sum(axis=1))
        # global_pvalue = 1 - stats.chi2.cdf(global_stat, len(X.columns))
        # perm_global_pvalue = 1 - stats.chi2.cdf(perm_global_stats, len(X.columns))
        # final_pvalue = sum([(itr < global_pvalue)*1 for itr in perm_global_pvalue.tolist()]) / n_perm
        global_pvalue = sum([(itr >= global_stat) * 1 for itr in perm_global_stats]) / n_perm

    if method == 'Zaykin':
        global_stat = Zaykin(pvals, 0.05)
        perm_global_stats = [Zaykin(pvals_perm.iloc[itr, :], 0.05) for itr in range(pvals_perm.shape[0])]
        global_pvalue = sum([(itr >= global_stat) * 1 for itr in perm_global_stats]) / n_perm

    if method == 'Tippett':
        global_stat = pvals.min()
        perm_global_stats = pvals_perm.min(axis=1)
        # global_pvalue = 1 - ((1 - global_stat) ** len(X.columns))
        # perm_global_pvalue = [1 - ((1 - itr) ** len(X.columns)) for itr in perm_global_stats]
        # final_pvalue = sum([(itr < global_pvalue) * 1 for itr in perm_global_pvalue]) / n_perm
        global_pvalue = sum([(itr <= global_stat) * 1 for itr in perm_global_stats]) / n_perm

    if return_report:
        output = global_pvalue, global_stat, perm_global_stats
    else:
        output = global_pvalue

    return output


def Zaykin(pvalues, alpha):
    indicator_function = (pvalues <= alpha)*1
    t = np.product(pvalues ** pvalues)
    return t

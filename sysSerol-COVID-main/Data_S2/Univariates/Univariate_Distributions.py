from Import import*
from Univariate_Statistics import *

# df defined and statistics defined in Univariate_Statistics.py

# Aesthetic Setup
pal = ['#0276BA', '#F9BA00', '#99185F']
hue_order = ['Moderate', 'Severe', 'Deceased']


# p value plot function
def pvalstr(pvalue):
    thresholds = 0.5 * np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    position = (pvalue < thresholds).sum()
    if position == 0:
        pvaluestr = 'ns'
    else:
        pvaluestr = '*' * position
    return pvaluestr


########################################################################################################################
# Luminex
########################################################################################################################
signif_ind = 0
i = 1
rng1 = list(range(0, 66, 6))
rng2 = list(range(66, 82, 3))

for r in range(len(rng1)):

    # Figure
    sns.set_style("whitegrid",
                  {'axes.linewidth': 1, 'axes.edgecolor': 'black', 'xtick.bottom': True, 'xtick.top': False,
                   'ytick.left': True, 'ytick.right': False, })
    sns.set_context("paper")
    f, axss = plt.subplots(1, 6, figsize=(9, 1.5), dpi=user_dpi, constrained_layout=True, sharey=True, sharex=True)

    for enum, axs in enumerate(axss):

        ################################################
        # Plotting
        ################################################
        sns.violinplot(x=df['week'], y=df.columns[i], ax=axs, hue=df['group'], data=df, linewidth=0.25, palette=pal,
                       hue_order=hue_order, scale='width', cut=0, inner="quartile", bw=0.2)

        for l in axs.lines:
            l.set_linestyle('--')
            l.set_linewidth(0.6)
            l.set_color('w')
            l.set_alpha(0.5)
        for l in axs.lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1.3)
            l.set_color('w')
            l.set_alpha(0.7)

        groups = [['Moderate', 'Severe'], ['Severe', 'Deceased'], ['Moderate', 'Deceased']]
        xloc = [[-0.25, 0], [0, 0.25], [-0.25, 0.25]]
        yloc = [3.1, 3.25, 3.4]

        # Plotting statistical significance
        for ra in range(3):
            dd = df.loc[df['week'] == ra+1, df.columns[i]]
            if reject[signif_ind]:
                for q in range(3):
                    group1 = dd.loc[df['group'] == groups[q][0]].values
                    group2 = dd.loc[df['group'] == groups[q][1]].values
                    t, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                    if p <= 0.05:
                        x1 = xloc[q][0] + ra
                        x2 = xloc[q][1] + ra
                        dx = (x2-x1)/2
                        mid = x1 + dx
                        y = yloc[q]
                        string = pvalstr(p)

                        axs.plot([x1, x2], [y, y], 'k', lw=0.5)
                        axs.text(mid, y+0.05, string, va='center', ha='center', fontsize=6)
            signif_ind += 1

        # Axis settings
        title = df.columns[i].replace('_', ' ')
        axs.set_title(title, fontsize=6.5, ha='center', fontweight="bold")

        if enum == 0:
            axs.set_ylabel('log' + r'$_{10}$' + ' MFI', fontsize=6)
        else:
            axs.set_ylabel(None)

        axs.grid(axis="x", lw=0)
        axs.grid(axis="y", lw=0)
        axs.get_legend().remove()

        axs.set_yticks(np.arange(0, 6, 1))
        axs.set_xticklabels(['1', '2', '>3'], rotation=0, fontweight="normal", fontsize=6)
        axs.set_xlabel('Week', fontsize=6)
        axs.set_ylim(-0.05, 3.55)
        axs.tick_params(axis="both", direction='out', labelsize=6, width=0.5, length=2)

        sns.despine(ax=axs, top=True, bottom=False, left=False, right=True)
        [q.set_linewidth(0.5) for q in axs.spines.values()]

        i += 1

########################################################################################################################
# Functions
########################################################################################################################

ylims = [45, 15, 2.4, 25, 50]
ylims = np.repeat(ylims, 3)
ylims = np.append(ylims, 2500)

yticks = [np.arange(0, 54, 9), np.arange(0, 18, 3),  np.arange(0, 3, 0.6),  np.arange(0, 30, 5),  np.arange(0, 60, 10)]
yticks = [[itr]*3 for itr in yticks]
yticks = [item for sublist in yticks for item in sublist]
yticks = yticks + [np.arange(0, 3000, 500)]

for r in range(len(rng2)-1):

    # Figure
    sns.set_style("whitegrid",
                  {'axes.linewidth': 1, 'axes.edgecolor': 'black', 'xtick.bottom': True, 'xtick.top': False,
                   'ytick.left': True, 'ytick.right': False, })
    sns.set_context("paper")
    f, axss = plt.subplots(1, 3, figsize=(4.5, 1.5), dpi=user_dpi, constrained_layout=True, sharey=True, sharex=True)

    for enum, axs in enumerate(axss):

        ################################################
        # Plotting
        ################################################

        sns.violinplot(x=df['week'], y=df.columns[i], ax=axs, hue=df['group'], data=df, linewidth=0.25, palette=pal,
                       hue_order=hue_order, scale='width', cut=0, inner="quartile", bw=0.2)

        for l in axs.lines:
            l.set_linestyle('--')
            l.set_linewidth(0.6)
            l.set_color('w')
            l.set_alpha(0.5)
        for l in axs.lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1.3)
            l.set_color('w')
            l.set_alpha(0.7)

        # Plotting statistical significance
        groups = [['Moderate', 'Severe'], ['Severe', 'Deceased'], ['Moderate', 'Deceased']]
        xloc = [[-0.25, 0], [0, 0.25], [-0.25, 0.25]]

        # Plotting statistical significance
        for ra in range(3):
            dd = df.loc[df['week'] == ra+1, df.columns[i]]
            yi_max = dd.max()*1.03
            yi_intv = yi_max * 0.03
            y = [yi_max + (yi_intv * itr) for itr in range(1, 4, 1)]

            if reject[signif_ind]:
                for q in range(3):
                    group1 = dd.loc[df['group'] == groups[q][0]].values
                    group2 = dd.loc[df['group'] == groups[q][1]].values
                    t, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                    if p <= 0.05:

                        x1 = xloc[q][0] + ra
                        x2 = xloc[q][1] + ra
                        dx = (x2-x1)/2
                        mid = x1 + dx

                        string = pvalstr(p)

                        axs.plot([x1, x2], [y[q], y[q]], 'k', lw=0.5)
                        axs.text(mid, y[q], string, va='center', ha='center', fontsize=6)
            signif_ind += 1

        # Axis settings
        title = df.columns[i].replace('_', ' ')
        axs.set_title(title, fontsize=6.5, ha='center', fontweight="bold")

        if enum == 0:
            axs.set_ylabel("Score", fontsize=6)
        else:
            axs.set_ylabel(None)

        axs.grid(axis="x", lw=0)
        axs.grid(axis="y", lw=0)
        axs.get_legend().remove()

        axs.set_yticks(yticks[i-67])
        axs.set_ylim(-0.05, ylims[i-67])
        axs.tick_params(axis="both", direction='out', labelsize=6, width=0.5, length=2)
        axs.set_xticklabels(['1', '2', '>3'], rotation=0, fontweight="normal", fontsize=6)
        axs.set_xlabel('Week', fontsize=6)
        axs.tick_params(axis="x", direction='out', labelsize=6)

        sns.despine(ax=axs, top=True, bottom=False, left=False, right=True)
        [q.set_linewidth(0.5) for q in axs.spines.values()]

        i += 1





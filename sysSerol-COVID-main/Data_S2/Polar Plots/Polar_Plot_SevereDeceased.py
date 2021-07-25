from Import import *
from matplotlib.colors import Normalize
from Non_Parametric_Combination import *

# Aesthetic and plotting setup
hue_order = ['Severe', 'Deceased']
titles = ['Week 1', 'Week 2', 'Week > 3']

df = pd.concat([df_RawData, df_Info.loc[:, 'days'], df_Info.loc[:, 'group'], df_Info.loc[:, 'patient_ID']], axis=1)

# RNA Negatives
df_negatives = df[df['group'] == 'Negative']
df_negatives = df_negatives.iloc[:, :-3].reset_index(drop=True)

negatives_mean = df_negatives.mean()
negative_std = df_negatives.std()

df.loc[:, negatives_mean.index] = df.loc[:, negatives_mean.index] - negatives_mean

for i in df.columns[:-3]:
    df.loc[df.loc[:, i] < 0, i] = 0

# Shape and define weeks
df = df.loc[df['group'].isin(['Severe', 'Deceased']), :]

df.loc[:, 'week'] = np.ones(df.shape[0])
df.loc[((7 < df['days']) & (df['days'] < 15)), 'week'] = 2
df.loc[df['days'] > 14, 'week'] = 3

# rank percentiles across time intervals and groups
din = df.loc[:, df.columns[:82]].copy()
df.iloc[:, :82] = din.rank(pct=True, method='average')
df = df.loc[:, df.columns != 'days']

#############################################################
# region: Polar Plots / Flower Plots / Rose Nightingale Plots
#############################################################
signif_idx = 0
for i in range(0, 3, 1):

    # Set with correct time period
    X = df.loc[df['week'] == i+1, :]

    lut = dict(zip(df['patient_ID'], df['group']))

    # Average patient with multiple time points
    X = X.groupby(by='patient_ID').mean().reset_index()
    X.loc[:, 'group'] = X['patient_ID'].map(lut)
    X = X.loc[:, ~X.columns.str.contains('patient_ID|week|NT50')]

    # Calculate group mean
    xinput = X.groupby(by='group').mean()

    ######################################
    # Plotting
    ######################################
    sns.set_style("darkgrid",
                  {'axes.linewidth': 2, 'axes.edgecolor': 'black', 'xtick.bottom': True, 'xtick.top': False,
                   'ytick.left': True, 'ytick.right': False})
    # sns.set_context("paper")
    f, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=user_dpi, constrained_layout=True, subplot_kw=dict(polar=True))
    f.suptitle(titles[i], fontsize=10)

    for c in range(2):
        x_case = xinput.iloc[c, :].T

        rng = list(range(0, 66, 6)) + list(range(66, 82, 3))
        inv = np.diff(rng)

        # Define color of bars
        saturation = 1
        hues = sns.husl_palette(22, h=0, s=saturation, l=0.7)[9:15]  # Titers
        hues = hues + sns.husl_palette(22, h=0, s=saturation, l=0.7)[17:]  # FcRS
        hues = hues + sns.husl_palette(40, h=0, s=saturation, l=0.75)[3:8]  # Functions

        cmaps = []
        for enum, co in enumerate(hues):
            cmaps = cmaps + [sns.light_palette(co, input="rgb", as_cmap=True)] * inv[enum]

        # Define polar plot properties
        incr = 0.03
        N = x_case.shape[0]
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        radii = x_case.values
        width = 2 * np.pi / N
        origin = 0
        limit = 1 + incr
        my_norm = Normalize(vmin=origin, vmax=limit)

        ##############################
        # Main response bars
        ##############################
        bars = axs[c].bar(theta, radii, width=width*.80, bottom=origin, alpha=1, linewidth=0)
        axs[c].bar(theta, [0.004]*len(theta), width=width*.80, bottom=radii, alpha=1, linewidth=0,
                   facecolor='w', edgecolor='w')

        # Use custom colors and opacity
        for index, (r, bar) in enumerate(zip(radii, bars)):
            cmap_tmp = cmaps[index]
            bar.set_facecolor(cmap_tmp(my_norm(1)))
            bar.set_edgecolor(cmap_tmp(my_norm(1)))

        ##############################
        # Segment legend bars
        ##############################
        theta_group_bar = [theta[itr] - (width / 2) for itr in range(6, 66, 6)] + \
                          [theta[itr] - (width / 2) for itr in range(66, 81, 3)] + [np.pi * 2 - (width / 2)]
        width_group = np.array([width * itr for itr in inv])

        theta_group_in = np.array(theta_group_bar) - (width_group / 2)
        legend_bars = axs[c].bar(theta_group_in, [limit]*len(theta_group_in), width=width_group*0.95,
                                 bottom=1, alpha=1, linewidth=0, facecolor='#CCD0D9', edgecolor='w')

        # Use custom colors and opacity
        for index, bar in enumerate(legend_bars):
            cmap_tmp = cmaps[rng[index]]
            bar.set_facecolor(cmap_tmp(my_norm(1)))
            bar.set_edgecolor(cmap_tmp(my_norm(1)))

        ##############################
        # Axis settings
        ##############################
        axs[c].set_title(x_case.name, fontsize=10, pad=10)
        axs[c].set_rorigin(origin)
        axs[c].set_ylim(origin, limit)
        axs[c].set_rticks(np.arange(origin, limit + 0.25 - incr, 0.25))
        axs[c].tick_params(axis="y", labelsize=5)

        axs[c].set_rlabel_position(0)
        axs[c].set_theta_zero_location('N')
        axs[c].set_theta_direction(-1)

        theta_group = [theta[itr] - (width / 2) for itr in range(6, 66, 6)] + \
                      [theta[itr] - (width / 2) for itr in range(66, 81, 3)] + [np.pi*2 - (width / 2)]
        axs[c].set_xticks(theta_group)
        axs[c].set_xticks([i + (width / 2) for i in theta], minor=True)

        axs[c].set_xticklabels(range(1, 17, 1))
        axs[c].set_yticklabels([])
        axs[c].tick_params(axis='both', pad=-5)

        linecolor = 'w'
        axs[c].grid(axis='y', which='major', alpha=1, lw=0.5, ls='-', color=linecolor)
        axs[c].grid(axis='x', which='major', alpha=1, lw=0.5, ls='-', color=linecolor)
        axs[c].grid(axis='x', which='minor', alpha=0.5, lw=0.20, ls='-', color=linecolor)

        axs[c].spines['polar'].set(alpha=0.5, ec=linecolor, lw=0.5)
        axs[c].spines['inner'].set(alpha=0.5, ec=linecolor, lw=0.5)
        axs[c].set_facecolor('#CCD0D9')
# endregion

################################################################
# LABELS
################################################################

column_names = df.columns.tolist()

func_names = [itr.split('_')[0] for itr in column_names]
indexes = np.unique(func_names, return_index=True)[1]
func_names = [func_names[index] for index in sorted(indexes)]
func_names = func_names[:14] + ['ADNKA CD107a', 'ADNKA MIP1b']

f, axs = plt.subplots(figsize=(9, 2), dpi=user_dpi, constrained_layout=True)
sns.set_style("white")
sns.set_context("paper")
legend_elements = [Line2D([0], [0], color=hues[i], marker='s', markersize=8,
                          lw=0, label=(str(i + 1) + ' ' + func_names[i])) for i in range(len(func_names))]

list1 = list(range(0, 8, 1))
list2 = list(range(8, 16, 1))
[list1.insert(i, list2[index]) for index, i in enumerate(range(1, 16, 2))]
legend_elements = [legend_elements[i] for i in list1]
axs.legend(handles=legend_elements, loc='center', fontsize=7, ncol=8)
f.patch.set_visible(False)
axs.axis('off')

###################################################################
# Flower Statistics Weighted Z-test Features Grouped Non-Parametric
###################################################################

pz_values = []
dfvalues = pd.concat([din, df.loc[:, df.columns[-3:]]], axis=1)
for i in range(0, 3, 1):

    # Set with correct time period
    X = df.loc[df['week'] == i+1, :]

    lut = dict(zip(df['patient_ID'], df['group']))

    # Average patient with multiple time points
    X = X.groupby(by='patient_ID').mean().reset_index()
    y = X['patient_ID'].map(lut)
    X = X.loc[:, ~X.columns.str.contains('patient_ID|week|NT50')]

    rng = [0, 18, 30, 36, 66, 82]
    for p in range(len(rng)-1):
        X_sub = X.iloc[:, rng[p]:rng[p+1]]
        pz_values.append(NPC(X_sub, y, n_perm=1000, method='Fisher'))

# Bonferroni Correction for multiple hypothesis
rngi = range(0, 20, 5)
reject = []
for i in range(len(rngi)-1):
    res_tup = multitest.multipletests(pz_values[rngi[i]:rngi[i+1]], alpha=0.1, method='fdr_bh')
    reject = reject + res_tup[0].tolist()
    corrected_pvalues = res_tup[1]

new_reject = [(pz_values[i] <= 0.05) * reject[i] for i in range(len(pz_values))]

# plotting results
df_pz = pd.DataFrame(index=range(len(pz_values)), columns=['Interval', 'Feature', 'Pz_value'])
df_pz.loc[:, 'Interval'] = np.repeat(titles, len(['IgG', 'IgA', 'IgM', 'FcRs', 'Functions']))
df_pz.loc[:, 'Feature'] = np.tile(['IgG', 'IgA', 'IgM', 'FcRs', 'Functions'], 3)
df_pz.loc[:, 'Pz_value'] = ''

for i, itr in enumerate(pz_values):
    if itr > 0.05:
        df_pz.iloc[i, 2] = ''
    if itr < 0.05:
        df_pz.iloc[i, 2] = '*'

df_pz_values = df_pz.copy()
df_pz_values.iloc[:, :] = 0.3

sns.set_style("white",
              {'axes.linewidth': 2, 'axes.edgecolor': 'black', 'xtick.bottom': True, 'xtick.top': False,
               'ytick.left': True, 'ytick.right': False})
sns.set_context("paper")
f, axs = plt.subplots(figsize=(3, 3), dpi=150, constrained_layout=True)
df_pz_values = df_pz_values.astype(float)
sns.heatmap(df_pz_values, vmin=0, vmax=1, ax=axs, cmap='Greys', cbar=False, lw=0.5, square=False,
            cbar_kws={"shrink": .3, "ticks": [-3, 0, 3]},
            annot=df_pz.values, fmt='',
            annot_kws={"size": 10, "ha": 'center', "va": 'center', "color": 'k'})
axs.set_xticks([])
axs.set_yticks([])

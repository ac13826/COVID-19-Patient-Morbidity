from Import import*

df = pd.concat([df_RawData, df_Info.loc[:, 'days'], df_Info.loc[:, 'group'], df_Info.loc[:, 'patient_ID']], axis=1)

############################
# Corrections
############################

idx = df.columns.str.contains('ADCP|ADNP')
df.iloc[:, idx] = df.iloc[:, idx] / 1e3

idx = df.columns.str.contains('ADCD')
df.iloc[:, idx] = np.log10(df.iloc[:, idx])

idx = df.columns.str.contains('ADNKA')
df.iloc[:, idx] = df.iloc[:, idx] - df.iloc[:, idx].min()

idx = ~df.columns.str.contains('ADCP|ADNP|ADNKA|ADCD|days|group|patient_ID')
df.iloc[:, idx] = np.log10(df.iloc[:, idx])

# RNA Negatives
df_negatives = df[df['group'] == 'Negative']
df_negatives = df_negatives.iloc[:, :-3].reset_index(drop=True)

negatives_mean = df_negatives.mean()
negative_std = df_negatives.std()

df.loc[:, negatives_mean.index] = df.loc[:, negatives_mean.index] - negatives_mean

for i in df.columns[:-3]:
    df.loc[df.loc[:, i] < 0, i] = 0

# Shape and combining samples
df = df.loc[df['group'].isin(['Moderate', 'Severe', 'Deceased']), :]

df.loc[:, 'week'] = np.ones(df.shape[0])
df.loc[((7 < df['days']) & (df['days'] < 15)), 'week'] = 2
df.loc[df['days'] > 14, 'week'] = 3

lut = dict(zip(df['patient_ID'], df['group']))

df = df.loc[:, df.columns != 'group'].groupby(by=['week', 'patient_ID']).mean().reset_index()
df.loc[:, 'group'] = df['patient_ID'].map(lut)
df = df.loc[:, ~df.columns.str.contains('patient_ID|days')]

########################################################################################################################
# Kruskal Wallis p values
########################################################################################################################

uncorrected_KW_pvalues = []
for i in range(1, 83, 1):

    ################################################
    # Luminex
    ################################################

    for r in range(3):
        dd = df.loc[df['week'] == r+1, df.columns[i]]
        stat, pk = stats.kruskal(dd[df['group'] == 'Moderate'],
                                 dd[df['group'] == 'Severe'],
                                 dd[df['group'] == 'Deceased'])
        uncorrected_KW_pvalues.append(pk)


res_tup = multitest.multipletests(uncorrected_KW_pvalues, alpha=0.05, method='fdr_bh')
reject = res_tup[0]
corrected_KW_pvalues = res_tup[1]

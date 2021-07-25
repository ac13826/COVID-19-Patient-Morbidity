from Import import*
from itertools import permutations, combinations
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import *

# Aesthetic Setup
palet = ['#0276BA', '#F9BA00', '#99185F']
hue_order = [0, 1]

df = pd.concat([df_RawData, df_Info.loc[:, 'days'], df_Info.loc[:, 'group'], df_Info.loc[:, 'patient_ID']], axis=1)

# RNA Negatives
df_negatives = df[df['group'] == 'Negative']
df_negatives = df_negatives.iloc[:, :-3].reset_index(drop=True)

negatives_mean = df_negatives.mean()
negative_std = df_negatives.std()

# Shape and combining samples
df = df.loc[df['group'].isin(['Moderate', 'Severe', 'Deceased']), :]

df.loc[:, 'week'] = np.ones(df.shape[0])
df.loc[((7 < df['days']) & (df['days'] < 15)), 'week'] = 2
df.loc[df['days'] > 14, 'week'] = 3

lut = dict(zip(df['patient_ID'], df['group']))

df = df.loc[:, df.columns != 'group'].groupby(by=['week', 'patient_ID']).mean().reset_index()
df.loc[:, 'group'] = df['patient_ID'].map(lut)
df = df.loc[:, ~df.columns.str.contains('patient_ID|days')]

titles = ['Week 1', 'Week 2']
case_labels = ['Moderate', 'Severe', 'Deceased']

######################################################################
# Loop For Week 1: Severe Deceased and then Week 2: Moderate Severe
######################################################################

group_combination = [[[1, 2]], [[0, 1]]]

for i in range(0, 2):

    # Set with correct time period
    X = df.loc[df['week'] == i+1, :]
    Yu = X['group']
    X = X.loc[:, ~X.columns.str.contains('week|group')]

    combins_i = group_combination[i]

    for r in combins_i:

        title = titles[i] + ' with ' + case_labels[r[0]] + ' & ' + case_labels[r[1]]

        pal = [palet[r[0]], palet[r[1]]]
        rs = [case_labels[itr] for itr in r]

        Y_r = Yu.loc[Yu.isin(rs)]
        X_r = X.loc[Y_r.index, :]

        # Remove columns that are 70% below value
        neg_threshold = negatives_mean + negative_std*1

        Xz = X_r - neg_threshold
        Xz[Xz < 0] = 0

        col_above = neg_threshold[((Xz == 0).sum() / Xz.shape[0]) < 0.7].index.tolist()
        X_r = X_r.loc[:, col_above]

        # Predict Severity
        Y_r = (Y_r == rs[1]) * 1
        _, counts = np.unique(Y_r, return_counts=True)

        X_r = X_r.reset_index(drop=True)
        Y_r = Y_r.reset_index(drop=True)

        ######################################################################
        # Random Forest Feature Selection
        ######################################################################

        nrepeats = 10
        nfolds = 4
        nperms = 50

        # Random Forest Hyperparameters
        n_estimators = [int(x) for x in range(100, 250, 50)]
        max_depth = [int(x) for x in range(50, 350, 50)]
        max_depth.append(None)
        max_features = np.arange(0.01, 0.21, 0.01).tolist()
        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'max_features': max_features}

        # Real data storage
        repeat_space = pd.DataFrame(index=range(0, nrepeats), columns=X_r.columns)
        BalancedAccuracy = []
        Accuracy = []
        confusions = np.empty((2, 2, nrepeats))

        # Shuffled storage
        confusions_shuffle_C = np.empty((2, 2, nperms, nfolds, nrepeats))

        perm_space = pd.DataFrame(index=range(0, nfolds * nrepeats * nperms), columns=X_r.columns)
        perm_space.iloc[:, :] = 0

        y_scores = pd.DataFrame(index=X_r.index, columns=range(nrepeats))

        ind = 0
        start_time = time.time()
        for rep in range(nrepeats):
            # Outer Cross Validation
            skf = StratifiedKFold(n_splits=nfolds, random_state=None, shuffle=True)
            skf_inner = StratifiedKFold(n_splits=3, random_state=None, shuffle=True)

            # real inner storage
            inner_successes = []
            y_lengths = []
            inner_confusion = np.zeros((2, 2))
            fold_space = pd.DataFrame(index=range(0, nfolds), columns=X_r.columns)

            for j, (train_index, test_index) in enumerate(skf.split(X_r, Y_r)):

                X_train, X_test = X_r.loc[train_index, :], X_r.loc[test_index, :]
                y_train, y_test = Y_r.loc[train_index], Y_r.loc[test_index]

                # Hyper parameter optimization for fold balanced accuracy
                rfc = RandomForestClassifier(bootstrap=True, criterion='gini', class_weight="balanced_subsample",
                                             oob_score=True)
                rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, cv=skf_inner,
                                                scoring='balanced_accuracy', n_jobs=-1, iid=False, n_iter=30)
                rfc_random.fit(X_train, y_train)
                best_rfc = rfc_random.best_estimator_

                ################################################
                # REFCV
                ################################################
                rfecv = RFECV(estimator=best_rfc, step=3, cv=skf_inner, scoring='accuracy', n_jobs=-1,
                              min_features_to_select=5)
                selector = rfecv.fit(X_train, y_train)
                model = selector.estimator_
                y_pred = model.predict(X_test.loc[:, selector.support_])
                y_scores.loc[X_test.index, rep] = model.predict_proba(X_test.loc[:, selector.support_])[:, 1]

                inner_confusion = inner_confusion + confusion_matrix(y_test, y_pred)

                fold_space.iloc[j, selector.support_] = 1

                ################################################
                # Permutation Testing
                ################################################
                for p in range(nperms):
                    train_shuffle, test_shuffle = np.random.permutation(train_index), np.random.permutation(test_index)
                    y_train_shuffle, y_test_shuffle = Y_r.loc[train_shuffle], Y_r.loc[test_shuffle]

                    y_train_shuffle.index = y_train.index.values
                    y_test_shuffle.index = y_test.index.values

                    ###############################################
                    # REFCV
                    ###############################################
                    rfecv = RFECV(estimator=best_rfc, step=3, cv=skf_inner, scoring='accuracy', n_jobs=-1,
                                  min_features_to_select=5)
                    selector = rfecv.fit(X_train, y_train_shuffle)
                    model = selector.estimator_
                    y_pred_shuffle = model.predict(X_test.loc[:, selector.support_])

                    confusions_shuffle_C[:, :, p, j, rep] = confusion_matrix(y_test_shuffle, y_pred_shuffle)
                    perm_space.iloc[ind, :] = max(selector.ranking_) - selector.ranking_
                    ind += 1

            # Real Data Fold Saving
            confusions[:, :, rep] = inner_confusion
            BalancedAccuracy.append(np.mean(np.diag(confusions[:, :, rep]) / np.sum(confusions[:, :, rep], axis=1))*100)
            Accuracy.append((np.sum(np.diag(confusions[:, :, rep])) / np.sum(confusions[:, :, rep]))*100)
            repeat_space.iloc[rep, :] = fold_space.sum()

            update_progress('Random Forest Selection for ' + title, rep/(nrepeats-1))
            # print('\nCV Accuracy %0.2f' % np.median(Accuracy))

        ################################################
        # Post Perm Processing
        ################################################

        BalancedAccuracy_shuffle_C = []
        Accuracy_shuffle_C = []
        pvalue_C = []

        confusions_C = np.empty((2, 2, nperms, nrepeats))

        for rep in range(nrepeats):
            count_C = 0
            for p in range(nperms):
                confusion_perm_C = np.sum(confusions_shuffle_C[:, :, p, :, rep], axis=2)

                confusions_C[:, :, p, rep] = confusion_perm_C

                BalancedAccuracy_shuffle_C.append(
                    np.mean(np.diag(confusion_perm_C) / np.sum(confusion_perm_C, axis=1)) * 100)

                Accuracy_shuffle_C.append((np.sum(np.diag(confusion_perm_C)) / np.sum(confusion_perm_C)) * 100)

                count_C += (Accuracy[rep] < (np.sum(np.diag(confusion_perm_C)) / np.sum(confusion_perm_C)) * 100) * 1

            pvalue_C.append(count_C / nperms)

        print('\n Perm CV Accuracy C: ', "%0.2f" % np.median(Accuracy_shuffle_C), ' P: ', "%0.2f" % np.median(pvalue_C))

        export_permtesting = pd.DataFrame(index=range(len(Accuracy_shuffle_C)),
                                          columns=['True Accuracy', 'Perm Accuracy C',
                                                   'True Balanced Accuracy', 'Perm Balanced Accuracy C'])

        ins = [Accuracy, Accuracy_shuffle_C, BalancedAccuracy,
               BalancedAccuracy_shuffle_C]

        for u in range(len(ins)):
            export_permtesting.iloc[:len(ins[u]), u] = ins[u]

        ################################################
        # CV Confusion Matrix
        ################################################
        sns.set()
        sns.axes_style("darkgrid")
        sns.set_context("paper")
        f, axss = plt.subplots(figsize=(1.5, 1.5), dpi=150, constrained_layout=True)
        mean_confusion = np.mean(confusions, axis=2)

        titlez = title + '\n Accuracy: ' + ('%0.2f' % np.mean(Accuracy)) + '\n Balanced Accuracy: ' + \
                 ('%0.2f' % np.mean(BalancedAccuracy))
        axss.set_title(titlez, fontsize=6.5)
        mean_confusion = mean_confusion.astype(float)
        sns.heatmap(mean_confusion, ax=axss, cmap='Greys', cbar_kws={"shrink": .3}, annot=True, lw=0.5, vmin=-30,
                    cbar=False, vmax=100, fmt='0.0f', annot_kws={"size": 8, "ha": 'center', "va": 'center_baseline'})

        ylabs = [case_labels[r[0]], case_labels[r[1]]]
        xlabs = [ylabs[itr] + '\n(n=' + str(counts[itr]) + ')' for itr in range(2)]

        axss.set_yticklabels(ylabs, ha='right', va='center', fontweight='normal')
        axss.set_xticklabels(xlabs, ha='center', va='top', fontweight='normal')
        axss.tick_params(axis="both", length=2, labelsize=6)

        ################################################
        # Plot Top Rankings
        ################################################
        sns.set()
        sns.set_style("white",
                      {'axes.linewidth': 2, 'axes.edgecolor': 'black', 'xtick.bottom': True, 'xtick.top': False,
                       'ytick.left': True, 'ytick.right': False})
        sns.set_context("paper")
        f, axss = plt.subplots(figsize=(3, 2), dpi=150, constrained_layout=True)

        nfeat = 15
        xvec = np.arange(0, nfeat)

        chosen_scoressums2 = repeat_space.sum().sort_values(ascending=False) / (nfolds*nrepeats)
        chosen_scoressums_level = chosen_scoressums2.mean()

        chosen = chosen_scoressums2.iloc[:nfeat]*100

        direction = pd.concat([X_r, Y_r], axis=1).groupby('group').median().idxmax()
        direction = direction.loc[chosen.index]

        lut = dict(zip([0, 1], pal))
        facecolors = direction.map(lut).values.tolist()

        axss.bar(xvec, chosen, color=facecolors, tick_label=chosen.index)
        axss.set_xticklabels(chosen.index)
        axss.tick_params(axis='x', direction='out', length=3, labelsize=7, rotation=90)
        axss.tick_params(axis='y', direction='out', length=3, labelsize=6, rotation=0)
        axss.set_xlim(-0.5, nfeat-0.5)
        axss.set_ylim(0, 100)
        sns.despine(top=True, right=True, left=False, bottom=False)
        axss.set_ylabel('% Selected', fontsize=7, labelpad=0)
        axss.set_title('Top Feature Selection', fontsize=8)

        ################################################
        # Roc curves with repetitions
        ################################################
        sns.set()
        sns.set_style("white",
                      {'axes.linewidth': 2, 'axes.edgecolor': 'black', 'xtick.bottom': True, 'xtick.top': False,
                       'ytick.left': True, 'ytick.right': False})
        sns.set_context("paper")
        f, ax = plt.subplots(figsize=(2, 2), dpi=user_dpi, constrained_layout=True)

        Y_real = np.vstack([(Y_r == 0).values*1, (Y_r == 1).values*1]).T

        mean_fpr = np.linspace(0, 1, 200)

        roc_auc = []
        tprs = []
        fprs = []
        for rep in range(nrepeats):
            fpr, tpr, _ = roc_curve(Y_r.values, y_scores.iloc[:, rep].values)

            roc_auc.append(auc(fpr, tpr))
            if rep == nrepeats-1:
                ste = ax.step(fpr, tpr, color='#CFD5E6', lw=1, alpha=1, label='All')
            else:
                ste = ax.step(fpr, tpr, color='#CFD5E6', lw=1, alpha=1)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)

            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)

        mean_tpr[-1] = 1.0

        fpr_prob_average, tpr_prob_average, _ = roc_curve(Y_r.values, y_scores.mean(axis=1))
        mean_auc = auc(fpr_prob_average, tpr_prob_average)
        ax.step(fpr_prob_average, tpr_prob_average, color='orange', lw=2, alpha=1, label='Mean')

        ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=.8)

        ax.grid(axis="both", lw=0)
        ax.set_xticks(np.arange(0, 1.25, 0.25))
        ax.set_yticks(np.arange(0, 1.25, 0.25))
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([0, 1.01])
        ax.set_title(r'Mean AUC: %0.2f' % mean_auc, fontsize=7)
        ax.tick_params(axis='both', direction='in', length=2, width=0.5, labelsize=7)
        ax.set_xlabel('FPR', fontsize=7)
        ax.set_ylabel('TPR', fontsize=7)
        ax.legend(loc='lower right', fontsize=6)
        [ax.spines[i].set_linewidth(0.5) for i in ax.spines]

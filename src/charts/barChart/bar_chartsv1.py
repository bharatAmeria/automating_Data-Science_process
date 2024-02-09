def bar_charts(df, feature, label, roundto=4, p_threshold=0.05):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    """ Create the bar chart """
    sns.barplot(x=df[feature], y=df[label])

    """ Create the numeric lists needed too calculate the ANOVA """
    groups = df[feature].unique()
    groups_list = []
    for g in groups:
        groups_list.append(df[df[feature] == g][label])

    f, p = stats.f_oneway(*groups_list)  # <- same as (group_list[0], group_list[1], ..., group_list[n])

    """ calculate individual pairwise t-test for each pair of groups """
    ttests = []
    for i1, g1 in enumerate(groups):
        for i2, g2 in enumerate(groups):
            if i2 > i1:
                list1 = df[df[feature] == g1][label]
                list2 = df[df[feature] == g2][label]
                t, tp = stats.ttest_ind(list1, list2)
                ttests.append([f'{g1} - {g2}', round(t, roundto), round(tp, roundto)])

    """ make a Bonferroni Correction -> adjust the p-value threshold to be 0.05/ n of ttests """

    bonferroni = p_threshold / len(ttests)
    print(p_threshold, len(ttests))

    """ Create textstr to add statistics to chart """

    textstr = f'F: {round(f, roundto)}\n'
    textstr += f'p: {round(p, roundto)}\n'
    textstr += f'Bonferroni p: {round(bonferroni, roundto)}'
    for ttest in ttests:
        if ttest[2] >= bonferroni:
            textstr += f'\n{ttest[0]}: t:{ttest[1]}, p:{ttest[2]}'

    """ If there are too many feature groups, print x labels vertically """

    if df[feature].nunique() > 7:
        plt.xticks(rotation=90)

    """ Create textstr to add statistics to chart """
    textstr = f'F: {round(f, roundto)}\n'
    textstr += f'p: {round(p, roundto)}'

    plt.text(.95, 0.10, textstr, fontsize=12, transform=plt.gcf().transFigure)
    plt.show()

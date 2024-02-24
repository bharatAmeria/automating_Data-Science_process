def univariate(df):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    output_df = pd.DataFrame(columns=['feature', 'type', 'count', 'missing', 'unique', 'mode', 'min', 'q1', 'median',
                                      'q3', 'max', 'mean', 'std', 'skew', 'kurt'])
    output_df.set_index('feature', inplace=True)

    for col in df:
        dtype = df[col].dtype
        count = df[col].count()
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        mode = df[col].mode()[0]

        if pd.api.types.is_numeric_dtype(df[col]):
            min_ = df[col].min()
            q1 = df[col].quantile(.25)
            median = df[col].median()
            q3 = df[col].quantile(.75)
            max_ = df[col].max()
            mean = df[col].mean()
            std = df[col].std()
            skew = df[col].skew()
            kurt = df[col].kurt()

            output_df.loc[col] = [dtype, count, missing, unique, mode, min_, q1, median,
                                  q3, max_, mean, std, skew, kurt]
            sns.histplot(data=df, x=col)
        else:
            output_df.loc[col] = [dtype, count, missing, unique, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
            sns.countplot(data=df, x=col)

        plt.show()
    return output_df


def scatterplot(df, feature, label, roundto=4, linecolor='orange'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # create the plot
    sns.regplot(x=df[feature], y=df[label], line_kws={"color": linecolor})

    # calculate the regression line
    m, b, r, p, err = stats.linregress(df[feature], df[label])
    tau, tp = stats.kendaltau(df[feature], df[label])
    rho, rp = stats.spearmanr(df[feature], df[label])
    fskew = round(df[feature].skew(), roundto)
    lskew = round(df[label], roundto)

    # Add all of those value in the plot
    textstr = "Regression line:\n"
    textstr += f'y = {round(m, roundto)}x + {round(b, roundto)}\n'
    textstr += f'r = {round(r, roundto)}, p = {round(p, roundto)}\n'
    textstr += f'tau = {round(tau, roundto)}, p = {round(tp, roundto)}\n'
    textstr += f'rho = {round(rho, roundto)}, p = {round(rp, roundto)}\n'
    textstr += f'{feature} skew = {round(fskew, roundto)}\n'
    textstr += f'{label} skew = {round(lskew, roundto)}\n'

    plt.text(.95, 0.2, textstr, fontsize=12, transform=plt.gcf().transFigure)
    plt.show()


def bar_chart(df, feature, label, roundto=4, p_threshold=0.05):
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


def crosstab_(df, feature, label, roundto=4):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy import stats

    # Generate the crosstab
    crosstab = pd.crosstab(df[feature], df[label])

    # Calculate the statistics
    X, p, dof, contingency_table = stats.chi2_contingency(crosstab)

    textstr = f'X2 : {round(X, roundto)}\n'
    textstr += f'p: {round(p, roundto)}'
    plt.text(.95, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

    ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), columns=crosstab.columns, index=crosstab.index)
    sns.heatmap(ct_df, annot=True, fmt='d', cmap='coolwarm')

    plt.show()


def bivariate_stats(df, label, roundto=4):
    import pandas as pd
    from scipy import stats

    output_df = pd.DataFrame(
        columns=['missing %', 'skew', 'type', 'unique', 'p', 'r', 'tau', 'sigma', 'y=m(x) + b', 'F', 'X2'])

    for feature in df:
        if feature != label:

            df_temp = df[[feature, label]].copy()
            df_temp = df_temp.dropna().copy()
            missing = round((df.shape[0] - df_temp.shape[0]) / df.shape[0], roundto) * 100
            dtype = df_temp[feature].dtype
            unique = df_temp[feature].nunique()

            if pd.api.types.is_numeric_dtype(df[feature]) and pd.api.types.is_numeric_dtype(df[label]):
                # process N2N relationships
                m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])
                output_df.loc[feature] = [f'{missing}%', round(p, roundto), round(r, roundto),
                                          f"y={round(m, roundto)}x + {round(b, roundto)}", '-', '-']
                scatterplot(df_temp, feature, label, )
            elif not pd.api.types.is_numeric_dtype(df_temp[feature]) and not pd.api.types.is_numeric_dtype(
                    df_temp[label]):
                # process C2C relationships
                contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
                X2, p, dof, expected = stats.chi2_contingency(contingency_table)
                output_df.loc[feature] = [f'{missing}%', p, '-', '-', '-', '-', '-', X2]
            else:
                # process C2N and N2C relationships
                if pd.api.types.is_numeric_dtype(df_temp[feature]):
                    skew = round(df[feature].skew(), roundto)
                    num = feature
                    cat = label
                else:
                    skew = '-'
                    num = label
                    cat = feature

                groups = df_temp[cat].unique()
                groups_list = []
                for g in groups:
                    groups_list.append(df_temp[df_temp[cat] == g][num])

                f, p = stats.f_oneway(*groups_list)  # <- same as (group_list[0], group_list[1], ..., group_list[n])

                output_df.loc[feature] = [f'{missing}%', skew, dtype, unique, round(p, roundto), '-', '-',
                                          round(f, roundto), '-']

    return output_df.sort_values(by=['p'], ascending=True)


'''
 how to use 

  import sys
  sys.path.append('functions.py file path')
  import functions as fun

  fun.bivairate(dataset, feature, label)


'''

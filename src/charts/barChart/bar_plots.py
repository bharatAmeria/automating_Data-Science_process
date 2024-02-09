def bar_charts(df, feature, label, roundto=4):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # Create the bar chart
    sns.barplot(x=df[feature], y=df[label])

    # Create the numeric lists needed too calculate the ANOVA
    groups = df[feature].unique()
    groups_list = []
    for g in groups:
        groups_list.append(df[df[feature] == g][label])

    f, p = stats.f_oneway(*groups_list)  # <- same as (group_list[0], group_list[1], ..., group_list[n])

    # If there are too many feature groups, print x labels vertically
    if df[feature].nunique() > 7:
        plt.xticks(rotation=90)
    # Create textstr to add statistics to chart
    textstr = f'F: {round(f, roundto)}\n'
    textstr += f'p: {round(p, roundto)}'

    plt.text(.95, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)
    plt.show()

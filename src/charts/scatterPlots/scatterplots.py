def scatterplot(df, feature, label, roundto=4, linecolor='orange'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # create the plot
    sns.regplot(x=df[feature], y=df[label], line_kws={"color": linecolor})

    # calculate the regression line
    m, b, r, p, err = stats.linregress(df[feature], df[label])

    # Add all of those value in the plot
    textstr = "Regression line:\n"
    textstr += f'y = {round(m, roundto)}x + {round(b, roundto)}\n'
    textstr += f'r = {round(r, roundto)}\n'
    textstr += f'p = {round(p, roundto)}'

    plt.text(.95, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)
    plt.show()

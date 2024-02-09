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

    return crosstab

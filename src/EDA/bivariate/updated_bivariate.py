""" Bivariate analysis using charts """

def bivariate_updated(df, label, roundto=4):
    import pandas as pd
    from src.charts.barChart.bar_plots import bar_charts
    from src.charts.scatterPlots.scatterplots import scatterplot
    from src.charts.crossTab.crosstab import crosstab_
    from scipy import stats

    output_df = pd.DataFrame(columns=['missing %', 'p', 'r', 'y=m(x) + b', 'f', 'X2'])

    for feature in df:
        if feature != label:

            df_temp = df[[feature, label]].copy()
            df_temp = df_temp.dropna().copy()
            missing = round((df.shape[0] - df_temp.shape[0]) / df.shape[0], roundto) * 100
            if pd.api.types.is_numeric_dtype(df[feature]) and pd.api.types.is_numeric_dtype(df[label]):
                # process N2N relationships
                m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])
                output_df.loc[feature] = [f'{missing}%', round(p, roundto), round(r, roundto),
                                          f"y={round(m, roundto)}x + {round(b, roundto)}", '-', '-']
                scatterplot(df_temp, feature, label,)
            elif not pd.api.types.is_numeric_dtype(df_temp[feature]) and not pd.api.types.is_numeric_dtype(
                    df_temp[label]):
                # process C2C relationships
                contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
                X2, p, dof, expected = stats.chi2_contingency(contingency_table)
                output_df.loc[feature] = [f'{missing}%', p, '-', '-', '-', X2]
                crosstab_(df_temp, feature, label)
            else:
                # process C2N and N2C relationships
                if pd.api.types.is_numeric_dtype(df_temp[feature]):
                    num = feature
                    cat = label
                else:
                    num = label
                    cat = feature

                groups = df_temp[cat].unique()
                groups_list = []
                for g in groups:
                    groups_list.append(df_temp[df_temp[cat] == g][num])

                f, p = stats.f_oneway(*groups_list)  # <- same as (group_list[0], group_list[1], ..., group_list[n])

                output_df.loc[feature] = [f'{missing}%', round(p, roundto), '-', '-', round(f, roundto), '-']
                bar_charts(df_temp, cat, num)
            return output_df.sort_values(by=['p'], ascending=True)

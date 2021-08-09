import matplotlib.pyplot as plt
import pylab as plt
import pandas as pd
import numpy as np
import sklearn.impute as skin
from sqlalchemy import create_engine
import locale
import scipy as sp
import typing


def cnfl(x):
    import numpy as np
    if ',' in x:
        x = np.float32(locale.atof(x))
    elif x == 'NULL':
        x = np.nan
    else:
        try:
            x = np.float32(x)
        except:
            x = np.nan
    return x


def value_filtering(df):
    df = df[df['pat_bd'].notnull()]
    df = df[df['pat_datetime'].notna()]
    # df.dropna(axis=1,how='all',inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    return df


def population_descriptives(df, columns):
    grouped = df.groupby('pat_hosp_id', sort=False)
    df1 = grouped.get_group((list(grouped.groups)[0]))
    df1 = lab_values_feature_building(df1, columns, df1['pat_hosp_id'].iloc[0])
    for pat, group in grouped:
        df_temp = lab_values_feature_building(group, columns, pat)
        if 'Death' in group['Status'].unique():
            df_temp['Label'] = 'Death'
        else:
            df_temp['Label'] = 'Alive'
        df1 = df1.append(df_temp)
        del df_temp
    return df1


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)  # imshow
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)


def column_sorter(columns):
    a = []
    b = []
    for column in columns:
        if 'mon' in column:
            b.append(column)
        elif 'vent_m_' in column:
            a.append(column)
        else:
            b.append(column)
    return a, b


def lab_values_feature_building(dfl, columns, pat):
    """"df=pd.DataFrame(columns=[],index=dfl['pat_hosp_id'].unique())
    print(df.head())#data={'dummy':0},index=['a'])
    for pat, group in dfl.groupby('pat_hosp_id',sort=False):
        print('start_group')
        list_columns=[] """
    df = pd.DataFrame()
    group = dfl
    for colum in columns:
        if (group[colum].dtypes == float):
            temp = pd.DataFrame(group[colum].describe().to_numpy()[None],
                                columns=[colum+'_count', colum+'_mean', colum+'_std', colum+'_min', colum+'_25%', colum+'_50%', colum + '_75%', colum+'_max'], index=[pat])

            df = pd.concat([temp, df], axis=1)
        elif (group[colum].dtypes == object):

            temp = pd.DataFrame(group[colum].describe().to_numpy()[None],
                                columns=[colum+'_count', colum+'_unique', colum+'_top', colum+'_freq'], index=[pat])

            df = pd.concat([temp, df], axis=1)
        else:
            temp = pd.DataFrame(group[colum].describe(datetime_is_numeric=True)['max'].to_numpy()[None],
                                columns=[colum+'_max'], index=[pat])  # [colum+'_count',colum+'_mean',colum+'_min',colum+'_25%',colum+'_50%',colum+'_75%',colum+ '_max'],index=[pat])

            df = pd.concat([temp, df], axis=1)
        del temp
    return df


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]


def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values


def fft_feat(x, pat, colum):
    """
    Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"aggtype": s} where s str and in ["centroid", "variance",
        "skew", "kurtosis"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    def get_moment(y, moment=1):
        """
        Returns the (non centered) moment of the distribution y:
        E[y**moment] = \\sum_i[index(y_i)^moment * y_i] / \\sum_i[y_i]

        :param y: the discrete distribution from which one wants to calculate the moment
        :type y: pandas.Series or np.array
        :param moment: the moment one wants to calcalate (choose 1,2,3, ... )
        :type moment: int
        :return: the moment requested
        :return type: float
        """
        return y.dot(np.arange(len(y), dtype=float)**moment) / y.sum()+1

    def get_centroid(y):
        """
        :param y: the discrete distribution from which one wants to calculate the centroid
        :type y: pandas.Series or np.array
        :return: the centroid of distribution y (aka distribution mean, first moment)
        :return type: float
        """
        return get_moment(y, 1)

    def get_variance(y):
        """
        :param y: the discrete distribution from which one wants to calculate the variance
        :type y: pandas.Series or np.array
        :return: the variance of distribution y
        :return type: float
        """
        return get_moment(y, 2) - get_centroid(y) ** 2

    def get_skew(y):
        """
        Calculates the skew as the third standardized moment.
        Ref: https://en.wikipedia.org/wiki/Skewness#Definition

        :param y: the discrete distribution from which one wants to calculate the skew
        :type y: pandas.Series or np.array
        :return: the skew of distribution y
        :return type: float
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, skew should be 0 and variance 0.  However, in the discrete limit,
        # the skew blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                get_moment(y, 3) - 3 * get_centroid(y) *
                variance - get_centroid(y)**3
            ) / get_variance(y)**(1.5)

    def get_kurtosis(y):
        """
        Calculates the kurtosis as the fourth standardized moment.
        Ref: https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments

        :param y: the discrete distribution from which one wants to calculate the kurtosis
        :type y: pandas.Series or np.array
        :return: the kurtosis of distribution y
        :return type: float
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, kurtosis should be 3 and variance 0.  However, in the discrete limit,
        # the kurtosis blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                get_moment(y, 4) - 4 * get_centroid(y) * get_moment(y, 3)
                + 6 * get_moment(y, 2) * get_centroid(y)**2 -
                3 * get_centroid(y)
            ) / get_variance(y)**2

    fft_abs = np.abs(np.fft.rfft(x))
    pdser = pd.Series(data=get_centroid(fft_abs), index=[
                      pat], name=(colum + '_fft_centoid'))
    pdser.append(pd.Series(get_variance(fft_abs), index=[
                 pat], name=(colum + '_fft_variance')))
    pdser.append(pd.Series(get_skew(fft_abs), index=[
                 pat], name=(colum + '_fft_skew')))
    pdser.append(pd.Series(get_kurtosis(fft_abs), index=[
                 pat], name=(colum + '_fft_kurtosis')))

    return pdser

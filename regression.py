import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import yfinance as yf

from data import reader


def regression(df_factor, issuer):

    # Reads ticker data of issuer
    df_issuer = reader([issuer])

    # Merges and splits risk factor and issuer data so it is equal
    factor_ticker_list = df_factor.keys()

    df = pd.merge(df_factor, df_issuer, how="inner", on="Date")
    df = df.dropna()

    df_factor = df[factor_ticker_list[0]]
    df_factor = pd.DataFrame(df_factor)
    df_factor.columns = [factor_ticker_list[0]]

    for i in range(1, len(factor_ticker_list)):
        df_factor[factor_ticker_list[i]] = df[factor_ticker_list[i]]

    for ticker in factor_ticker_list:
        del df[ticker]

    # Applies regression
    regres = LinearRegression().fit(df_factor, df)
    alphas = np.array(regres.coef_[0])
    alphas_norm = alphas / (np.sqrt(np.dot(alphas, alphas)))
    beta = regres.score(df_factor, df)

    return alphas_norm, beta

def orthogonalize(df_factor):
    """orthogonalize systematic risk factors through linear regression"""
    y = pd.DataFrame(df_factor[df_factor.keys()[1]])
    x = pd.DataFrame(df_factor[df_factor.keys()[0]])

    regres = LinearRegression().fit(x,y)
    gamma = np.array(regres.coef_[0])
    df_factor[df_factor.keys()[1]] = df_factor[df_factor.keys()[1]] - df_factor[df_factor.keys()[0]]*gamma[0]
    # df_factor[df_factor.keys()[1]] = (df_factor[df_factor.keys()[1]] - df_factor[df_factor.keys()[1]].mean())/df_factor[df_factor.keys()[1]].std()
    return df_factor

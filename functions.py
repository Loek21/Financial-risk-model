import numpy as np
import yfinance as yf
import pandas as pd
from collections import OrderedDict
import pycountry_convert as pc
from tqdm import tqdm
import concurrent.futures
from data import reader
import matplotlib.pyplot as plt
import sys
from copulae import GumbelCopula, NormalCopula, ClaytonCopula
from scipy.stats import invgauss, norm


def rescale_alpha(alpha, omega):
    """rescales the alpha parameters to
    make a*omega*a = 1"""

    alpha_copy = alpha

    for i in range(len(alpha)):
        scaling = 1
        last_try = abs(np.dot(np.array(alpha[i]), np.dot(omega, np.array(alpha[i]))) - 1)
        for scaling_param in np.linspace(0.1, 5, 500):
            if abs(np.dot(scaling_param*np.array(alpha[i]), np.dot(omega, scaling_param*np.array(alpha[i]))) - 1) < last_try:
                scaling = scaling_param
                last_try = abs(np.dot(scaling_param*np.array(alpha[i]), np.dot(omega, scaling_param*np.array(alpha[i]))) - 1)

        alpha_copy[i] = list(scaling*np.array(alpha_copy[i]))

    return alpha_copy


def issuer_correlation(issuers, alpha, beta, omega):
    df_issuers = reader(issuers)
    # calculate the pairwise correlation from the model
    correlations = []
    for i in range(len(alpha)):
        print(np.dot(np.array(alpha[i]), np.dot(omega, np.array(alpha[i]))))
        for j in range(len(alpha)):
            if i == j:
                correlation = 1-beta[i] + np.sqrt(beta[i]*beta[j])*np.dot(np.array(alpha[i]).T, np.dot(omega, np.array(alpha[j])))

            else:
                correlation = np.sqrt(beta[i]*beta[j])*np.dot(np.array(alpha[i]).T, np.dot(omega, np.array(alpha[j])))

            correlations.append(correlation)

    # get the empirical pairwise correlation
    corr = df_issuers.corr()
    listed = list(np.hstack(corr.values))
    listed_new = []
    for val in listed:
        if val < 0.9999:
            listed_new.append(val)

    return listed_new, correlations

def gather_data(ticker, factors):
    """Reads issuer info"""
    temp_factor = []

    for factor in factors:
        k = yf.Ticker(ticker).info[factor]

        if k == "Consumer Defensive":
            temp_factor.append("Consumer Services")

        if factor == "country":
            country_code = pc.country_name_to_country_alpha2(
                k, cn_name_format="default"
            )
            continent_name = pc.country_alpha2_to_continent_code(country_code)
            temp_factor.append(continent_name)

        else:
            temp_factor.append(k)

    return temp_factor


def initialAlpha(ticker_dict):
    """"Creates alpha vector to assign systematic risk factor"""
    size = 0
    for key, value in ticker_dict.items():
        size += len(value)

    return np.zeros(size)


def random_F_drawing(corr_cholesky):
    # make vector of iid normal rv's
    z = np.array([np.random.normal() for _ in range(len(corr_cholesky))])
    # transform the vector to vector of desired dist
    x = np.dot(corr_cholesky, z)

    return x


def reverseDict(ticker_dict):
    """Inverts layers of dictionary by keys"""
    temp = []
    for k, v in ticker_dict.items():
        for i, j in v.items():
            temp.append((j, dict([(i, k)])))
    return OrderedDict(temp)


def alphaMatrix(df, reverse_dict, issuers, factors):
    index = df.index
    issuers_catagorized = analyseIssuer(issuers, factors)
    alpha_vectors = []

    for i in range(len(issuers)):
        a = []
        issuer_specific = issuers_catagorized[i]

        for k, v in reverse_dict.items():
            for x, y in v.items():
                # print(issuer_specific, x)
                if x in issuer_specific:
                    a.append(1)
                else:
                    a.append(0)

        alpha_vectors.append(a)

    factor_combo = [
        pd.DataFrame(np.matmul(df.values, np.diag(a)), index=index)
        for a in alpha_vectors
    ]
    factor_combo_dropped = [k.loc[:, (k != 0).any(axis=0)] for k in factor_combo]

    return alpha_vectors, factor_combo_dropped


def cholesky(ticker_dict, issuers, factors):
    alpha = initialAlpha(ticker_dict)
    risk = []
    reverse_dict = reverseDict(ticker_dict)

    for key, value in reverse_dict.items():
        risk.append(key)

    # gathers historical data of every systematic factor
    df = yf.Ticker(risk[0]).history(period="10y", interval="1mo")["Close"]
    df = pd.DataFrame(df)
    df.columns = [risk[0]]

    for ticker in risk[1:]:
        ticker_object_data = pd.DataFrame(
            yf.Ticker(ticker).history(period="10y", interval="1mo")["Close"]
        )
        ticker_object_data.columns = [f"{ticker}"]
        df = pd.merge(df, ticker_object_data, how="inner", on="Date")

    # Calculates log returns of systematic factors
    df = np.log(df) - np.log(df.shift(1))
    df = df.dropna()

    # Standardizes log returns
    for column in df:
        df[column] = (df[column] - df[column].mean()) / df[column].std()

    # Correlation matrix
    corr = df.corr().values
    chol = np.linalg.cholesky(corr)

    print(len((np.array(df.values).T)))
    data = np.random.normal(size=(50,13))
    cop = NormalCopula(13)
    cop.fit(np.array(df.values))
    #cop.fit(data)

    # alpha vectors: is the new alpha vector based on which systematic risk factors are used
    # alpha dropped: Risk factor returns
    alpha_vectors, alpha_dropped = alphaMatrix(df, reverse_dict, issuers, factors)
    # print(alpha_vectors)
    # print(alpha_dropped)
    return chol, alpha_vectors, alpha_dropped, cop


def analyseIssuer(issuers, factors):
    all_factors = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        values = [executor.submit(gather_data, ticker, factors) for ticker in issuers]

        concurrent.futures.wait(values, timeout=None, return_when='ALL_COMPLETED')
        for f in values:
            all_factors.append(f.result())
    return all_factors


def MC_sim(issuers, alpha, beta, threshhold, chol, cop):

    loss_list = []
    #X_list = []
    for i in range(100000):
        L = 0
        #random_F = random_F_drawing(chol)
        # random_X = np.random.multivariate_normal([0 for i in range(len(chol))], np.dot(chol, chol.T))
        random_F1 = cop.random(1)
        # random_F2 = cop.random(1)
        # box_muller = [np.sqrt(-2*np.log(x))*np.cos(2*np.pi*y) for x,y in zip(random_F1, random_F2)]
        # random_F = box_muller
        random_F = norm.ppf(random_F1)
        #print(random_X, random_F)
        for j in range(len(issuers)):

            a = alpha[j]
            b = beta[j]
            thresh = threshhold[j]

            Xi = (
                np.sqrt(b) * np.dot(np.array(a).T, random_F)
                + np.sqrt(1 - b) * np.random.normal()
            )

            #X_list.append(Xi)

            Li = 1 / len(issuers) if Xi < thresh else 0

            L += Li
        loss_list.append(L)
    return loss_list


def simulateCorrelation(issuers, alpha, beta, threshhold, chol, cop):

    loss_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        values = [executor.submit(MC_sim, issuers, alpha, beta, threshhold, chol, cop) for _ in range(10)]

        for f in concurrent.futures.as_completed(values):
            loss_list.append(f.result())

    loss_list = np.hstack(loss_list)

    return loss_list

def MC_sim_nocor(issuers, threshhold):
    loss_list = []

    for i in tqdm(range(100000)):
        L = 0

        for j in range(len(issuers)):

            thresh = threshhold[j]

            Xi = np.random.normal()

            Li = 1 / len(issuers) if Xi < thresh else 0

            L += Li

        loss_list.append(L)

    return loss_list

def simulateNoCorrelation(issuers, threshhold):

    loss_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        values = [executor.submit(MC_sim_nocor, issuers, threshhold) for _ in range(10)]

        for f in concurrent.futures.as_completed(values):
            loss_list.append(f.result())

    loss_list = np.hstack(loss_list)


    return loss_list

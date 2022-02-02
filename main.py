import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.stats import multivariate_normal, norm, t
from regression import regression, orthogonalize
# from functions import (
#     initialAlpha,
#     reverseDict,
#     cholesky,
#     analyseIssuer,
#     alphaMatrix,
#     simulateCorrelation,
#     simulateNoCorrelation,
#     issuer_correlation,
#     rescale_alpha,
# )
from credit import (
    initialAlpha,
    reverseDict,
    cholesky,
    analyseIssuer,
    alphaMatrix,
    simulateCorrelation,
    simulateNoCorrelation,
    issuer_correlation,
    rescale_alpha,
)

from tqdm import tqdm
from collections import OrderedDict
import sys

def execute(portfolio):
    issuers = []
    threshhold = []
    issuer_factors = []

    df = 100

    with open(portfolio) as f:
    # with open('issuer.csv') as f:
        read = csv.reader(f, delimiter=',')
        next(read)
        for row in read:
            issuers.append(str(row[0]))
            threshhold.append(norm.ppf(float(row[1])))
            issuer_factors.append([str(row[2]), str(row[3])])
            # threshhold.append(t.ppf(float(row[1]), df))


    ticker_dict = {
        "Region": {
            "EU": "EXSA.MI",
            "NA": "IYY",
            # "SA": "LTAM.PA",
            "OC": "IPAC",
            "AF": "XMAF.L",
            "AS": "AAXJ",
        },
        "Sector": {
            "Technology": "IYW",
            "Financial Services": "XDWF.MI",
            "Communication Services": "VOX",
            "Healthcare": "IYH",
            "Industrials": "XDWI.L",
            "Consumer Services": "XDWS.MI",
            "Consumer Cyclical": "XDWC.MI",
            "Basic Materials": "XDWM.MI",
        },
    }

    beta = []
    model = ["country", "sector"]

    # Returns cholesky decomp of Correlation matrix of systematic factors
    # alpha mapping vectors for issuers
    # and systematic factor returns
    chol, alpha, a_dropped, cop = cholesky(ticker_dict, issuers, model, issuer_factors)

    # For each issuer
    for i in range(len(issuers)):

        # orthogonalizes systematic factors
        ortho = orthogonalize(a_dropped[i])

        # Calibrates issuer returns on systematic factor Returns
        # Returns alpha weights and beta (R^2) sensitivity
        # a, b = regression(a_dropped[i], issuers[i])
        a, b = regression(ortho, issuers[i])

        beta.append(b)
        counter = 0

        # changes alpha mapping vector to the correct alphas found through regression
        for x, entry in enumerate(alpha[i]):

            if entry == 1 and counter == 0:
                counter += 1
                alpha[i][x] = a[0]

            elif entry == 1:
                alpha[i][x] = a[1]

    # print(len(alpha))
    # print(beta)
    # print(alpha)
    rescaled_alpha = rescale_alpha(alpha, np.dot(chol, chol.T))

    alpha = rescaled_alpha
    # print(alpha)
    #loss_list = simulateNoCorrelation(issuers, threshhold)
    loss_list = simulateCorrelation(issuers, alpha, beta, threshhold, chol, cop, df)
    loss_list = np.array(loss_list)*len(issuers)
    sort_loss = sorted(loss_list)
    print(sort_loss[int(len(sort_loss)*0.99)], sort_loss[int(len(sort_loss)*0.995)], sort_loss[int(len(sort_loss)*0.999)])

    # corr1, corr2 = issuer_correlation(issuers, alpha, beta, np.dot(chol, chol.T))

    return loss_list

if __name__ == "__main__":

    csv_list = ["issuer_normal.csv", "issuer_middle.csv", "issuer_heavy.csv"]
    # csv_list = ["issuer_heavy_correlated.csv"]

    low_cor = execute(csv_list[0])
    mid_cor = execute(csv_list[1])
    high_cor = execute(csv_list[2])

    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist([low_cor, mid_cor, high_cor], bins=30, label=["Low", "Medium", "High"])
    ax.set_yscale('log')
    ax.set_ylabel("Number of occurences")
    ax.set_xlabel("Defaults")
    ax.set_title("Number of defaults using a Gaussian copula")
    ax.legend()
    plt.show()


    # plt.hist(corr2, label="Model correlation")
    # plt.hist(corr1, label="Empirical correlation")
    # plt.xlabel("Correlation")
    # plt.ylabel("Number of occurences")
    # plt.legend()
    # plt.show()

    # x, y = np.random.multivariate_normal([0, 0, 0, 0, 0], corr)
    # x, y = np.random.multivariate_normal([0, 0, 0, 0, 0], corr)

    # x = np.linspace(0, 5, 10, endpoint=False)
    # y = multivariate_normal.pdf(x, mean=None, cov=corr)
    # print(y)

    # plt.plot(rv.pdf)
    # plt.show()
    # plt.plot(x, y, "x")
    # plt.axis("equal")
    # plt.show()

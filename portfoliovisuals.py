import numpy as np
import yfinance as yf
import pandas as pd
from collections import OrderedDict
import pycountry_convert as pc
import concurrent.futures
import matplotlib.pyplot as plt
import csv
from scipy.stats import multivariate_normal, norm


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

if __name__ == "__main__":
    issuers = []
    threshhold = []
    model = ["country", "sector"]

    with open('issuer_heavy_correlated.csv') as f:
        read = csv.reader(f, delimiter=',')
        next(read)
        for row in read:
            issuers.append(str(row[0]))
            threshhold.append(float(row[1]))


    all_factors = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        values = [executor.submit(gather_data, ticker, model) for ticker in issuers]

        concurrent.futures.wait(values, timeout=None, return_when='ALL_COMPLETED')
        for f in values:
            all_factors.append(f.result())

    dict_region = {}
    dict_score = {}
    dict_sect = {}
    for issuer in all_factors:
        if issuer[0] not in dict_region.keys():
            # if issuer[0] == "NA":
            #     issuer[0] = "North America"
            # if issuer[0] == "EU":
            #     issuer[0] = "Europe"
            dict_region[issuer[0]] = 1
        else:
            dict_region[issuer[0]] += 1

        if issuer[1] not in dict_sect.keys():
            dict_sect[issuer[1]] = 1
        else:
            dict_sect[issuer[1]] += 1

    for score in threshhold:
        if score == 0.000141:
            score = "AA"
        if score == 0.000544:
            score = "A"
        if score == 0.001997:
            score = "BBB"
        if score == 0.008538:
            score = "BB"
        if score == 0.2433872:
            score = "CCC"
        if score not in dict_score.keys():
            dict_score[score] = 1
        else:
            dict_score[score] += 1

    label_region = dict_region.keys()
    label_sect = dict_sect.keys()
    label_score = dict_score.keys()
    region_size = []
    sect_size = []
    score_size = []
    for key in label_region:
        region_size.append(dict_region[key])
    for key in label_sect:
        sect_size.append(dict_sect[key])
    for key in label_score:
        score_size.append(dict_score[key])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.pie(region_size, labels=label_region, autopct='%1.1f%%')
    ax1.set_title("Portfolio region distribution")
    # plt.show()
    ax2.pie(sect_size, labels=label_sect, autopct='%1.1f%%')
    ax2.set_title("Portfolio sector distribution")
    plt.show()
    plt.pie(score_size, labels=label_score, autopct='%1.1f%%')
    plt.title("Portfolio credit score distribution")
    plt.show()

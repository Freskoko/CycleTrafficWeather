import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from loguru import logger
from scipy.stats import spearmanr

# get current filepath to use when opening/saving files
PWD = Path().absolute()
FIGPWD = f"{PWD}/src"


def graph_hour_variance(df: pd.DataFrame) -> None:
    """
    Given a dataframe with its index as date in the format of "2023-01-01 14:00:00",
    this function graphs the variance of traffic values in "Total_trafikk" for each hour.
    The variance is calculated from the total traffic values for each hour.
    """

    df.index = pd.to_datetime(df.index)
    df["hour"] = df.index.hour

    # find hourly variance
    hourly_variance = df.groupby("hour")["Total_trafikk"].var()

    plt.figure(figsize=[10, 5])
    plt.bar(hourly_variance.index, hourly_variance.values)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Variance in Total Traffic")
    plt.suptitle("Variance in Total Traffic by Hour of the Day")
    plt.title("(After removing traffic in the 99th quantile)")
    plt.xticks(range(0, 24))
    plt.savefig(f"{PWD}/figs/hour_variance_99")
    plt.clf()


def graph_hour_diff(df: pd.DataFrame) -> None:
    """
    Given a dataframe with its index as date in the format of "2023-01-01 14:00:00",
    this function calculates and graphs the difference between the maximum and minimum
    traffic values in "Total_trafikk" for each hour.
    """

    df["hour"] = df.index.hour
    df_grouped = df.groupby("hour")["Total_trafikk"]
    hourly_max = df_grouped.max()
    hourly_min = df_grouped.min()

    # Calculate the difference
    hourly_diff = hourly_max - hourly_min

    # Plot the difference
    plt.figure(figsize=[10, 5])
    plt.bar(hourly_diff.index, hourly_diff.values)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Difference in Total Traffic")
    plt.title("Difference in Total Traffic by Hour of the Day")
    plt.xticks(range(0, 24))
    plt.savefig(f"{PWD}/figs/traffic_diff_perhour")
    plt.clf()


def graph_total_traffic_overtime(df: pd.DataFrame, VERSION: str) -> None:
    """
    Given a dataframe with its index as date in the format of "2023-01-01 14:00:00",
    Graphs total traffic over time. Traffic is found in the "Total_trafikk" col.

    VERSION is a string to help indicate if graphing is pre or post processing
    """
    plt.clf()
    plt.figure(figsize=(15, 7))
    plt.plot(
        df.index,
        df["Total_trafikk"],
        label="Total Traffic",
    )

    plt.xlabel("Time")
    plt.ylabel("Traffic")
    plt.suptitle(f"Time vs Traffic_{VERSION}")

    plt.grid(True)
    plt.legend()
    plt.savefig(f"{PWD}/figs/timeVStraffic_{VERSION}.png")


def graph_weekly_amounts(df: pd.DataFrame) -> None:
    """
    Given a dataframe with its index as date in the format of "2023-01-01 14:00:00",
    Graphs average traffic per day. Traffic is found in the "Total_trafikk" col.
    """

    days = [
        "d_Monday",
        "d_Tuesday",
        "d_Wednesday",
        "d_Thursday",
        "d_Friday",
        "d_Saturday",
        "d_Sunday",
    ]

    avg_traffic = [df[df[day] == 1]["Total_trafikk"].mean() for day in days]

    avg_traffic_df = pd.DataFrame({"Day": days, "Average_Traffic": avg_traffic})

    fig = px.bar(
        avg_traffic_df,
        x="Day",
        y="Average_Traffic",
        title="Average Traffic per Day of the Week",
    )
    fig.write_image(f"{PWD}/figs/weekly_traffic.png")


def graph_monthly_amounts(df: pd.DataFrame):
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    avg_traffic_monthly = df.groupby("month")["Total_trafikk"].mean()

    avg_traffic_monthly_df = pd.DataFrame(
        {"Month": months, "Average_Traffic": avg_traffic_monthly}
    )

    fig = px.bar(
        avg_traffic_monthly_df,
        x="Month",
        y="Average_Traffic",
        title="Average Traffic per Month",
    )
    fig.write_image(f"{PWD}/figs/monthly_traffic.png")


def graph_a_vs_b(
    titletext: str, df: pd.DataFrame, a: str, b: str, alabel: str, blabel: str
) -> None:
    """
    General function to plot two items in a dataframe against eachother

    Also calculates spearmann and pearson correlation between the two values

    Inputs:
        titletext: the title of the plot
        df: df contaning the two items needing to be graphed
        a: item 1 to be graphed
        b: item 2 to be graphed
        alabel: label to be used for this variable in the graph
        blabel: label to be used for this variable in the graph
    """

    # see limits on data
    print(f"GRAPHING : working on graphing '{a} vs {b}'")
    print(f"GRAPHING : {a} looks like :")
    print(f"GRAPHING : max {a} is {max(df[a])}")
    print(f"GRAPHING : min {a} is {min(df[a])}")

    # Set limits on x - axis to limits + 5 in order to see range of data
    start_time = time.time()
    plt.xlim([min(df[a]) - 5, max(df[a]) + 5])  # x axis limits
    plt.figure(figsize=(15, 7))
    plt.bar(df[a], df[b])

    plt.xlabel(f"{a} ({alabel})")
    plt.ylabel(f"{b} ({blabel})")
    plt.suptitle(f"{a} vs {b} ")

    # Calculate spearmann/pearson correlation to see if trends observed visually also can be seen statistically
    pear = round(pearson_r_corr(df[a], df[b]), 4)
    spear = round(spearman_rho_corr(df[a], df[b]), 4)

    plt.title(f"""pearson_corr = {pear} spearmann_corr = {spear} {titletext}""")
    plt.grid(True)
    plt.savefig(f"{PWD}/figs/{a}VS{b}_{titletext}")

    # :warning: clear fig is very important as not clearing will cause many figs to be created ontop of eachother
    plt.clf()
    print(f"GRAPHING : saved fig '{a} VS {b}' in figs")
    print(f"GRAPHING : --- Graph took: {round(time.time() - start_time,2)} seconds ---")


def pearson_r_corr(a: float, b: float) -> float:
    """
    Helper function to generate pearson correlation
    """
    corr = np.corrcoef(a, b)[0, 1]
    return corr


def spearman_rho_corr(a: float, b: float) -> float:
    """
    Helper function to generate spearmann corr
    """
    corr, _ = spearmanr(a, b)
    return corr


def create_df_matrix(titletext: str, df: pd.DataFrame) -> None:
    """
    Function to create a covariance and correlation matrix for values in a dataframe

    Inputs:
        titletext: Text for graph
        df : the dataframe to create a covarience and correlation matrix from
    """

    # drop uneeded cols:
    df = df.drop(
        labels=[
            "d_Monday",
            "d_Tuesday",
            "d_Wednesday",
            "d_Thursday",
            "d_Friday",
            "d_Saturday",
            "d_Sunday",
            "public_holiday",
            "raining",
            "summer",
            "winter",
            "rush_hour",
            "sleeptime",
            "weekend",
            "month",
            "weekend",
            "hour",
            "Vindretning_radians",
        ],
        axis=1,
        inplace=False,
        errors="ignore",
    )

    # calculate the covariance matrix
    cov_matrix = df.cov()

    # normalizing values between 0 and 1
    cov_matrix_normalized = (cov_matrix - cov_matrix.min().min()) / (
        cov_matrix.max().max() - cov_matrix.min().min()
    )

    plt.figure(figsize=(16, 16))
    sns.heatmap(
        cov_matrix_normalized, annot=True, cmap="RdBu", vmin=-1, vmax=1, center=0
    )
    plt.title(f"Covariance Matrix Heatmap {titletext}")
    plt.savefig(f"{PWD}/figs/covv_matrix_{titletext}.png")

    plt.clf()

    # calculate the correlation matrix
    corr_matrix = df.corr()

    plt.figure(figsize=(16, 16))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu", vmin=-1, vmax=1, center=0)
    plt.title(f"Correlation Matrix Heatmap {titletext}")
    plt.savefig(f"{PWD}/figs/corr_matrix_{titletext}.png")


def graph_all_models(main_df: pd.DataFrame, pre_change: bool) -> None:
    """
    Large wrapper function to call many other graphing functions.

    Inputs:
        main_df: dataframe to perform graphing on
        pre_change boolean describing if graphing is occouring pre/post processing
    """

    if pre_change:
        titletext = "PRE_CHANGES"
    else:
        titletext = "POST_CHANGES"

    print(f"GRAPHING : Graphing all graphs... {titletext}")

    graph_total_traffic_overtime(main_df, VERSION=titletext)

    create_df_matrix(titletext, main_df)
    graph_a_vs_b(
        titletext,
        main_df,
        "Globalstraling",
        "Total_trafikk",
        "stråling",
        "antall sykler",
    )
    graph_a_vs_b(
        titletext, main_df, "Solskinstid", "Total_trafikk", "solskinn", "antall sykler"
    )
    graph_a_vs_b(
        titletext,
        main_df,
        "Lufttemperatur",
        "Total_trafikk",
        "grader celcius",
        "antall sykler",
    )

    if not pre_change:
        graph_a_vs_b(
            titletext,
            main_df,
            "Vindretning_x",
            "Total_trafikk",
            "Grader",
            "antall sykler",
        )
        graph_a_vs_b(
            titletext,
            main_df,
            "Vindretning_y",
            "Total_trafikk",
            "Grader",
            "antall sykler",
        )

    if pre_change:
        graph_a_vs_b(
            titletext,
            main_df,
            "Vindretning",
            "Total_trafikk",
            "Grader",
            "antall sykler",
        )

        graph_a_vs_b(titletext, main_df, "Vindstyrke", "Vindkast", "Styrke", "Kast")

        graph_a_vs_b(
            titletext, main_df, "Vindstyrke", "Total_trafikk", "Styrke", "Sykler"
        )

    graph_a_vs_b(
        titletext, main_df, "Lufttrykk", "Total_trafikk", "hPa", "antall sykler"
    )
    graph_a_vs_b(
        titletext, main_df, "Vindkast", "Total_trafikk", "m/s", "antall sykler"
    )

    print("GRAPHING : Finished graphing!")

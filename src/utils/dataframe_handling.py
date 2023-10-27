import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DEBUG = True
PWD = Path().absolute()


def feauture_engineer(df: pd.DataFrame, data2023: bool) -> pd.DataFrame:
    """
    Input: A dataframe containing traffic and weather data with DateFormatted as the index

    Adds:

    hour: 0-24
    day_in_week: (each day has their own col)
    month: 1-12
    weekend: 0/1
    public holiday: 0/1
    raining: 0/1
    summer: 0/1
    winter: 0/1
    rush_hour: 0/1
    sleeptime: : 0/1
    Vindretning x/y : 0-1

    Returns: df with more features
    """

    # BASIC DATE FEATURES
    # hour as own coloumn 0-23
    df["hour"] = df.index.hour

    # Instead of "day_in_week" being a num 0-6, add 7 coloumns to the dataframe, monday, tuesday .. etc
    # Have the value be 0 or 1, 0 if it is not that day of the week, 1 if it is

    day_week_dict = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    df["d"] = df.index.weekday.map(day_week_dict)

    # make each day their own coloumn
    df = pd.get_dummies(df, columns=["d"])

    complete_days = [
        "d_Monday",
        "d_Tuesday",
        "d_Wednesday",
        "d_Thursday",
        "d_Friday",
        "d_Saturday",
        "d_Sunday",
    ]

    # loop through days
    for day in complete_days:
        # if the day is not a column in df, add it with a value of 0
        if not day in df.columns:
            df[day] = 0

    # add month
    df["month"] = df.index.month

    # add weekend
    df["weekend"] = (df.index.weekday >= 5).astype(int)

    # MORE ADVANCED FEATURES

    # add public holidays
    holidays = [
        # jul osv
        "12-24",
        "12-25",
        "01-01",
        # påske
        "04-06",
        "04-07",
        "04-08",
        "04-09",
        "04-10",
        # labour day/17 mai
        "05-01",
        "05-17",
        "05-18",
    ]

    # add public holiday
    df["public_holiday"] = df.index.strftime("%m-%d").isin(holidays).astype(int)

    # add coloumn for rain if air pressure is higher than 1050 see README
    df["raining"] = df["Lufttrykk"] <= 996

    # add seasons
    df["summer"] = (df["month"] > 5) & (df["month"] < 8)
    df["winter"] = (df["month"] >= 10) | (df["month"] <= 2)

    # add when we expect a lot of traffic
    df["rush_hour"] = (df["hour"].between(7, 9)) | (df["hour"].between(15, 17))

    # add when we do not expect a lot of traffic
    df["sleeptime"] = (df["hour"] >= 22) | (df["hour"] < 6)

    # df["Vindretning"] is full of values 0-360, transform these to points on a circle
    df["Vindretning_radians"] = np.radians(df["Vindretning"])
    df["Vindretning_x"] = np.cos(df["Vindretning_radians"])
    df["Vindretning_y"] = np.sin(df["Vindretning_radians"])

    # we cant train where there are no traffic values
    if not data2023:  # dont drop values in 2023 data!
        df = df.dropna(subset=["Total_trafikk"])

    # change all values of TRUE in all rows to 1 and FALSE to 0
    # models need NUMERIC data
    df = df.replace({True: 1, False: 0})

    return df


def merge_frames(frames: list) -> (pd.DataFrame, pd.DataFrame):
    """
    Given a list of dataframes, merges the frames to one large dataframe, given that the
    index is a date, and the same across all dataframes

    """

    # grab first df
    df_final = frames[0]

    # index to date
    df_final.index = pd.to_datetime(df_final.index)

    for frame in frames[1:]:
        frame.index = pd.to_datetime(frame.index)

        # since dataframes have the same index, -> they can be merged
        df_final = df_final.merge(frame, how="outer", left_index=True, right_index=True)

    df_final = df_final.sort_index()

    df_2023 = df_2023 = df_final.loc["2023-01-01":"2023-12-31"]
    # get where index is between 2023-01-01 00:00:00 and 2023-12-31 00:00:00 to save.

    df_final = df_final.dropna(subset=["Trafikkmengde_Totalt_i_retning_Florida"])

    # finding means of values lead to floating point errors, round to fix these
    df_final = df_final.apply(pd.to_numeric, errors="ignore").round(30)

    # combine the two traffic cols to one total trafikk col!
    df_final["Total_trafikk"] = (
        df_final["Trafikkmengde_Totalt_i_retning_Florida"]
        + df_final["Trafikkmengde_Totalt_i_retning_Danmarksplass"]
    )

    df_final = df_final.drop(
        labels=[
            "Trafikkmengde_Totalt_i_retning_Florida",
            "Trafikkmengde_Totalt_i_retning_Danmarksplass",
        ],
        axis=1,
        inplace=False,
    )

    return df_2023, df_final


def trim_transform_outliers(df: pd.DataFrame, data2023: bool) -> pd.DataFrame:
    """
    Given a dataframe, trims values in the dataframe that are considered abnormal.

    What values are considered abnormal are covered in the README under "Dropped values"
    """
    # debug dict to look at lengths
    length_dict = {"before": len(df)}
    length_dict["afterGlobal"] = len(df)

    # Transform malformed data to NaN.

    # df["Globalstraling"] values above 1000 are set to NaN
    df["Globalstraling"] = np.where(
        (df["Globalstraling"] >= 1000),
        np.nan,
        df["Globalstraling"],
    )

    # df["Solskinstid"] values above 10.01 are set to NaN
    df["Solskinstid"] = np.where(df["Solskinstid"] >= 10.01, np.nan, df["Solskinstid"])
    length_dict["afterSolskinn"] = len(df)

    # df["Lufttemperatur"] values above 50 are set to NaN
    df["Lufttemperatur"] = np.where(
        df["Lufttemperatur"] >= 50, np.nan, df["Lufttemperatur"]
    )
    length_dict["afterLufttemp"] = len(df)

    # df["Lufttrykk"] values above 1050 are set to NaN
    df["Lufttrykk"] = np.where(df["Lufttrykk"] >= 1050, np.nan, df["Lufttrykk"])
    length_dict["afterLufttrykk"] = len(df)

    # df["Vindkast"] values above 65 are set to NaN
    df["Vindkast"] = np.where(df["Vindkast"] >= 65, np.nan, df["Vindkast"])
    length_dict["afterVindkast"] = len(df)

    # df["Vindretning"] values above 360 are set to NaN
    df["Vindretning"] = np.where(df["Vindretning"] >= 361, np.nan, df["Vindretning"])
    length_dict["afterVindretning"] = len(df)

    # df["Vindstyrke"] values above 1000 and below 0 are set to NaN
    df["Vindstyrke"] = np.where(df["Vindstyrke"] < 0, np.nan, df["Vindstyrke"])
    df["Vindstyrke"] = np.where(df["Vindstyrke"] >= 1000, np.nan, df["Vindstyrke"])
    length_dict["afterVindstyrke"] = len(df)

    # replace outliers (this should be fixed above, but this is just in case)
    df = df.replace(99999, np.nan)

    # observe NaN
    num_nan = df.isna().sum()
    print(f"PARSING : Number of NaNs in each column:\n{num_nan}")

    if not data2023:
        # "Total_trafikk" column, will not used for imputation -> keep it for later
        total_traffic_series = df["Total_trafikk"]

        # "Total_trafikk" column, will not used for imputation -> remove it from df
        df_no_traffic = df.drop(columns=["Total_trafikk"])

    if data2023:
        # 2023 data does not have
        df_no_traffic = df

    # Drop "Relativ luftfuktighet" as this data only exists in 2022 and 2023.
    # errors="ignore" since pandas complains when dropping data from dataframes where it does not exist

    df_no_traffic = df_no_traffic.drop(
        columns=["Relativ luftfuktighet"], errors="ignore"
    )

    # n_neighbors = 20 is best -> see report
    imputer = KNNImputer(n_neighbors=20, weights="distance")
    df_imputed = imputer.fit_transform(df_no_traffic)

    df_fixed = pd.DataFrame(
        df_imputed, columns=df_no_traffic.columns, index=df_no_traffic.index
    )

    if not data2023:
        df_fixed = pd.concat([df_fixed, total_traffic_series], axis=1)

    return df_fixed


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, normalizes certain values to a 0-1 scale

    Normalized values are covered in the README under "Normalized values"
    """

    # These were ideas that were considered but dropped. This was kept in for learning purposes.

    # scaler = MinMaxScaler()
    # df[["Globalstraling", "Lufttrykk", "Solskinstid",]] = scaler.fit_transform(
    #     df[
    #         [
    #             "Globalstraling",
    #             "Lufttrykk",
    #             "Solskinstid",
    #         ]
    #     ]
    # )

    # df["Vindkast"] = df["Vindkast"]**2

    print(f"PARSING : Values pre removal of outliers: {len(df)}")

    quant = df["Total_trafikk"].quantile(0.99)
    df = df[df["Total_trafikk"] <= quant]

    print(f"PARSING : Values post removal of outliers: {len(df)}")

    return df


def drop_uneeded_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, drops values deemed not needed, as these may just provide noise for the model,
    or their values are already represented as another coloumn

    Dropped values are covered in the README under "Dropped coloumns"
    """

    df.drop(["Vindretning", "Vindretning_radians", "Vindstyrke"], axis=1, inplace=True)

    return df


def train_test_split_process(
    df: pd.DataFrame,
) -> (dict, pd.DataFrame, pd.DataFrame, pd.DataFrame):  # fix
    """
    Given a df, data is split into training, test and validation.

    Returns:

    split_dict : dict containing x_train, y_train, etc... for all x,y vals.
    training_df : reconstructed dataframe containing only training data
    test_df : reconstructed dataframe containing only test data
    validation_df : reconstructed dataframe containing only validation data
    """

    y = df["Total_trafikk"]
    x = df.drop(["Total_trafikk"], axis=1)

    # transform 70% to training data
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, shuffle=False, test_size=0.3
    )

    # split 30% which was validation into 15% val and 15% test
    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, shuffle=False, test_size=0.5
    )

    # data exploration is only supposed to look at training data -> use x_train/y_train
    test_df = x_test.merge(y_test, how="outer", left_index=True, right_index=True)
    validation_df = x_val.merge(y_val, how="outer", left_index=True, right_index=True)
    training_df = x_train.merge(y_train, how="outer", left_index=True, right_index=True)

    split_dict = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_val": x_val,
        "y_val": y_val,
    }

    return split_dict, training_df, test_df, validation_df


def treat_2023_file(df: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    """
    A 2023 file handler, to fill in missing values given weather data

    Inputs:
        df: A dataframe contaning 2023 data
        model: the model to use to predict cycle trafikk
    Returns:
        A dataframe much like the input, with the cycle traffic values filled in.

    """
    df = df.drop(
        columns=[
            "Trafikkmengde_Totalt_i_retning_Danmarksplass",
            "Trafikkmengde_Totalt_i_retning_Florida",
        ]
    )

    df_fixed = trim_transform_outliers(df, True)

    # add important features to help the model
    df_final = feauture_engineer(df_fixed, True)
    print("PARSING : Features engineered")

    # drop coloumns which are not needed (noise)
    df_final = drop_uneeded_cols(df_final)
    print("PARSING : Uneeded cols dropped")

    try:
        df_final["Total_trafikk"] = model.predict(df_final)
    except ValueError as e:
        print(f"WARNING: MODEL PREDICTION ERROR {e}")

    # convert time, date and prediction to wanted format
    df_final["Dato"] = pd.to_datetime(df_final.index).date
    df_final["Tid"] = pd.to_datetime(df_final.index).hour

    df_final["Prediksjon"] = df_final["Total_trafikk"]

    new_df = df_final[["Dato", "Tid", "Prediksjon"]].copy()

    # make predictions to ints as float number of cyclists makes no sense.
    new_df["Prediksjon"] = new_df["Prediksjon"].astype(int)

    new_df.reset_index()

    new_df.to_csv(f"{PWD}/out/predictions.csv")

    return df_final

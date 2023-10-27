import json
import os
import pickle
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

# get current filepath to use when opening/saving files
PWD = Path().absolute()
RANDOM_STATE = 2
DEBUG = True


def treat_florida_files(filename: str) -> pd.DataFrame:
    """
    Input: filename of a florida weather file

    Process:
    set date to index
    change from 1 hour having 6 values, to one hour having the mean of those 6 values
    drop date and time coloumns which are now represented in the index

    Output:
    a dataframe of the csv file
    """

    df = pd.read_csv(filename, delimiter=",")

    # format date-data to be uniform, will help match data with traffic later
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Dato"] + row["Tid"], "%Y-%m-%d%H:%M"), axis=1
    )

    # drop uneeded coloums
    df = df.drop(columns=["Dato", "Tid"])

    # change date to index, in order to
    df.set_index("DateFormatted", inplace=True)

    # combine all 6 values for a given hour into its mean
    df = df.resample("H").mean()

    return df


def treat_trafikk_files(filename: str) -> pd.DataFrame:
    """
    Input: filename of a traffic data file

    Process:
    set date to index

    Output:
    a dataframe of the csv file
    """

    # read file as string
    with open(filename, "r") as f:
        my_csv_text = f.read()

    # replace | with ; to get uniform delimiter, and open to StringIO to be read by pandas
    csvStringIO = StringIO(my_csv_text.replace("|", ";"))

    # now that delimiter is uniform, file can be handled

    df = pd.read_csv(csvStringIO, delimiter=";", low_memory=False)

    # change to a uniform date -> see # Issues in README
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Fra"], "%Y-%m-%dT%H:%M%z").strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        axis=1,
    )

    # replace '-' with NaN and convert column into numeric
    df["Trafikkmengde"] = df["Trafikkmengde"].replace("-", np.nan).astype(float)

    # replace " " in 'Felt' values with "_" to avoid errors
    df["Felt"] = df["Felt"].str.replace(" ", "_")

    # dropping cols - see README on "Dropped coloumns"
    df = df.drop(
        columns=[
            "Trafikkregistreringspunkt",
            "Navn",
            "Vegreferanse",
            "Fra",
            "Til",
            "Dato",
            "Fra tidspunkt",
            "Til tidspunkt",
            "Dekningsgrad (%)",
            "Antall timer total",
            "Antall timer inkludert",
            "Antall timer ugyldig",
            "Ikke gyldig lengde",
            "Lengdekvalitetsgrad (%)",
            "< 5,6m",
            ">= 5,6m",
            "5,6m - 7,6m",
            "7,6m - 12,5m",
            "12,5m - 16,0m",
            ">= 16,0m",
            "16,0m - 24,0m",
            ">= 24,0m",
        ]
    )

    # drop all rows where the coloum "Felt" != "Totalt i retning Danmarksplass" or "Totalt i retning Florida"
    # the two other values for felt are "1" and "2" and are the same as the "Totalt ... Danmarkplass" and  "Totalt ... Florida"
    df = df[
        df["Felt"].isin(["Totalt_i_retning_Danmarksplass", "Totalt_i_retning_Florida"])
    ]

    # create empty dataframe with 'DateFormatted' as index
    result_df = pd.DataFrame(index=df["DateFormatted"].unique())

    # loop through unique 'Felt' values, filter original dataframe by 'Felt',
    # drop 'Felt' column and join to the result dataframe

    # so basically we sort of pivot the "Totalt i retning Danmarksplass" and "Totalt i retning Florida"
    # from being values, to them being coloums contaning the values in "trafikkmengde" (since we dropped all other cols)
    for felt in df["Felt"].unique():
        felt_df = (
            df[df["Felt"] == felt]
            .drop(columns="Felt")
            .add_suffix(
                f"_{felt}"
            )  # add suffix to column names to distinguish them when joining
            .set_index("DateFormatted_{0}".format(felt))
        )

        result_df = result_df.join(felt_df)

    return result_df


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
    traffic: the value of the traffic in a given hour

    Returns: df with more features
    """

    # BASIC DATE FEATURES
    # hour as own coloumn 0-23
    df["hour"] = df.index.hour  # get first two values

    # Instead of "day_in_week" being a num 0-6, add 7 coloumns to the dataframe, monday, tuesday .. etc
    # And have the value being 0/1 , 0 if it is not that day, 1 if it is

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
    df = pd.get_dummies(df, columns=["d"])  # convert to True/False
    # TODO: ADD THE COLOUMNS FOR ALL THE OTHER DAYS AND MAKE THEIR VALUE 0

    complete_days = [
        "d_Monday",
        "d_Tuesday",
        "d_Wednesday",
        "d_Thursday",
        "d_Friday",
        "d_Saturday",
        "d_Sunday",
    ]

    # iterate through the complete set of days
    for day in complete_days:
        # if the day is not a column in df, add it with a value of 0
        if not day in df.columns:
            df[day] = 0

    # ---------------
    df["month"] = df.index.month

    # MORE ADVANCED FEATURES

    # add weekend
    df["weekend"] = (df.index.weekday >= 5).astype(int)

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
    if not data2023:
        df = df.dropna(subset=["Total_trafikk"])

    # change all values of TRUE in all rows to 1 and FALSE to 0
    # models need NUMERIC data
    df = df.replace({True: 1, False: 0})

    # once we done with it drop month
    # df.drop("month",axis=1,inplace=True)

    return df


def merge_frames(frames: list) -> pd.DataFrame:
    """
    Given a list of dataframes, merges the frames to one large dataframe, given that the
    index is a date, and the same across all dataframes

    """

    # first DataFrame
    df_final = frames[0]

    # convert index (date) from string to datetime once only as we'll apply it to other frames
    df_final.index = pd.to_datetime(df_final.index)

    for frame in frames[1:]:
        # convert index from string to datetime
        frame.index = pd.to_datetime(frame.index)

        # since the dataframes have the same index, they can be merged!
        df_final = df_final.merge(frame, how="outer", left_index=True, right_index=True)

    df_final = df_final.sort_index()

    # remove lines where we dont have traffic information, as many missing values can cause model training
    # to overly rely on values which are there often.

    # Drops missing values
    PWD = Path().absolute()
    directory = f"{str(PWD)}/src/out"

    df_2023 = df_2023 = df_final.loc["2023-01-01":"2023-12-31"]

    # get where index is between 2023-01-01 00:00:00 and 2023-12-31 00:00:00

    df_final = df_final.dropna(subset=["Trafikkmengde_Totalt_i_retning_Florida"])

    # finding means of values lead to floating point errors, round to fix these
    df_final = df_final.apply(pd.to_numeric, errors="ignore").round(30)

    # combine the two traffic cols
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
    # Transform malformed data to NaN.
    length_dict = {"before": len(df)}

    length_dict["afterGlobal"] = len(df)

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
    # print(f"Number of NaNs in each column:\n{num_nan}")

    if not data2023:
        # reserve the "Total_trafikk" column, it will not used for imputation
        total_traffic_series = df["Total_trafikk"]

        # drop the "Total_trafikk" column from the main DataFrame
        df_no_traffic = df.drop(columns=["Total_trafikk"])

    if data2023:
        df_no_traffic = df

    # Drop "Relativ luftfuktighet" as this data only exists in 2022 and 2023.
    # errors="ignore" as most of the data (back to 2015) will not have this coloumn
    # this also leads to memory errors?
    # Drop "Relativ luftfuktighet" as this data only exists in 2022 and 2023.
    # errors="ignore" as most of the data (back to 2015) will not have this coloumn
    # this also leads to memory errors?

    # Save the DateFormatted column

    df_no_traffic = df_no_traffic.drop(
        columns=["Relativ luftfuktighet"], errors="ignore"
    )

    pkl_filename = "pickle_knn.pkl"

    try:
        with open(pkl_filename, "rb") as file:
            imputer = pickle.load(file)

            df_imputed = imputer.transform(df_no_traffic)
            print("PICKLE : FOUND PICKLE")

    except FileNotFoundError as e:
        print("DID NOT FIND PICKLE -> MAKING IT!")
        imputer = KNNImputer(n_neighbors=20, weights="distance")
        df_imputed = imputer.fit_transform(df_no_traffic)

        # pickle
        if len(df_no_traffic) > 40000:
            with open(pkl_filename, "wb") as file:
                print("PICKLE : PICKLING MADE MODEL")
                pickle.dump(imputer, file)

    # print("DF NO TRAFFIC -> ")
    # print(df_no_traffic)

    # print("DF IMPUTED ->>>>>>>>>>>>>")
    # print(df_imputed)

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

    # print(f"Values pre removal of outliers: {len(df)}")

    quant = df["Total_trafikk"].quantile(0.99)
    df = df[df["Total_trafikk"] <= quant]

    # print(f"Values post removal of outliers: {len(df)}")

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

    # df = df.reset_index()
    # df = df.drop(["DateFormatted"], axis=1)

    y = df["Total_trafikk"]
    x = df.drop(["Total_trafikk"], axis=1)

    # vi gjør at 70% blir treningsdata
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, shuffle=False, test_size=0.3
    )

    # deler opp 30% som var validation til 15% val og 15% test
    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, shuffle=False, test_size=0.5
    )

    # utforskende anaylse ser kun på trenignsdata -> bruk x_train/y_train
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


def treat_2023_file(df, model):
    df = df.drop(
        columns=[
            "Trafikkmengde_Totalt_i_retning_Danmarksplass",
            "Trafikkmengde_Totalt_i_retning_Florida",
        ]
    )

    df_fixed = trim_transform_outliers(df, True)

    # add important features to help the model
    df_final = feauture_engineer(df_fixed, True)
    # print("Features engineered")

    # normalize coloumns from 0-1 or square coloumns^2
    # df_final = normalize_cols(df_final)
    # print("Coloumns normalized")

    # drop coloumns which are not needed (noise)
    df_final = drop_uneeded_cols(df_final)
    # print("Uneeded cols dropped")

    try:
        df_final["Total_trafikk"] = model.predict(df_final)
    except ValueError as e:
        print(e)

    return df_final


def load_best_model() -> RandomForestRegressor:
    """
    Loads the best model
    """

    # if the model already exists as pickle, return it
    try:
        pickled_model = pickle.load(open("model.pkl", "rb"))
        return pickled_model
    except FileNotFoundError as e:
        pass

    print("INFO : No pre-existing model found... building from baseline")

    print("INFO : Starting parsing on loading best model ... ")
    # loop over files in local directory
    directory = f"{str(PWD)}/raw_data"  # change

    # multiple florida files will all be converted to df's, placed in this list, and concacted
    florida_df_list = []

    for filename in os.scandir(directory):
        if "Florida" in str(filename):
            florida_df = treat_florida_files(f"{str(directory)}/{filename.name}")
            florida_df_list.append(florida_df)

        if "trafikkdata" in str(filename):
            trafikk_df = treat_trafikk_files(f"{str(directory)}/{filename.name}")

    print("INFO : All files parsed!")

    # concat all the florida df's to one
    big_florida_df = pd.concat(florida_df_list, axis=0)
    print("INFO : Florida files concacted")

    # merge the dataframes
    df_2023, df_final = merge_frames([big_florida_df, trafikk_df])
    print("INFO : All files looped over")

    # divide data into training,test and validation
    split_dict_pre, training_df, test_df, validation_df = train_test_split_process(
        df_final
    )
    print("INFO : Training df received from test train split")

    dataframes_pre = {
        "training_df": training_df,
    }

    # loop through each data frame and transform the data
    # this is done seperatley for each df (test/train/validation)
    # so that the training data is never influced by the other data
    dataframes_post = {}

    for name, df_transforming in dataframes_pre.items():
        print(
            f"PARSING : Applying KNN imputer on missing data, and removing outliers.. for {name}"
        )
        print("PARSING : This could take a while...")

        # transform NaN and outliers to usable data
        df_transforming = trim_transform_outliers(df_transforming, False)
        print(f"PARSING : Outliers trimmed for {name}")

        # add important features to help the model
        df_transforming = feauture_engineer(df_transforming, False)
        print(f"PARSING : Features engineered for {name}")

        # normalize data outliers
        df_transforming = normalize_data(df_transforming)
        print(f"PARSING : Coloumns normalized for {name}")

        # drop coloumns which are not needed or redundant
        df_transforming = drop_uneeded_cols(df_transforming)
        print(f"PARSING : Uneeded cols dropped for {name}")

        dataframes_post[name] = df_transforming

    training_df = dataframes_post["training_df"]

    split_dict_post = {
        "y_train": training_df["Total_trafikk"],
        "x_train": training_df.drop(["Total_trafikk"], axis=1),
    }

    X_train = split_dict_post["x_train"]
    y_train = split_dict_post["y_train"]

    # BEST MODEL:
    print("MODEL : Training best model")
    best_model = RandomForestRegressor(n_estimators=181, random_state=2)
    best_model.fit(X_train, y_train)

    # save model as pickle
    pickle.dump(best_model, open("model.pkl", "wb"))

    return best_model


def prep_data_from_user(input_dict):
    """
    Loads the best model
    """
    print("INFO : Starting prep data from user ... ")

    col_keys = [
        "DateFormatted",
        "Globalstraling",
        "Solskinstid",
        "Lufttemperatur",
        "Vindretning",
        "Vindstyrke",
        "Lufttrykk",
        "Vindkast",
    ]

    df_dict = {}

    date_format = "%Y-%m-%d %H:%M:%S"

    for col in col_keys:
        try:
            if col in input_dict.keys():
                if input_dict[col] == "":
                    df_dict[col] = [np.nan]
                    continue

                if col == "DateFormatted":
                    df_dict[col] = [datetime.strptime(input_dict[col], date_format)]
                else:
                    df_dict[col] = [float(input_dict[col])]
            else:
                df_dict[col] = [np.nan]

        except (TypeError, ValueError) as e:
            print(e)
            return "ERROR"

    # print(df_dict)

    df = pd.DataFrame(df_dict)  # Converts input data into dataframe

    df["DateFormatted"] = pd.to_datetime(df["DateFormatted"])
    df.set_index("DateFormatted", inplace=True)

    name = "userinp"

    print(
        f"PARSING : Applying KNN imputer on missing data, and removing outliers.. for {name}"
    )
    print("PARSING : This could take a while...")

    # transform NaN and outliers to usable data
    df = trim_transform_outliers(df, True)
    print(f"PARSING : Outliers trimmed for {name}")

    # add important features to help the model
    df = feauture_engineer(df, True)
    print(f"PARSING : Features engineered for {name}")

    # drop coloumns which are not needed or redundant
    df = drop_uneeded_cols(df)
    print(f"PARSING : Uneeded cols dropped for {name}")

    print("PARSING : re-aranging df to fit")

    df = df[
        [
            "Globalstraling",
            "Solskinstid",
            "Lufttemperatur",
            "Lufttrykk",
            "Vindkast",
            "hour",
            "d_Friday",
            "d_Monday",
            "d_Saturday",
            "d_Sunday",
            "d_Thursday",
            "d_Tuesday",
            "d_Wednesday",
            "month",
            "weekend",
            "public_holiday",
            "raining",
            "summer",
            "winter",
            "rush_hour",
            "sleeptime",
            "Vindretning_x",
            "Vindretning_y",
        ]
    ]
    return df


if __name__ == "__main__":
    """
    Example run of prep data
    """

    best_model = load_best_model()

    input_dict = {
        "DateFormatted": "2023-01-01 08:00:00",
        "Globalstraling": "-0.3166666666666667",
        "Solskinstid": "0.0",
        "Lufttemperatur": "-2.566666666666667",
        "Vindretning": "320",
        "Vindstyrke": "10",
        "Lufttrykk": "910",
        "Vindkast": "12",
    }

    df = prep_data_from_user(input_dict)
    prediction = best_model.predict(df)

    print(f"Prediction = {int(prediction[0])}")

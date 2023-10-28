import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils.dataframe_handling import (
    drop_uneeded_cols,
    feauture_engineer,
    merge_frames,
    normalize_data,
    train_test_split_process,
    treat_2023_file,
    trim_transform_outliers,
)
from utils.file_parsing import treat_florida_files, treat_trafikk_files
from utils.graphing import (
    graph_a_vs_b,
    graph_all_models,
    graph_hour_diff,
    graph_hour_variance,
    graph_monthly_amounts,
    graph_weekly_amounts,
)
from utils.models import (
    find_hyper_param,
    find_hyper_param_further,
    train_best_model,
    train_models,
)

# get current filepath to use when opening/saving files
PWD = Path().absolute()
DEBUG = False
GRAPHING = False
TRAIN_MANY = False
FINAL_RUN = False
RANDOM_STATE = 2


def main():
    print("INFO : Starting parsing ... ")
    # loop over files in local directory
    directory = f"{str(PWD)}/raw_data"

    # multiple florida files will all be converted to df's, placed in this list, and concacted
    florida_df_list = []

    # parse files
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
    print("INFO : All files merged over")

    # divide data into training,test and validation
    split_dict_pre, training_df, test_df, validation_df = train_test_split_process(
        df_final
    )
    print("INFO : Data divided into training,validation and test")

    # average traffic per year to observe
    average_traffic_per_year = (
        training_df["Total_trafikk"].groupby(training_df.index.year).mean()
    )

    print("INFO : Average traffic per year (for training data):")
    print(average_traffic_per_year)

    print("INFO : Description of data PRE PROCESSING")
    print(training_df.describe())

    if GRAPHING:
        graph_all_models(training_df, pre_change=True)
        print("INFO : Graphed all models PRE-CHANGE")

    # make dataframe dict to treat them differently
    dataframes_pre = {
        "training_df": training_df,
        "validation_df": validation_df,
        "test_df": test_df,
    }

    # loop through each data frame and transform the data
    # this is done seperatley for each df (test/train/validation)
    # so that the training data is never influced by the other data
    dataframes_post = {}

    for name, df_transforming in dataframes_pre.items():
        print(
            f"INFO : Applying KNN imputer on missing data, and removing outliers.. for {name}"
        )
        print("INFO : This could take a while...")

        # transform NaN and outliers to usable data
        df_transforming = trim_transform_outliers(df_transforming, False)
        print(f"INFO : Outliers trimmed for {name}")

        # add features to help the model
        df_transforming = feauture_engineer(df_transforming, False)
        print(f"INFO : Features engineered for {name}")

        # normalize data outliers
        df_transforming = normalize_data(df_transforming)
        print(f"INFO : Coloumns normalized for {name}")

        if GRAPHING:
            if name == "training_df":
                # graph Vindkast vs
                graph_a_vs_b(
                    "POST_CHANGES",
                    df_transforming,
                    "Vindstyrke",
                    "Vindkast",
                    "Styrke",
                    "Kast",
                )
                graph_a_vs_b(
                    "POST_CHANGES",
                    df_transforming,
                    "Vindstyrke",
                    "Total_trafikk",
                    "Retning",
                    "Sykler",
                )
                graph_a_vs_b(
                    "POST_CHANGES",
                    df_transforming,
                    "Vindretning",
                    "Total_trafikk",
                    "Retning",
                    "Sykler",
                )

        # drop coloumns which are not needed or redundant
        df_transforming = drop_uneeded_cols(df_transforming)
        print(f"INFO : Uneeded cols dropped for {name}")

        # save dataframes to use later
        dataframes_post[name] = df_transforming

    training_df = dataframes_post["training_df"]
    test_df = dataframes_post["test_df"]
    validation_df = dataframes_post["validation_df"]

    # save training data to csv to have a look
    training_df.to_csv(f"{PWD}/out/main_training_data.csv")
    print("INFO : Data saved to CSV")

    print("INFO : Description of data POST PROCESSING")
    print(training_df.describe())

    if GRAPHING:
        # Graph post data processing to visualize and analyze data
        print("GRAPHING : GRAPHING HOUR DIFF")
        graph_hour_diff(training_df)
        print("GRAPHING : GRAPHING WEEKLY AMOUNTS")
        graph_weekly_amounts(training_df)
        print("GRAPHING : GRAPHING MONTHLY AMOUNTS")
        graph_monthly_amounts(training_df)
        print("GRAPHING : GRAPHING HOUR VARIANCE")
        graph_hour_variance(training_df)

        graph_all_models(training_df, pre_change=False)
        print("INFO : Graph all models POSTCHANGE")

    split_dict_post = {
        "y_train": training_df["Total_trafikk"],
        "x_train": training_df.drop(["Total_trafikk"], axis=1),
        "y_val": validation_df["Total_trafikk"],
        "x_val": validation_df.drop(["Total_trafikk"], axis=1),
        "y_test": test_df["Total_trafikk"],
        "x_test": test_df.drop(["Total_trafikk"], axis=1),
    }

    # train models
    if TRAIN_MANY:
        train_models(split_dict_post)
        # find hyper params for the best model
        find_hyper_param(split_dict_post)
        find_hyper_param_further(split_dict_post)

    # train the best model on validation data
    print("INFO: Training model on validation data")
    train_best_model(split_dict_post, test_data=False)

    if FINAL_RUN:
        # train best model on test data
        print("INFO: Training model on test data")
        train_best_model(split_dict_post, test_data=True)

        print("INFO : Treating 2023 files")

        # use the best model to get values for 2023
        best_model = RandomForestRegressor(n_estimators=181, random_state=RANDOM_STATE)

        X_train = split_dict_post["x_train"]
        y_train = split_dict_post["y_train"]

        # the best model is used to treat 2023 files.
        best_model.fit(X_train, y_train)
        df_with_values = treat_2023_file(df_2023, best_model)

    return split_dict_post, training_df, test_df, validation_df


def create_dirs() -> None:
    """
    Helper function to create directories for saving figs and files
    """
    try:
        os.mkdir("figs")
    except FileExistsError:
        pass

    try:
        os.mkdir("out")
    except FileExistsError:
        pass


if __name__ == "__main__":
    create_dirs()
    split_dict, training_df, test_df, validation_df = main()
    print("INFO : main function complete!")

from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# get current filepath to use when opening/saving files
PWD = Path().absolute()


def treat_florida_files(filename: str) -> pd.DataFrame:
    """
    Input:
        filename: filename of a florida weather file

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
    Input:
        filename: filename of a traffic data file

    Output:
        a dataframe of the csv file
    """

    # read file as string
    with open(filename, "r") as f:
        my_csv_text = f.read()

    # replace | with ; to get uniform delimiter, and open to StringIO to be read by pandas
    csvStringIO = StringIO(my_csv_text.replace("|", ";"))

    # now that delimiter is uniform, file can be handled
    df = pd.read_csv(csvStringIO, delimiter=";")

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
    df["Felt"] = df["Felt"].str.replace(" ", "_")  # todo try without

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
    df = df[
        df["Felt"].isin(["Totalt_i_retning_Danmarksplass", "Totalt_i_retning_Florida"])
    ]

    # create empty dataframe with 'DateFormatted' as index
    result_df = pd.DataFrame(index=df["DateFormatted"].unique())

    # What this is essentially doing is transforming "Totalt_i_retning_Danmarksplass" and "Totalt_i_retning_Florida" from being
    # values in a coloumn called "Felt", to being two columns. Their values for the hours are just the same
    # The "felt" coloumn is removed and has been transformed into two different "Felt" coloumns.

    # loop through each unique felt value in the df, in this case "Totalt_i_retning_Danmarksplass" and "Totalt_i_retning_Florida"
    for felt in df["Felt"].unique():
        # filter the dataframe so where the coloumn "felt" = the felt we have chosen for this iteration
        felt_df = df[df["Felt"] == felt]

        # remove felt, since we are making new cols
        felt_df = felt_df.drop(columns="Felt")

        # the name should be different for the two felt
        felt_df = felt_df.add_suffix(f"_{felt}")
        felt_df = felt_df.set_index(f"DateFormatted_{felt}")

        # put the filtered dataframe onto the result one.
        # after doing this for both they should line up nicely
        result_df = result_df.join(felt_df)

    return result_df

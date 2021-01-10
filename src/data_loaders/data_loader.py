from typing import List

import pandas as pd
from scipy.io import arff


def choose_class_column_name(meta: arff.MetaData, class_column: str = None):
    # if class_column is not given, try to guess column's name
    if not class_column:
        names = set(["class", "CLASS", "label", "LABEL"]) & set(meta.names())
        try:
            class_column = names.pop()
        except KeyError:
            raise ValueError(
                f"It is not possible to guess name of column that represents class."
            )

    if class_column not in meta.names():
        raise ValueError(f"There is no column called ({class_column}) in given file.")

    return class_column


def choose_coordinates_columns_names(
    meta: arff.MetaData, coordinates_columns: List[str] = None
):
    if not coordinates_columns:
        coordinates_columns = [
            name for name in meta.names() if meta[name][0] == "numeric"
        ]

    missing_columns = set(coordinates_columns) - set(meta.names())
    if missing_columns:
        raise ValueError(
            f"Given files does not contain all required attributes: {missing_columns}."
        )

    return coordinates_columns


def load_data(filename, coordinates_columns=None, class_column=None):
    """
    Load data from .arff file called `filename`.

    Args:
        filename: .arff file with dataset
        coordinates_columns: names of columns with coordinates
        class_column: class, label

    Returns:
        Dictionary with data required for clustering algorithms:
        "df": dataset
        "class_column": name of column with labels
        "coordinates_columns": list of columns with coordinates
        "possible_classes": list of possible class values
        Dataframe with data, possible classes (ex. "1", "2", "3", etc.).

    """
    # load data from .arff file
    data, meta = arff.loadarff(filename)

    class_column = choose_class_column_name(meta, class_column)
    coordinates_columns = choose_coordinates_columns_names(meta, coordinates_columns)

    # create dataframe: convert labels to string and coordinates to floats
    df = pd.DataFrame(data)
    df[class_column] = df[class_column].str.decode("utf-8")
    for col in coordinates_columns:
        df.astype({col: "float"}).dtypes

    result = {
        "df": df,
        "class_column": class_column,
        "coordinates_columns": list(coordinates_columns),
        "classes": list(meta[class_column][1]),
    }
    return result

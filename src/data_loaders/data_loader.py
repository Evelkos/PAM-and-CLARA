import pandas as pd
from scipy.io import arff


def load_data(filename):
    """
    Load data from .arff file called `filename`.

    Args:
        filename: .arff file with dataset

    Returns:
        Dataframe with data, possible classes (ex. "1", "2", "3", etc.).

    """
    # load data from .arff file
    data, meta = arff.loadarff(filename)

    # create dataframe and change columns names
    df = pd.DataFrame(data)
    df = df.rename(columns={"a0": "x", "a1": "y", "CLASS": "label"})
    df.label = pd.to_numeric(df.label)

    df.astype({"x": "float"}).dtypes
    df.astype({"y": "float"}).dtypes
    df.astype({"label": "str"}).dtypes

    classes = meta["CLASS"][1]

    return df, classes

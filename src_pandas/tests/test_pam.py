import pandas as pd

from src.pam import PAM


def test_get_initial_medoids_indices_for_two_clusters():
    df = pd.DataFrame(
        data={
            "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    pam = PAM(df, 2)

    assert pam.get_initial_medoids_indices(33) == [9, 2]
    assert pam.get_initial_medoids_indices(44) == [6, 8]
    assert pam.get_initial_medoids_indices(124128) == [6, 7]

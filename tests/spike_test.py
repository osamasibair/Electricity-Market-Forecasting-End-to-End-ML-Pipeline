import pandas as pd
from spikes import label_spikes


def test_label_spikes():
    df = pd.DataFrame({"price": [150.0, 100.0, 90.0], "price_roll_7d": [100.0, 100.0, 100.0]})
    assert list(label_spikes(df, 40)) == [1, 0, 0]
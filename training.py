import numpy as np
import data_tools
import pandas as pd


def get_best_rule(x_train: pd.DataFrame, y_train: pd.DataFrame) -> dict[str, any]:
    """
     Returns a dictionary with the best tree split rule
     given the avaliable features and thresholds. The dictionary
     includes both the feature the rule is based on, as well as the
     threshold for which the split occurs, using the keys "feature" and
     "threshold" accordingly.
     """
    best_feature, best_threshold = None
    min_rss = np.inf
    for feature in x_train.columns:
        threshold = x_train[feature].unique().tolist()
        threshold.sort()
        thresholds = thresholds[1:]
        for thresh in thresholds:
            y_left_split = x_train[feature] < thresh
            y_left, y_right = y_train[y_left_split], y_train[~y_left_split]
            thresh_rss = data_tools.double_rss_sum(y_left, y_right)
            if thresh_rss < min_rss:
                min_rss = thresh_rss
                best_threshold = thresh
                best_feature = feature
    return {'feature': best_feature, 'threshold': best_threshold}

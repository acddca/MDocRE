# -*- coding:utf-8 -*-
# @Author: wangjl
# @Time: 2022/11/17 17:50
# @File: utils.py
# @Email: wangjl.nju.2020@gmail.com.
from typing import Dict, Any
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd


def classification_report(y_true, y_pred) -> Dict[str, Any]:

    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    # total accuracy
    accuracy = (y_true == y_pred).mean()
    keys = np.unique(y_true)
    scores = precision_recall_fscore_support(
        y_true, y_pred, labels=list(keys), zero_division=0
    )
    df = pd.DataFrame(
        scores, columns=keys, index=["precision", "recall", "f-score", "support"]
    )
    marco_scores = df.mean(axis=1)

    return {
        "accuracy": accuracy,
        "precision": marco_scores["precision"],
        "recall": marco_scores["recall"],
        "fscore": marco_scores["f-score"],
        "detailed": df.to_dict(),
    }
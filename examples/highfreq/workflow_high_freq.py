#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import (
    backtest as normal_backtest,
    risk_analysis,
)
from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.tests.data import GetData
import pickle
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.data import Cal
from qlib.tests.data import GetData
from qlib.config import REG_CN, HIGH_FREQ_CONFIG

import sys
sys.path.append("./highfreq")

from highfreq_ops import get_calendar_day, DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut

if __name__ == "__main__":

    SPEC_CONF = {"custom_ops": [DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut], "expression_cache": None}

    MARKET = "all"
    BENCHMARK = "SH000300"

    start_time = "2020-09-15 00:00:00"
    end_time = "2021-01-18 16:00:00"
    train_end_time = "2020-11-30 16:00:00"
    test_start_time = "2020-12-01 00:00:00"

    DATA_HANDLER_CONFIG0 = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": train_end_time,
        "instruments": MARKET,
        "infer_processors": [{"class": "HighFreqNorm", "module_path": "highfreq_processor", "kwargs": {}}],
    }
    DATA_HANDLER_CONFIG1 = {
        "start_time": start_time,
        "end_time": end_time,
        "instruments": MARKET,
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG0,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (
                        test_start_time,
                        end_time,
                    ),
                },
            },
        },
        "dataset_backtest": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqBacktestHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG1,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (
                        test_start_time,
                        end_time,
                    ),
                },
            },
        },
    }

    """initialize qlib"""
    # use yahoo_cn_1min data
    QLIB_INIT_CONFIG = {**HIGH_FREQ_CONFIG, **SPEC_CONF}
    print(QLIB_INIT_CONFIG)
    provider_uri = QLIB_INIT_CONFIG.get("provider_uri")
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        GetData().qlib_data(target_dir=provider_uri, interval="1min", region=REG_CN)
    qlib.init(**QLIB_INIT_CONFIG)

    Cal.calendar(freq="1min")
    get_calendar_day(freq="1min")

    # get data
    dataset = init_instance_by_config(task["dataset"])
    xtrain, xtest = dataset.prepare(["train", "test"])
    print(xtrain, xtest)
    xtrain.to_csv("xtrain.csv")

    dataset_backtest = init_instance_by_config(task["dataset_backtest"])
    backtest_train, backtest_test = dataset_backtest.prepare(["train", "test"])
    print(backtest_train, backtest_test)


    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    # example_df = dataset.prepare("train")
    # print(example_df.head(10))
    # df_train, df_valid = dataset.prepare(
    #     ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    # )
    # with open("../example_train.pkl", "wb") as f:
    #     pickle.dump(df_train, f)
    # with open("../example_valid.pkl", "wb") as f:
    #     pickle.dump(df_valid, f)

    # start exp
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

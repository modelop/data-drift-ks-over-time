import json
import pandas as pd
import modelop.utils as utils
import modelop.schema.infer as infer
import numpy
import modelop.monitors.drift as drift
from pathlib import Path
from scipy.stats import ks_2samp

logger = utils.configure_logger()


#
# This method gets called when the monitor is loaded by the ModelOp runtime. It sets the GLOBAL values that are
# extracted from the report.txt to obtain the DTS and version info to append to the report
#

# modelop.init
def init(init_param):
    logger = utils.configure_logger()
    global NUMERICAL_COLUMNS
    global CATEGORICAL_COLUMNS
    global PREDICTION_DATE_COLUMN
    global JOB

    JOB = init_param
    job = json.loads(init_param['rawJson'])
    PREDICTION_DATE_COLUMN = job.get('jobParameters', {}).get('predictionDateColumn', "")

    input_schema_definition = infer.extract_input_schema(JOB)
    monitoring_parameters = infer.set_monitoring_parameters(
        schema_json=input_schema_definition, check_schema=True
    )
    NUMERICAL_COLUMNS = monitoring_parameters['numerical_columns']
    CATEGORICAL_COLUMNS = monitoring_parameters['categorical_columns']


def fix_numpy_nans_and_infs_in_dict(val: float) -> float:
    """A function to change numpy.nan or numpy.inf values to python Nones.

    Args:
        val: Input to fix.

    Returns:
        val: Fixed values.
    """

    # If value is numeric (not None), check for numpy.nan and numpy.inf
    # If True, change to None, else keep unchanged
    if val is not None:
        try:  # Some values are not numeric
            if numpy.isnan(val):
                val = None
            elif numpy.isinf(val):
                logger.warning(
                    "Infinity encountered while computing %s on column %s! Setting value to None.",
                    val
                )
                val = None
        except TypeError:
            pass

    return val

#
# This method is the modelops metrics method.  This is always called with a pandas dataframe that is arraylike, and
# contains individual rows represented in a dataframe format that is representative of all of the data that comes in
# as the results of the first input asset on the job.  This method will not be invoked until all data has been read
# from that input asset.
#
# INPUTS: For this metrics model, it takes in the baseline_data and the comparator_data.
#

# modelop.metrics
def metrics(df_baseline: pd.DataFrame, df_comparator: pd.DataFrame):
    logger.info("Running the metrics function")
    final_result = {}
    p_values_by_day_data = {}

    df_comparator[PREDICTION_DATE_COLUMN] = pd.to_datetime(df_comparator[PREDICTION_DATE_COLUMN]).dt.strftime(
        "%Y-%m-%d")
    day_list = sorted(df_comparator[PREDICTION_DATE_COLUMN].unique())
    final_result["firstPredictionDate"] = day_list[0]
    final_result["lastPredictionDate"] = day_list[len(day_list)-1]

    for feat in NUMERICAL_COLUMNS:
        feature_pvalue_array = []
        for day in day_list:
            df_comparator_current_day = df_comparator.loc[df_comparator[PREDICTION_DATE_COLUMN] == day]
            pvalue_result = ks_2samp(data1=df_baseline.loc[:, feat], data2=df_comparator_current_day.loc[:, feat])

            # Cast numpy.nan and numpy.inf values (if any) to python None for JSON encoding
            pvalue_result = fix_numpy_nans_and_infs_in_dict(
                val=pvalue_result[1]
            )
            feature_pvalue_array.append([day, float(pvalue_result.round(4))])

        p_values_by_day_data[feat] = feature_pvalue_array

    final_result["ks_p_values_by_day"] = {"title": "Data Drift Over Time - KS", "x_axis_label": "Day",
                                          "y_axis_label": "KS P-Value", "data": p_values_by_day_data}

    # Calculate full data profile of baseline and comparator data sets
    drift_detector = drift.DriftDetector(
        df_baseline=df_baseline, df_sample=df_comparator, job_json=JOB
    )
    full_data_profile = drift_detector.calculate_drift(pre_defined_test="Summary")
    final_result["data_drift"] = full_data_profile["data_drift"]

    yield final_result


#
# This main method is utilized to simulate what the engine will do when calling the above metrics function.  It takes
# the json formatted data, and converts it to a pandas dataframe, then passes this into the metrics function for
# processing.  This is a good way to develop your models to be conformant with the engine in that you can run this
# locally first and ensure the python is behaving correctly before deploying on a ModelOp engine.
#
def main():
    raw_json = Path('example_job.json').read_text()
    init_param = {'rawJson': raw_json}

    init(init_param)
    df1 = pd.read_csv("german_credit_data.csv")
    df2 = pd.read_csv("german_credit_data2.csv")
    print(json.dumps(next(metrics(df1, df2)), indent=2))


if __name__ == '__main__':
    main()

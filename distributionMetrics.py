import json
import pandas as pd
import modelop.utils as utils
import modelop.schema.infer as infer
import numpy
import modelop.monitors.drift as drift
import re
import time
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
    global PREDICTION_DATE_COLUMN
    global KS_THRESHOLD
    global JOB

    JOB = init_param
    job = json.loads(init_param['rawJson'])

    # Obtain the "prediction date column" from the job parameters. The user should add a job parameter in the UI called
    # "predictionDateColumn" and the value should be the name of the field/column in the comparator (production) data
    # set that contains the predictionDates for each record
    PREDICTION_DATE_COLUMN = job.get('jobParameters', {}).get('predictionDateColumn', "")

    # Obtain the threshold above which, the KS test will fail. If none provided in the job parameters, then set to a
    # default value
    KS_THRESHOLD = job.get('jobParameters', {}).get('ksThreshold', "")
    if KS_THRESHOLD:
        logger.info("KS Threshold extracted from job parameters, using this threshold: " + str(KS_THRESHOLD))
    else:
        KS_THRESHOLD = 0.05
        logger.info(
            'Did not find a KS Threshold in the job parameters, using this default threshold: ' + str(KS_THRESHOLD))

    input_schema_definition = infer.extract_input_schema(JOB)
    monitoring_parameters = infer.set_monitoring_parameters(
        schema_json=input_schema_definition, check_schema=True
    )
    NUMERICAL_COLUMNS = monitoring_parameters['numerical_columns']


# A function to change numpy.nan or numpy.inf values to python Nones
#
# INPUTS: Input to fix.
# RETURNS: Fixed values
def fix_numpy_nans_and_infs_in_dict(val: float) -> float:
    # If value is numeric (not None), check for numpy.nan and numpy.inf
    # If True, change to None, else keep unchanged
    if val is not None:
        try:  # Some values are not numeric
            if numpy.isnan(val):
                val = None
            elif numpy.isinf(val):
                logger.warning("Infinity encountered while computing %s on column %s! Setting value to None.", val)
                val = None
        except TypeError:
            pass

    return val


# Function to check if the prediction date column contains data of the format YYYY-ww
#
# INPUTS: text, typically a column that contains a date
# RETURNS: TRUE if it is of the format YYYY-ww
def check_if_week_format(input_text):
    pattern = re.compile(r"^([1-2][0-9]{3}-[0-9]{2})$", re.IGNORECASE)
    return pattern.match(input_text)


# Function to run Kolmogorov-Smirnov (KS) 2-sample t-test
#
# INPUTS: baseline_data and a comparator_data set
# RETURNS: KS p-value (float)
def run_ks_test(df_baseline: pd.DataFrame, df_comparator: pd.DataFrame):
    # Run Kolmogorov-Smirnov 2-sample t-test
    pvalue_result = ks_2samp(data1=df_baseline, data2=df_comparator)

    # Cast numpy.nan and numpy.inf values (if any) to python None for JSON encoding
    pvalue_result = fix_numpy_nans_and_infs_in_dict(val=pvalue_result[1])

    return pvalue_result


#
# This method is the modelops metrics method.  This is always called with a pandas dataframe that is arraylike, and
# contains individual rows represented in a dataframe format that is representative of all of the data that comes in
# as the results of the first input asset on the job.  This method will not be invoked until all data has been read
# from that input asset.
#
# INPUTS: For this metrics model, it takes in the baseline_data and the comparator_data.
# RETURNS: dict() of values for a ModelOp Metrics job, that is typically converted into a Model Test Result

# modelop.metrics
def metrics(df_baseline: pd.DataFrame, df_comparator: pd.DataFrame):
    logger.info("Running the metrics function")
    final_result = {}
    p_values_by_day_data = {}
    run_across_entire_data = False

    if PREDICTION_DATE_COLUMN in df_comparator:
        # The data contains a date column, so this monitor will compute the drift metrics for each date period

        # Check if it is of the format of YYYY-ww
        week_format_used = check_if_week_format(df_comparator.iloc[0][PREDICTION_DATE_COLUMN])

        if week_format_used:
            logger.info("Detected a prediction date column format of YYYY-WW. This will be used")
            df_comparator[PREDICTION_DATE_COLUMN] = df_comparator[PREDICTION_DATE_COLUMN].apply(
                lambda x: time.strftime("%Y-%m-%d", time.strptime(x + '-1', "%G-%V-%u")))
        else:
            try:
                # Format the prediction date column to be of the format YYYY-mm-dd
                df_comparator[PREDICTION_DATE_COLUMN] = pd.to_datetime(
                    df_comparator[PREDICTION_DATE_COLUMN]).dt.strftime("%Y-%m-%d")
                logger.info("Successfully able to extract the prediction date column and format the date column to "
                            "YYYY-mm-dd format")
            except ValueError as e:
                logger.warning("Could not convert the prediction date column format. Please use a standard ISO "
                               "format. Defaulting to running the metrics on the entire data set")
                run_across_entire_data = True
    else:
        # Prediction Date column was not provided, so the drift test will run across the entire data set, instead of
        # calculating the drift values for each day
        logger.info("No prediction date column provided. Running calculations across the entire data set")
        run_across_entire_data = True

    # CASE 1: Run the drift test across the entire data set (a date column was not provided)
    if run_across_entire_data:
        # For each feature, calculate the KS 2-sample t-test between the entire production (comparator) data set
        # as compared against the baseline data set
        feature_pvalue_array = []

        # Create a table of all failed runs
        ks_failures_current_run = []

        for feat in NUMERICAL_COLUMNS:
            # Call the KS 2-sample t-test for a given feature
            pvalue_result = run_ks_test(df_baseline.loc[:, feat], df_comparator.loc[:, feat])

            # Add the [feature, p-value] pair to the array of p-values for a given feature
            feature_pvalue_array.append({"feature": feat, "KS_P-Value": float(pvalue_result.round(4))})

            # Check for failures against the threshold. If they exist, add it to the running array
            failure_details_object = {}
            if pvalue_result > KS_THRESHOLD:
                failure_details_object = {"Feature": feat, "KS_P-Value": pvalue_result,
                                          "Amount_Above_Threshold": (pvalue_result - KS_THRESHOLD).round(4)}
                count_nulls = str(df_comparator.loc[:, feat].isna().sum())
                failure_details_object = utils.merge(failure_details_object,
                                                     df_comparator.loc[:, feat].describe())
                failure_details_object["Count_Nulls"] = count_nulls

                # print("local failure object is: ", failure_details_object)
                ks_failures_current_run.append(failure_details_object)

        # Create a Table of the failures
        drift_failures_table = {"Drift_Failures_By_Feature": ks_failures_current_run}

        # Create the Graph of data drift metrics for the full data set
        final_result["Data_Drift_Metrics_Full_Data_Set"] = {"title": "Data Drift Across Production Data Set - "
                                                                     "Kolmorogov-Smirnov", "x_axis_label": "Day",
                                                            "y_axis_label": "KS P-Value", "data": feature_pvalue_array}

    else:
        # CASE 2: Run the drift test for each day in the data set
        logger.info("Running KS for the given days")

        # Get all unique days across the production (comparator) data set
        dates_list = sorted(df_comparator[PREDICTION_DATE_COLUMN].unique())

        # Add the first and last prediction date to the test result, which can be used for quick aggregation of test
        # results
        final_result["firstPredictionDate"] = dates_list[0]
        final_result["lastPredictionDate"] = dates_list[len(dates_list) - 1]

        # Create a table of all failed runs
        ks_failures_current_run = []

        # For each feature, calculate the KS 2-sample t-test for each prediction day in the production (comparator)
        # data set as compared against the baseline data set
        for feat in NUMERICAL_COLUMNS:
            feature_pvalue_array = []

            for date_item in dates_list:
                # Create a an object for each failed KS test for each date in the list
                failure_details_object = {}

                # Run the KS test using the current date
                df_comparator_current_day = df_comparator.loc[df_comparator[PREDICTION_DATE_COLUMN] == date_item]
                pvalue_result = run_ks_test(df_baseline.loc[:, feat], df_comparator_current_day.loc[:, feat])

                # Add the [day, p-value] pair to the array of p-values for a given feature
                feature_pvalue_array.append([date_item, float(pvalue_result.round(4))])

                # Check for failures against the threshold. If they exist, add it to the running array of failures
                failure_details_object = {}
                if pvalue_result > KS_THRESHOLD:
                    failure_details_object = {"Feature": feat, "Date": date_item,
                                              "KS_P-Value": float(pvalue_result.round(4)),
                                              "Amount_Above_Threshold": float((pvalue_result - KS_THRESHOLD).round(4))}
                    count_nulls = str(df_comparator_current_day.loc[:, feat].isna().sum())
                    failure_details_object = utils.merge(failure_details_object,
                                                         df_comparator_current_day.loc[:, feat].describe())
                    failure_details_object["Count_Nulls"] = count_nulls

                    # print("local failure object is: ", failure_details_object)
                    ks_failures_current_run.append(failure_details_object)

            # Add the pvalues per date to the final object for showing drift results over time
            p_values_by_day_data[feat] = feature_pvalue_array

        # Create a Table of the failures
        drift_failures_table = {"Drift_Failures_By_Feature": ks_failures_current_run}

        # Create the "Data Drift over Time" line chart in the final result
        final_result["Data_Drift_Metrics_By_Day"] = {"title": "Data Drift Over Time - Kolmorogov-Smirnov",
                                                     "x_axis_label": "Day",
                                                     "y_axis_label": "KS P-Value", "data": p_values_by_day_data}

    # Use the OOTB drift package to calculate additional drift metrics
    drift_detector = drift.DriftDetector(
        df_baseline=df_baseline, df_sample=df_comparator, job_json=JOB
    )
    # Run full data profile of the data sets
    full_data_profile = drift_detector.calculate_drift(pre_defined_test="Summary")

    # Create the table of Kolmogorov-Smirnov p-values per feature
    ks_drift_metrics = drift_detector.calculate_drift(pre_defined_test="Kolmogorov-Smirnov",
                                                      flattening_suffix="_ks_pvalue")

    # Merge all drift metrics results for proper display in ModelOp
    final_result_all = utils.merge(drift_failures_table, ks_drift_metrics, full_data_profile, final_result)

    yield final_result_all


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
    df1 = pd.read_csv("german_credit_data3.csv")
    df2 = pd.read_csv("german_credit_data4.csv")
    print(json.dumps(next(metrics(df1, df2)), indent=2))


if __name__ == '__main__':
    main()

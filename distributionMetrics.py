import json
import pandas as pd
import modelop.utils as utils
import modelop.schema.infer as infer
import numpy
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
    job_json = init_param

    input_schema_definition = infer.extract_input_schema(job_json)
    monitoring_parameters = infer.set_monitoring_parameters(
        schema_json=input_schema_definition, check_schema=True
    )
    NUMERICAL_COLUMNS = monitoring_parameters['numerical_columns']

def fix_numpy_nans_and_infs_in_dict(values: dict, test_name: str) -> dict:
    """A function to change all numpy.nan and numpy.inf values in a flat dictionary to python Nones.

    Args:
        values (dict): Input dict to fix.
        test_name (str):  Name of test that's calling this function.

    Returns:
        dict: Fixed dict.
    """

    for key, val in values.items():
        # If value is numeric (not None), check for numpy.nan and numpy.inf
        # If True, change to None, else keep unchanged
        if val is not None:
            try:  # Some values are not numeric
                if numpy.isnan(val):
                    values[key] = None
                elif numpy.isinf(val):
                    logger.warning(
                        "Infinity encountered while computing %s on column %s! Setting value to None.",
                        test_name,
                        key,
                    )
                    values[key] = None
            except TypeError:
                pass

    return values

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

    ks_tests = []
    for feat in NUMERICAL_COLUMNS:
        logger.info("Computing KS on numerical_column %s", feat)
        ks_tests.append(ks_2samp(data1=df_baseline.loc[:, feat], data2=df_comparator.loc[:, feat]))

    pvalues = [x[1].round(4) for x in ks_tests]

    ks_pvalues = dict(zip(NUMERICAL_COLUMNS, pvalues))

    # Cast numpy.nan and numpy.inf values (if any) to python None for JSON encoding
    ks_pvalues = fix_numpy_nans_and_infs_in_dict(
        values=ks_pvalues, test_name="Kolmogorov-Smirnov p-value"
    )

    yield ks_pvalues

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

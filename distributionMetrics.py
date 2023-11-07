import json
import pandas as pd
import modelop.utils as utils
import modelop.schema.infer as infer
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
    job_json = init_param

    input_schema_definition = infer.extract_input_schema(job_json)
    monitoring_parameters = infer.set_monitoring_parameters(
        schema_json=input_schema_definition, check_schema=True
    )
    NUMERICAL_COLUMNS = monitoring_parameters['numerical_columns']
    CATEGORICAL_COLUMNS = monitoring_parameters['categorical_columns']


#
# This method is the modelops metrics method.  This is always called with a pandas dataframe that is arraylike, and
# contains individual rows represented in a dataframe format that is representative of all of the data that comes in
# as the results of the first input asset on the job.  This method will not be invoked until all data has been read
# from that input asset.
#
# For this example, we simply echo back the first row of that data as a json object.  This is useful for things like
# reading externally generated metrics from an SQL database or an S3 file and having them interpreted as a model test
# result for the association of these results with a model snapshot.
#
# data - The input data of the first input asset of the job, as a pandas dataframe
#

# modelop.metrics
def metrics(df_baseline: pd.DataFrame, df_sample: pd.DataFrame):
    logger.info("Running the metrics function")

    ks_tests = []
    for feat in NUMERICAL_COLUMNS:
        logger.info("Computing KS on numerical_column %s", feat)
        ks_tests.append(ks_2samp(data1=df_baseline.loc[:, feat], data2=df_sample.loc[:, feat]))

    pvalues = [x[1].round(4) for x in ks_tests]

    ks_pvalues = dict(zip(NUMERICAL_COLUMNS, pvalues))

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

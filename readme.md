# Data Drift over Time Monitor
This metrics model calculates data drift between two data sets using the Kolmogorov-Smirnov (KS) 2-sample t-test for 
each feature that is identified as a "drift candidate". It can be used for testing or on-going monitoring, where 
typically the first data set is the training or baseline data set, and the second data set is the hold-out or 
production data set. 

The output of this monitor is:
- Table of Features that have failed the Drift test, according to the threshold set
- Line graph of the Data drift per Day per feature, using the predictionDateColumn


## Input Assets

| Type            | Number | Description                                                              |
|-----------------| ------ |--------------------------------------------------------------------------|
| Baseline Data   | **1**  | The Baseline or Training data set                                        |
| Comparator Data | **1**  | The Hold-out or Production data set to compare against the Baseline_Data |

## Job Parameters
- (Optional) predictionDateColumn: this is the name of the column in the data set that identifies the specific prediction date. It can be used for calculating the drift per feature PER DAY. If this value is not provided, then the drift monitor will calculate drift per feature across the entire data set
- (Optional) ksThreshold: the specific KS p-value above which the monitor will detect a potential drift has occurred. The monitor will default to 0.05 if this job parameter is not provided.

## Assumptions & Requirements
- Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
- Input data contains at least one `numerical` column or one `categorical` column.


## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Data Drift Monitor** class and uses the job json asset to determine the `numerical_columns` and/or `categorical_columns` accordingly.
3. The **Kolmogorov-Smirnov** data drift test is run.
4. Test result is returned under the list of `data_drift_ks` tests.

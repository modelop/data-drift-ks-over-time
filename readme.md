# Example Custom Metrics Monitor
This metrics model calculates data drift between two data sets using the Kolmogorov-Smirnov (KS) 2-sample t-test for 
each feature that is identified as a "drift candidate". It can be used for testing or on-going monitoring, where 
typically the first data set is the training or baseline data set, and the second data set is the hold-out or 
production data set. The output of this metrics model is the KS p-value for each feature


## Input Assets

| Type            | Number | Description                                                              |
|-----------------| ------ |--------------------------------------------------------------------------|
| Baseline Data   | **1**  | The Baseline or Training data set                                        |
| Comparator Data | **1**  | The Hold-out or Production data set to compare against the Baseline_Data |

## Assumptions & Requirements
- Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
- Input data contains at least one `numerical` column or one `categorical` column.

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Data Drift Monitor** class and uses the job json asset to determine the `numerical_columns` and/or `categorical_columns` accordingly.
3. The **Kolmogorov-Smirnov** data drift test is run.
4. Test result is returned under the list of `data_drift_ks` tests.

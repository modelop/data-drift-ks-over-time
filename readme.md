# Example Custom Metrics Monitor
This python code provides a very basic example of a custom monitor, including how to use the init and metrics function, as well as how to test locally with a main() function. This initial example shows how you would incorporate externally generated metrics into the ModelOp system.
It will read any input asset and echo the first row of information returned from that asset.  So if you, for example,
stored your metrics in an s3 file generated outside of modelop, you can store it as a simple json record.  This monitor
will then simply read from that asset and echo the json back out.  That will then be transformed into a model test
result by the modelop system.

The reason for doing it this way is to allow you to have metrics in SQL, S3 or other secure locations, and allow a
runtime to be configured with the appropriate credentials.  Then instead of an external entity trying to read those
directly, or hard coding credentials directly into python, the engine can be configured through secret stores with that
appropriate access, and only the runtime will read those values and echo them into the ModelOp system for transformation
into a test result.


## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **1**  | Externally generated metrics in json format           |

## Assumptions & Requirements
 - This custom monitor assumes the input is a json object of metrics to be tracked and scored later.

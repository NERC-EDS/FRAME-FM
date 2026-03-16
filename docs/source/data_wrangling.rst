Data Wrangling
==============

One of the large focuses of FRAME-FM is dealing with all of the data wrangling so that users can focus on configuring and running Machine Learning.

Utils Classes and Functions
---------------------------

There are many base classes and functions within "utils" to allow loading a variety of input data in a variety of common formats (gridded timeseries, GeoTIFF, etc).
This ensures that scientists don't need to write their own data loaders.

Stages in Loading Data
----------------------

There are two steps involved in loading data:

* Retrieving metadata and other details on the input data.
* Applying transformations to retrieve slides of the data array and convert them into PyTorch tensors.

FRAME\_FM.datasets package
==========================

A location for the FRAME-FM datasets, which are all sub-classes of
``torch.utils.data.Dataset``. They all use the standard model of exposing a
*dataset* as a generator that can be directly used by a ``DataLoader`` object.
The common interface is:

- construction: ``__init__()``
- length: ``__len__()``
- get item by index: ``__getitem__(idx)``

Class Hierarchy
---------------

The class hierarchy is built as follows::

    torch.utils.data.Dataset
        - BaseDataset
            - BaseGriddedDataset
            - BaseGeoTIFFDataset
                - BaseASCIIGridDataset
            - BaseGriddedTimeSeriesDataset
                - SoilWaterIndexGriddedTimeSeriesDataset
            - BaseShapefileDataset
                - TopsoilDataset
            - CosmosUKDataset
        - BaseShapefileDataset
        - CorrespondingTilesDataset
        - TransformedDataset
        - TransformedInputDataset
        - TransformedInputCoordsDataset
        - TransformedInputTimeCoordsDataset
        - ZipDataset

Changing datasets using ``preprocessors`` and ``transforms``
-------------------------------------------------------------

Each ``Dataset`` class can have two types of *operations* defined by arguments
sent to the constructor:

``preprocessors``
    A list of operations that get run when the Dataset instance is created.

    - These get run once only.
    - They operate sequentially, with the first taking in an ``xr.Dataset``.
    - The final object is saved in ``self.data``.
    - The resulting output should be ready for use by the standard methods:

      .. code-block:: python

          def __len__(self):
          def __getitem__(self, idx):

``transforms``
    A list of operations that get run at training time, within the
    ``__getitem__(idx)`` call.

    - These are run whenever a ``DataLoader`` needs to access single items or
      batches of items with a ``Dataset`` object.
    - These are typically run like this:

      .. code-block:: python

          for transform in self.transforms:
              sample = transform(sample)

.. note::

    The ``FRAME_FM.transforms.transforms`` module contains all the transform
    classes that can be used in either or both of the ``preprocessors`` and
    ``transforms`` lists.

See the transform examples in the unit tests: ``tests/transforms/test_transforms.py``

See the ``Dataset`` unit tests for examples: ``tests/datasets/test_*.py``

Submodules
----------

FRAME\_FM.datasets.ImageLabel\_Dataset module
---------------------------------------------

.. automodule:: FRAME_FM.datasets.ImageLabel_Dataset
   :members:
   :undoc-members:
   :show-inheritance:

FRAME\_FM.datasets.InputOnly\_Dataset module
--------------------------------------------

.. automodule:: FRAME_FM.datasets.InputOnly_Dataset
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: FRAME_FM.datasets
   :members:
   :undoc-members:
   :show-inheritance:

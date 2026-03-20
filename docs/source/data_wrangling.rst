Data Wrangling
==============

One of the main focuses of FRAME-FM is to simplify the pre-processing of large
spatio-temporal datasets so that users can focus on configuring and running Machine Learning 
workflows.

Standard approach for setting up data workflows
-----------------------------------------------

The recommended approach for setting up data pre-processing and loading within the framework 
is to write _recipes_ that consist of a sequence of operations to be performed on the data.

All _datasets_ are sub-classes of ``torch.utils.data.Dataset``. They all use the standard 
``torch`` model of exposing a _dataset_ that can be directly used by a ``DataLoader`` object. 
The common interface is:

- contruction: ``__init__()``
- length: ``__len__()``
- get item by index: ``__getitem__(idx)``

Modifying datasets using ``preprocessors`` and ``transforms``
-------------------------------------------------------------

Each ``Dataset`` class can have two types of _operations_ defined by arguments 
sent to the constructor:

- ``preprocessors``:
    - A list of operations that get run when the Dataset instance is created.
    - These get run once only.
    - They operate sequentially, with the first taking in an ``xr.Dataset``
    - The final object is saved in ``self.data``
    - The resulting output should be ready for use by the standard methods:
        - ``def __len__(self):``
        - ``def __getitem__(self, idx):``

- ``transforms``:
    - A list of operations that get run at training time, within the ``__getitem__(idx)``
      call.
    - These are run whenever a ``DataLoader`` needs to access single items or batches
      of items with a ``Dataset`` object.
    - These are typically run like this:
      ```python
      for transform in transforms:
          sample = transform(sample)
      ```

Note that the ``FRAME_FM.transforms.transforms`` module contains all the transform 
classes that can be in either/both of the ``preprocessors`` and ``transforms`` lists.

See the examples in the unit tests: ``tests/transforms/test_transforms.py``

See the `Dataset` unit tests for examples: ``tests/datasets/test_*.py``


Transforms
==========

The "tranforms" directory is a location for all transformation classes and relevant utilities used to
integrate them into the FRAME-FM framework.

Overview of Transforms
----------------------

PyTorch ``Dataset`` classes typically employ a sequence of *transform* objects
that modify input data into a form suitable for model training or inference.

This usually happens in the ``__getitem__()`` method, but it may also be
appropriate to use transforms elsewhere, such as in the ``__init__()`` method
when initial modifications are required.

Within the ``FRAME_FM`` package, transforms are all children of the
``FRAME_FM.transforms.BaseTransform`` class. They are typically run as follows:

.. code-block:: python

   from FRAME_FM.transforms import NormalizeTransform

   ds = xr.load_dataset(<some_dataset>)
   da = ds["d2m"].isel(time=0)

   normalize_transform = NormalizeTransform()
   normalized_da = normalize_transform(
       da,
       mean=float(da.mean()),
       std=float(da.std())
   )

Using pre-existing Transforms
-----------------------------

Other types of PyTorch transforms, such as those in the ``torchvision`` library, 
can be included in a sequence of transforms:

* ``torchvision.transforms``  
  https://docs.pytorch.org/vision/main/transforms.html

Defining your own Transforms
----------------------------

You can create your own transforms, by ensuring they inherit the ``BaseTransform`` 
class and follow the following structure::

   import xarray as xr
   DS = xr.Dataset
   from FRAME_FM.transforms import BaseTransform

   class Add1Transform(BaseTransform):
       """
       Adds 1 to the data array.
       """
       def __init__(self, var_id: str):
           # You can define your own input parameters
           self.var_id = var_id
   
       def __call__(self, sample: DS) -> DS:
           # This must receive a data object, such as an xarray Dataset/DataArray
           check_object_type(sample, allowed_types=DS, caller=self.__class__.__name__)

           sample = sample[self.var_id] += 1
           return sample

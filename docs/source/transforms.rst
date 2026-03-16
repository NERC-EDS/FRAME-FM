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

Current Issue (Feb 2026) with ``ds.roll`` on ``xarray.Dataset``
---------------------------------------------------------------

The ``ds.roll()`` operation on an ``xarray.Dataset`` object can cause the system
to hang on some installations.

So far we have diagnosed that:

* Works correctly on ``xarray==2025.11.0``
* Fails (hangs) on ``xarray==2026.2.0``

Supporting Pre-existing Transforms
----------------------------------

Additionally we may support transforms from other tools/systems, e.g.:

* ``anemoi-transform``  
  https://anemoi.readthedocs.io/projects/transform/en/latest/

* ``torchvision.transforms``  
  https://docs.pytorch.org/vision/main/transforms.html

At present these are not supported.

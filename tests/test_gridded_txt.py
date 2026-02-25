import pytest

@pytest.fixture(scope="session")
def dataset_matcher():
    from xarray_dataloaders import xarray_end_to_end

    xarray_end_to_end()


    ...
    
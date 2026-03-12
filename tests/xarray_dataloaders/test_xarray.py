import pytest

# https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py

@pytest.fixture(scope="session")
def import_fixture():
    ...


def test_dataset():
    from FRAME_FM.dataloaders import gridded_txt_dataloader

    assert True==True

# This is a stupid test. But my brain isn't working right now.
def test_end_to_end(import_fixture):
    # This test will run after the xarray_dataloader_test fixture has completed
    from xarray_dataloaders import xarray_end_to_end
    try:
        xarray_end_to_end.main() # Call the main function to run the end-to-end test
    except:
        assert True==False  # Replace with actual assertions relevant to your tests
    
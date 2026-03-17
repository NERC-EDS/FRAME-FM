class CRSUnresolvableError(Exception):
    """Raised when the CRS of an Xarray object cannot be determined."""
    pass


class DimNotFoundError(Exception):
    """Raised when a required dimension is not found in the Xarray object."""
    pass


class ExpectedDimsMismatchError(Exception):
    """Raised when the object's tiled dimensions don't match the `expected` argument."""
    pass

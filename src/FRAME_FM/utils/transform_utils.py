

def check_object_type(obj: object, allowed_types: object | tuple[object, ...], caller: str) -> object:
    """
    Check if the object is an instance of the allowed types, and raise a TypeError if not.

    Args:
        - obj (object): The object to check.
        - allowed_types (object or tuple of objects): The type or types that the object is allowed to be.
        - caller (str): The name of the calling function or transform, used for error messages.
    Returns:
        - object: The original object if it is of an allowed type.
    Raises:
        - TypeError: If the object is not an instance of any of the allowed types."""
    # Check if allowed_types is a single type, if so convert it to a tuple
    if isinstance(allowed_types, type):
        allowed_types = (allowed_types,)

    for t in allowed_types:   # type: ignore
        if isinstance(obj, t):
            return obj

    raise TypeError(f"Expected an object of type: {allowed_types} when calling `{caller}`, but received: {type(obj)}.")
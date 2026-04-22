import inspect
import json
import os
from typing import Any

GITHUB_BASE_URL = "https://github.com/NERC-EDS/FRAME-FM/tree/main/src/"

# ---------------------------------------------------------------------------
# Class resolution
# ---------------------------------------------------------------------------

def _resolve_class(op_type: str):
    """
    Look up the implementing class for an operation type via transform_mapping.
    Returns the class, or None if the import or lookup fails.
    """
    try:
        from FRAME_FM.transforms import transform_mapping as tm
        return tm.get(op_type)
    except ImportError:
        return None


def _get_class_source_info(cls) -> tuple[str | None, int | None]:
    """
    Return (relative_source_file, line_number) for a class using inspect.
    The source file is made relative by stripping everything up to and
    including 'src/' so it can be appended to the base GitHub URL.

    Returns (None, None) if inspection fails.
    """
    try:
        source_file = inspect.getfile(cls)
        source_lines, start_line = inspect.getsourcelines(cls)
        # Normalise path: keep only the part from 'src/' onwards
        # e.g. /home/user/project/src/FRAME_FM/transforms.py
        #   -> FRAME_FM/transforms.py
        parts = source_file.replace("\\", "/").split("/src/")
        relative_path = parts[-1] if len(parts) > 1 else os.path.basename(source_file)
        return relative_path, start_line
    except (TypeError, OSError):
        return None, None


def _build_github_url(base_url: str, relative_path: str, line_number: int | None) -> str:
    """
    Assemble a GitHub URL pointing at the file, optionally anchored to a line.

    base_url  : e.g. "https://github.com/NERC-EDS/FRAME-FM/tree/main/src/"
    relative_path : e.g. "FRAME_FM/transforms.py"
    line_number   : e.g. 42
    """
    base_url = base_url.rstrip("/")
    url = f"{base_url}/{relative_path}"
    if line_number is not None:
        url += f"#L{line_number}"
    return url


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def _recipe_to_property_value(
    step_number: int,
    recipe: dict,
    github_base_url: str | None = None,
) -> dict:
    """
    Convert a single recipe dict into a schema.org PropertyValue object.

    The `type` key becomes the step name; all other keys become nested
    additionalProperty entries, preserving their values as JSON-serialisable
    strings (or native types where possible).

    If `github_base_url` is provided, the implementing class name and a
    direct GitHub URL (with line number anchor) are added as extra properties.
    """
    op_type = recipe.get("type", "unknown")

    # Parameters are everything except `type`
    params = {k: v for k, v in recipe.items() if k != "type"}

    param_properties = []
    for param_name, param_value in params.items():
        param_properties.append({
            "@type": "PropertyValue",
            "name": param_name,
            "value": _serialise_value(param_value),
        })

    # --- Class introspection ---
    cls = _resolve_class(op_type)
    if cls is not None:
        class_name = cls.__name__
        relative_path, line_number = _get_class_source_info(cls)

        param_properties.append({
            "@type": "PropertyValue",
            "name": "implementingClass",
            "value": class_name,
        })

        if github_base_url and relative_path:
            github_url = _build_github_url(github_base_url, relative_path, line_number)
            param_properties.append({
                "@type": "PropertyValue",
                "name": "sourceCodeURL",
                "value": github_url,
            })

    pv = {
        "@type": "PropertyValue",
        "name": f"step_{step_number}_{op_type}",
        "propertyID": "pipelineStep",
        "value": str(step_number),
        "description": f"Operation type: {op_type}",
    }

    if param_properties:
        pv["additionalProperty"] = param_properties

    return pv


def _serialise_value(value: Any) -> Any:
    """
    Convert a Python value to something JSON-safe and human-readable.
    - None        → "null"
    - tuple/list  → kept as list (JSON-serialisable)
    - everything else → returned as-is if primitive, else str()
    """
    if value is None:
        return "null"
    if isinstance(value, tuple):
        return list(value)          # JSON has no tuple type; list is fine
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_serialise_value(v) for v in value]
    return str(value)               # fallback for anything exotic


def _build_pipeline_block(
    label: str,
    description: str,
    recipes: list[dict],
    github_base_url: str | None = None,
) -> dict:
    """
    Build one top-level PropertyValue block for either preprocessors.
    """
    steps = [
        _recipe_to_property_value(i + 1, recipe, github_base_url)
        for i, recipe in enumerate(recipes)
    ]

    return {
        "@type": "PropertyValue",
        "name": label,
        "description": description,
        "value": steps,
    }


def build_additional_property(
    preprocessors: list[dict],
    github_base_url: str | None = GITHUB_BASE_URL,
) -> list[dict]:
    """
    Build the full `additionalProperty` list containing both pipeline blocks.

    Parameters
    ----------
    preprocessors : list[dict]
        Recipes run once at Dataset instantiation time.
    github_base_url : str or None
        Base GitHub URL for the source tree. When provided (and the
        FRAME_FM package is importable), each step will include an
        `implementingClass` and a `sourceCodeURL` pointing directly
        to the class definition with a line-number anchor.
        Set to None to disable class introspection entirely.

    Returns
    -------
    list[dict]
        A list suitable for use as the `additionalProperty` field in a
        Croissant JSON-LD record.
    """
    blocks = []

    if preprocessors:
        blocks.append(_build_pipeline_block(
            label="preprocessingPipeline",
            description=(
                "Ordered sequence of preprocessing operations applied once "
                "at Dataset instantiation time."
            ),
            recipes=preprocessors,
            github_base_url=github_base_url,
        ))

    return blocks


def save_additional_property(
    additional_property: list[dict],
    output_path: str,
) -> None:
    """
    Write the `additionalProperty` block to a JSON file.
    """
    output = {"additionalProperty": additional_property}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Written to {output_path}")


# # ---------------------------------------------------------------------------
# # Example / demo
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Representative sample of the recipe dicts from the spec
#     var_id = "t2m"
#     stride = 6

#     example_preprocessors = [
#         {"type": "rename", "var_id": var_id, "new_name": "dewpoint_temperature"},
#         {"type": "resample", "dim": "x", "freq": stride, "method": "mean"},
#         {"type": "reverse_axis", "dim": "latitude"},
#         {"type": "roll", "dim": "longitude", "shift": None},
#         {"type": "subset", "time": ("2000-01-01", "2000-01-10"),
#          "latitude": (60, -30), "longitude": (40, 100)},
#     ]

#     additional_property = build_additional_property(
#         example_preprocessors,
#         github_base_url=GITHUB_BASE_URL,
#     )

#     output_path = "croissant_additional_property.json"
#     save_additional_property(additional_property, output_path)

#     # Also pretty-print to stdout so you can inspect it immediately
#     print(json.dumps({"additionalProperty": additional_property}, indent=2))
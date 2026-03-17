"""
Dimension resolution and object-type detection.

Responsibilities
----------------
* Map single-letter dim codes ("t", "z", "y", "x") to the actual coordinate
  names found in an Xarray object, accounting for the ``_coarse`` / ``_fine``
  suffix convention.
* Validate that the object's tiled dims match an ``expected`` string.
* Classify an Xarray object as one of: "pixel", "tile", "tiled_array".
"""

from __future__ import annotations

import xarray as xr

from .constants import COARSE_SUFFIX, DIM_ALIASES, FINE_SUFFIX
from .exceptions import DimNotFoundError, ExpectedDimsMismatchError


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

OBJECT_TYPES = ("pixel", "tile", "tiled_array")

# Maps a dim letter to the resolved base name (e.g. "y" → "lat")
# and the coarse/fine coordinate names actually present.
class ResolvedDim:
    """Container for one resolved tiling dimension."""

    def __init__(
        self,
        letter: str,
        base_name: str,
        coarse_name: str,
        fine_name: str,
    ) -> None:
        self.letter = letter          # e.g. "y"
        self.base_name = base_name    # e.g. "lat"
        self.coarse_name = coarse_name  # e.g. "lat_coarse"
        self.fine_name = fine_name      # e.g. "lat_fine"

    def __repr__(self) -> str:
        return (
            f"ResolvedDim(letter={self.letter!r}, base={self.base_name!r}, "
            f"coarse={self.coarse_name!r}, fine={self.fine_name!r})"
        )


# ---------------------------------------------------------------------------
# Dim string parsing
# ---------------------------------------------------------------------------

def parse_dim_string(dims: str | list[str] | None) -> list[str]:
    """
    Normalise *dims* to a lowercase list of single-letter codes.

    Accepts:
      * ``"tyx"``         → ``["t", "y", "x"]``
      * ``["t", "y", "x"]`` → ``["t", "y", "x"]``
      * ``None``          → ``[]``
    """
    if dims is None:
        return []
    if isinstance(dims, str):
        return [c for c in dims.lower() if not c.isspace()]
    return [d.lower().strip() for d in dims]


# ---------------------------------------------------------------------------
# Dim resolution
# ---------------------------------------------------------------------------

def resolve_dims(
    obj: xr.DataArray | xr.Dataset,
    dim_letters: list[str],
    *,
    strict: bool = True,
) -> dict[str, ResolvedDim]:
    """
    For each letter in *dim_letters*, find the matching ``_coarse`` and
    ``_fine`` coordinates in *obj*.

    Parameters
    ----------
    obj:
        The Xarray object to inspect.
    dim_letters:
        Ordered list of single-letter dim codes, e.g. ``["t", "y", "x"]``.
    strict:
        If *True* (default), raise :class:`DimNotFoundError` when a letter
        cannot be resolved.  Set to *False* to silently skip unresolved dims.

    Returns
    -------
    dict[str, ResolvedDim]
        Keyed by letter code.
    """
    all_names: set[str] = set(obj.coords) | set(obj.dims)
    resolved: dict[str, ResolvedDim] = {}

    for letter in dim_letters:
        aliases = DIM_ALIASES.get(letter)
        if aliases is None:
            raise DimNotFoundError(
                f"Unknown dimension letter {letter!r}. "
                f"Supported letters: {sorted(DIM_ALIASES.keys())}"
            )

        matched: ResolvedDim | None = None
        for alias in aliases:
            coarse = alias + COARSE_SUFFIX
            fine = alias + FINE_SUFFIX
            if coarse in all_names and fine in all_names:
                matched = ResolvedDim(
                    letter=letter,
                    base_name=alias,
                    coarse_name=coarse,
                    fine_name=fine,
                )
                break

        if matched is None:
            if strict:
                candidates = [
                    f"{a}{COARSE_SUFFIX} / {a}{FINE_SUFFIX}"
                    for a in aliases
                ]
                raise DimNotFoundError(
                    f"Could not find tiled coordinates for dimension {letter!r}.\n"
                    f"Looked for (any of): {candidates}\n"
                    f"Available names: {sorted(all_names)}"
                )
        else:
            resolved[letter] = matched

    return resolved


# ---------------------------------------------------------------------------
# Expected-dims validation
# ---------------------------------------------------------------------------

def validate_expected(
    obj: xr.DataArray | xr.Dataset,
    expected: str | list[str] | None,
) -> dict[str, ResolvedDim]:
    """
    Validate that *obj* has tiled (``_coarse`` / ``_fine``) coordinates for
    every letter in *expected*.

    Returns the resolved dim map so callers don't have to resolve twice.

    Unlike :func:`resolve_dims`, this function returns an empty dict rather
    than raising when no coarse/fine coordinates are found at all — that
    signals a pixel object to the factory, which handles it separately.

    Raises :class:`ExpectedDimsMismatchError` only when *some* but not all
    of the expected dims have coarse/fine coords (a genuine mismatch), or
    when an unknown dim letter is given.
    """
    letters = parse_dim_string(expected)
    if not letters:
        return {}

    resolved = resolve_dims(obj, letters, strict=False)

    # If nothing resolved at all AND the caller explicitly provided `expected`,
    # this is a genuine mismatch (e.g. raw non-tiled data passed with expected="tzyx").
    # Return empty only when no expected was given — that signals a pixel to the factory.
    missing = [l for l in letters if l not in resolved]
    if missing:
        # Nothing resolved at all — no coarse/fine structure found for any dim.
        if not resolved:
            raise ExpectedDimsMismatchError(
                f"Object does not match `expected` dims {expected!r}. "
                f"No coarse/fine tiling coordinates were found for any of: {letters}. "
                f"Available coordinates: {sorted(set(obj.coords) | set(obj.dims))}."
            )
        # Some resolved, some not → partial mismatch.
        raise ExpectedDimsMismatchError(
            f"Object does not match `expected` dims {expected!r}. "
            f"Could not find coarse/fine coordinates for: {missing}. "
            f"Found coarse/fine for: {list(resolved.keys())}."
        )

    return resolved


def has_tiling_structure(obj: xr.DataArray | xr.Dataset) -> bool:
    """
    Return True if *obj* has any ``_coarse`` / ``_fine`` coordinate pairs,
    indicating it was produced by the tiling pipeline.

    Used by the factory to short-circuit ``validate_expected`` for plain
    pixel objects that have no tiling structure at all.
    """
    all_names = set(obj.coords) | set(obj.dims)
    return any(
        (name.endswith(COARSE_SUFFIX) and name[:-len(COARSE_SUFFIX)] + FINE_SUFFIX in all_names)
        for name in all_names
    )


# ---------------------------------------------------------------------------
# Object-type detection
# ---------------------------------------------------------------------------

def detect_object_type(
    obj: xr.DataArray | xr.Dataset,
    resolved: dict[str, ResolvedDim],
) -> str:
    """
    Classify *obj* as ``"pixel"``, ``"tile"``, or ``"tiled_array"``.

    Classification rules
    --------------------
    * **tiled_array** – the object has a stacked MultiIndex dimension whose
      levels include at least one ``_coarse`` coordinate, *or* any
      ``_coarse`` coordinate has size > 1 along a non-fine axis.
    * **tile** – ``_coarse`` coordinates exist but each has only a single
      value (i.e. this is one tile).
    * **pixel** – no ``_coarse`` / ``_fine`` coordinates at all, or all
      spatial dims are scalar.
    """
    if not resolved:
        return "pixel"

    # Check for MultiIndex (stacked tiled array)
    for dim in obj.dims:
        idx = obj.indexes.get(dim)
        if idx is not None and hasattr(idx, "levels"):
            # pandas MultiIndex — check if any level matches a coarse name
            level_names = set(idx.names)
            coarse_names = {rd.coarse_name for rd in resolved.values()}
            if level_names & coarse_names:
                return "tiled_array"

    # Fall back to size inspection on coarse coordinates
    for rd in resolved.values():
        if rd.coarse_name in obj.coords:
            coarse_coord = obj.coords[rd.coarse_name]
            if coarse_coord.size > 1:
                return "tiled_array"

    # All coarse coords have size == 1 → single tile
    coarse_names = {rd.coarse_name for rd in resolved.values()}
    if coarse_names & (set(obj.coords) | set(obj.dims)):
        return "tile"

    return "pixel"

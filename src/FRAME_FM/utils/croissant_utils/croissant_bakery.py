#!/usr/bin/env python
# coding: utf-8

import cf_xarray # noqa: F401
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import hashlib
import json

import xarray as xr

from FRAME_FM.utils.croissant_utils.additional_properties import build_additional_property


def get_temporal_resolution(ds: xr.Dataset) -> Tuple[float, str]:
    """
    Get the temporal resolution of a dataset.

    Parameters:
    ds (xr.Dataset): The input xarray Dataset.

    Returns:
    tuple: A tuple containing the temporal resolution value and unit.
    """
    # Check if 'time' coordinate exists
    if 'time' not in ds.coords:
        raise ValueError("Dataset does not contain a 'time' coordinate.")

    # Extract time values and convert to pandas datetime
    time_values = pd.to_datetime(ds['time'].values)

    # Calculate the differences between consecutive time points
    time_diffs = time_values[1:] - time_values[:-1]

    # Get the most common time difference
    mcd: pd.Timedelta = time_diffs.value_counts().idxmax()   # type: ignore

    # Convert the most common difference to a human-readable format
    if mcd < pd.Timedelta('1 hour'):
        return mcd.total_seconds(), 'second'
    elif mcd < pd.Timedelta('1 day'):
        return mcd.total_seconds() / 3600, 'hour'
    elif mcd < pd.Timedelta('31 day'):
        return mcd.total_seconds() / 86400, 'day'
    elif mcd < pd.Timedelta('366 day'):
        return mcd.total_seconds() / (30 * 86400), 'month'
    else:
        return mcd.total_seconds() / (365 * 86400), 'year'

def _get_bbox_string(ds: xr.Dataset) -> str:
    """Get bounding box string in the format 'lat_min lon_min lat_max lon_max'"""
    lat, lon = ds.cf["latitude"].values, ds.cf["longitude"].values
    return f"{lat.min()} {lon.min()} {lat.max()} {lon.max()}"


def _get_crs(metadata: Dict[str, Any], ds: xr.Dataset | None) -> str:
    """Get coordinate reference system (CRS) from metadata or dataset attributes"""
    print("TODO: Switch to resolve CRS using FRAME_FM.utils.geometry_utils.resolve_crs() once implemented")
    if "crs" in metadata:
        return metadata["crs"]

    # Try to resolve from Dataset
    import rioxarray  # noqa: F401
    rio_crs = ds.rio.crs
    
    return str(rio_crs) if rio_crs else "Undefined"


class ZarrToCroissantConverter:
    """Dynamic converter for NASA POWER data to GeoCroissant format"""

    def __init__(
        self,
        zarr_url: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ):
        """
        Initialize the converter with the Zarr URL

        Args:
            zarr_url: URL or path to the Zarr dataset (e.g., S3 URL or local path)
            metadata: Optional dictionary of additional metadata to include in the GeoCroissant output
            verbose: If True, print detailed information during processing
        """
        self.zarr_url = str(zarr_url)
        self.metadata = metadata or {}
        self.verbose = verbose

        # Placeholder for the loaded dataset
        self.ds: Optional[xr.Dataset] = None

    def load_dataset(self) -> bool:
        """Load the dataset. Returns True if successful, False otherwise."""
        try:
            if self.verbose:
                print(f"Attempting to load dataset from: {self.zarr_url}")
            self.ds = xr.open_zarr(self.zarr_url)
            if self.verbose:
                print("Dataset loaded successfully!")
                print(f"  - Dimensions: {self.ds.dims}")
                print(f"  - Total size: {self.ds.nbytes / 1e9:.2f} GB")
                print(f"  - Variables: {len(self.ds.data_vars)}")
                print(
                    f"  - Time range: {self.ds.time.values[0]} to"
                    f" {self.ds.time.values[-1]}"
                )
            return True
        except Exception as e:
            raise Exception(f"Error loading dataset due to issue: {e}")

    def get_available_variables(self) -> Dict[str, Any]:
        """Get list of available variables with their metadata"""
        if not self.ds:
            return {}

        variables = {}
        for var_name, var in self.ds.data_vars.items():
            variables[var_name] = {
                "shape": list(var.shape),
                "dimensions": list(var.dims),
                "dtype": str(var.dtype),
                "size_mb": float(var.nbytes / 1e6),
                "attributes": (
                    dict(var.attrs) if hasattr(var, "attrs") and var.attrs else {}
                ),
            }
        return variables

    def generate_checksum(self, content: str) -> str:
        """Generate MD5 checksum for content"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def create_croissant_metadata(self, output_file: str = "record.json") -> Dict[str, Any]:
        """
        Create GeoCroissant metadata for the data

        Args:
            output_file: Output file path (if None, auto-generated)

        Returns:
            dict: GeoCroissant metadata
        """
        if not self.ds:
            print("Error: No dataset available. Call load_dataset() first.")
            return {}

        # Generate checksum
        hash_input = f"{repr(self.ds.dims)}|{repr(self.ds.data_vars)}|{repr(self.ds.attrs)}"
        md5_hash = self.generate_checksum(hash_input)

        # Create time extent
        if "time" not in self.ds.coords:
            print("Error: 'time' coordinate not found in dataset. Setting dummy time extent.")
            start, end = "01-01-01T00:00:00Z", "9999-12-31T00:00:00Z"
        else:
            time_values = self.ds.time.values
            start, end = [pd.to_datetime(tm).strftime("%Y-%m-%dT%H:%M:%SZ") for tm in [time_values[0], time_values[-1]]]

        # Use mid-points to calculate spatial resolution, since the edges may not be evenly spaced
        lat, lon = self.ds.cf["latitude"].values, self.ds.cf["longitude"].values
        lat_mid_index = int(len(lat) / 2)
        lon_mid_index = int(len(lon) / 2)
        lat_diff = float(lat[lat_mid_index] - lat[lat_mid_index - 1]) if len(lat) > 1 else "undefined"
        lon_diff = float(lon[lon_mid_index] - lon[lon_mid_index - 1]) if len(lon) > 1 else "undefined"

        # Create GeoCroissant metadata
        croissant = {
            "@context": {
                "@language": "en",
                "@vocab": "https://schema.org/",
                "citeAs": "cr:citeAs",
                "column": "cr:column",
                "conformsTo": "dct:conformsTo",
                "cr": "http://mlcommons.org/croissant/",
                "geocr": "http://mlcommons.org/croissant/geocr/",
                "rai": "http://mlcommons.org/croissant/RAI/",
                "dct": "http://purl.org/dc/terms/",
                "sc": "https://schema.org/",
                "data": {"@id": "cr:data", "@type": "@json"},
                "examples": {"@id": "cr:examples", "@type": "@json"},
                "dataBiases": "cr:dataBiases",
                "dataCollection": "cr:dataCollection",
                "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
                "extract": "cr:extract",
                "field": "cr:field",
                "fileProperty": "cr:fileProperty",
                "fileObject": "cr:fileObject",
                "fileSet": "cr:fileSet",
                "format": "cr:format",
                "includes": "cr:includes",
                "isLiveDataset": "cr:isLiveDataset",
                "jsonPath": "cr:jsonPath",
                "key": "cr:key",
                "md5": "cr:md5",
                "parentField": "cr:parentField",
                "path": "cr:path",
                "personalSensitiveInformation": "cr:personalSensitiveInformation",
                "recordSet": "cr:recordSet",
                "references": "cr:references",
                "regex": "cr:regex",
                "repeated": "cr:repeated",
                "replace": "cr:replace",
                "samplingRate": "cr:samplingRate",
                "separator": "cr:separator",
                "source": "cr:source",
                "subField": "cr:subField",
                "transform": "cr:transform",
            },
            "@type": "sc:Dataset",
            "name": self.metadata.get("name", "Unknown dataset"),
            "alternateName": self.metadata.get("alternateNames", []),
            "description": self.metadata.get("description", "No description provided."),
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "version": "1.0.0",
            "creator": {
                "@type": "Organization",
                "name": self.metadata.get("creator", "Unknown creator"),
                "url": self.metadata.get("creatorUrl", None),
            },
            "url": self.metadata.get("url", None),
            "keywords": self.metadata.get("keywords", []),
            "citeAs": self.metadata.get("citeAs", None),
            "datePublished": self.metadata.get("datePublished", datetime.now().isoformat().split(".")[0]),
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "spatialCoverage": {
                "@type": "Place",
                "geo": {
                    "@type": "GeoShape",
                    "box": _get_bbox_string(self.ds)
                }
            },
            "geocr:temporalExtent": {"startDate": start, "endDate": end},
            "geocr:temporalResolution": get_temporal_resolution(self.ds),
            "geocr:spatialResolution": f"{lat_diff}° lat x {lon_diff}° lon",
            "geocr:coordinateReferenceSystem": _get_crs(self.metadata, self.ds),
            "geocr:mlTask": self.metadata.get("mlTask", None),
            "distribution": [
                {
                    "@type": "cr:FileObject",
                    "@id": self.zarr_url,
                    "name": self.zarr_url.split("/")[-1],
                    "description": f"Zarr dataset at: {self.zarr_url}",
                    "contentUrl": self.zarr_url,
                    "encodingFormat": "application/x-zarr",
                    "md5": md5_hash,
                }
            ],
            "recordSet": [
                {
                    "@type": "cr:RecordSet",
                    "@id": "variables_and_coordinates",
                    "name": "Variable and Coordinate Fields",
                    "description": "Fields for variables and coordinates in the dataset",
                    "field": [],
                }
            ],
        }

        # Include additional properties if exists
        if self.metadata.get("additionalProperty"):
            croissant["additionalProperty"] = self.metadata["additionalProperty"]

        # Add fields for each variable
        fields = croissant["recordSet"][0]["field"]

        # Add coordinate fields
        for coord_name, coord in self.ds.coords.items():
            mn, mx = [float(i) for i in [coord.values.min(), coord.values.max()]]
            coord_field = {
                "@type": "cr:Field",
                "@id": coord_name,
                "name": coord_name,
                "description": f"Coordinate: {coord_name}",
                "dataType": "sc:Float" if coord.dtype.kind == "f" else "sc:Date",
                "source": {
                    "fileObject": {
                        "@id": self.zarr_url
                    },
                    "extract": {"jsonPath": f"$.{coord_name}"},
                },
                "geocr:dataShape": list(coord.shape),
                "geocr:validRange": (
                    {
                        "min": mn,
                        "max": mx,
                    }
                ),
                "geocr:units": coord.attrs.get("units", "")
            }
            # Remove None values
            coord_field = {k: v for k, v in coord_field.items() if v is not None}
            fields.append(coord_field)

        # Add data variable fields
        for var_name, var in self.ds.data_vars.items():
            var_field = {
                "@type": "cr:Field",
                "@id": var_name,
                "name": var_name,
                "description": var.attrs.get("long_name", var_name),
                "dataType": "sc:Float",
                "source": {
                    "fileObject": {
                        "@id": self.zarr_url
                    },
                    "extract": {"jsonPath": f"$.{var_name}"},
                },
                "geocr:dataShape": list(var.shape),
                "geocr:validRange": (
                    {
                        "min": float(var.attrs.get("valid_min", "UNDEFINED")),
                        "max": float(var.attrs.get("valid_max", "UNDEFINED")),
                    }
                    if var.attrs.get("valid_min") is not None
                    and var.attrs.get("valid_max") is not None
                    else None
                ),
                "geocr:units": var.attrs.get("units", ""),
                "geocr:standardName": var.attrs.get("standard_name", ""),
                "geocr:definition": var.attrs.get("definition", ""),
                "geocr:cellMethods": var.attrs.get("cell_methods", ""),
                "geocr:cellMeasures": var.attrs.get("cell_measures", ""),
                "geocr:chunkSizes": {d: next(iter(v)) for d, v in var.chunksizes.items()}  
            }
            # Remove None values
            var_field = {k: v for k, v in var_field.items() if v is not None}
            fields.append(var_field)

        # Save metadata
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(croissant, f, indent=4, ensure_ascii=False)

        print(f"GeoCroissant metadata saved to {output_file}")
        print(f"Total fields: {len(fields)}")

        return croissant

    def convert(
        self,
        output_file: str = "record.json"
    ) -> Dict[str, Any]:
        """
        Complete conversion pipeline

        Args:
            metadata: Metadata to include in the conversion (overrides instance metadata)
            output_file: Output file path, if None auto-generated

        Returns:
            dict: GeoCroissant metadata
        """
        print("Starting conversion for data...")

        # Load dataset
        if not self.load_dataset():
            return {}

        # Generate metadata
        croissant_record = self.create_croissant_metadata(output_file=str(output_file))

        print("Conversion completed successfully!")
        return croissant_record


def write_croissant_file(
        data_uri: str | Path, 
        output_file: str | Path = None, 
        metadata: Optional[Dict[str, Any]] = None,
        preprocessors: list[dict] = None) -> None:
    """
    Write a Croissant JSON-LD record for a Zarr file. Use input metadata about the original data source 
    and the preprocessors applied to create a traceable record.

    Args:
        data_uri: URI or path to the Zarr file (e.g., S3 URL or local path)
        output_file: path to write the Croissant JSON-LD record to (if None: generate from data_uri)
        metadata: Optional metadata to include in the Croissant record
        preprocessors: List of preprocessors applied to the data

    Returns:
        None
    """
    metadata = metadata or {}
    data_uri = str(data_uri)

    # If there are preprocessors, then send them to the `build_additional_property` function to 
    # create the `additionalProperty` block for the Croissant record
    if preprocessors:
        additional_property = build_additional_property(preprocessors=preprocessors)
        metadata["additionalProperty"] = additional_property

    # Set some defaults
    data_uri_stem = Path(data_uri).stem
    metadata.setdefault("name", f"Dataset for {data_uri_stem}")
    metadata.setdefault("description", f"Dataset converted from Zarr file at {data_uri}. Generated by FRAME-FM package, courtesy of NERC-EDS.")
    metadata.setdefault("url", data_uri)

    converter = ZarrToCroissantConverter(
        zarr_url=data_uri,
        metadata=metadata,
        verbose=False
    )

    # If output_file is not provided, generate a default name based on the data_uri
    if output_file is None:
        dri = Path(data_uri)
        output_file = str(dri.with_name(dri.stem + "_croissant.json"))
    
    converter.convert(output_file=output_file)
    print(f"Croissant record written to {output_file}")
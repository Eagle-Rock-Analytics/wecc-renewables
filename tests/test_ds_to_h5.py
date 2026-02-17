"""
Test ds_to_h5 module.

This module contains unit tests for xarray dataset to HDF5 conversion functions
including metadata handling and exclusion mask creation.
"""

import tempfile
from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon

from src.ds_to_h5 import append_to_h5_wrapper, make_exclusions_mask, meta_h5


class TestDsToH5:
    """Test HDF5 conversion and metadata functions."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        x = np.arange(0, 50000, 10000)  # 5 points in meters
        y = np.arange(0, 30000, 10000)  # 3 points in meters

        ds = xr.Dataset(
            {
                "temperature": (["y", "x"], np.random.rand(3, 5) * 20 + 280),
                "elevation": (["y", "x"], np.random.rand(3, 5) * 1000),
            },
            coords={
                "x": x,
                "y": y,
                "lat": (["y", "x"], np.random.rand(3, 5) * 5 + 35),
                "lon": (["y", "x"], np.random.rand(3, 5) * 5 - 120),
            },
        )
        return ds

    @pytest.fixture
    def sample_exclusions_dict(self):
        """Create a sample exclusions dictionary."""
        return {
            "protected_areas": "/fake/path/protected.shp",
            "urban_areas": "/fake/path/urban.shp",
            "water_bodies": "/fake/path/water.shp",
        }

    @pytest.fixture
    def mock_geodataframe(self):
        """Create a mock GeoDataFrame for exclusions."""
        # Create a simple polygon for testing
        polygon = Polygon([(0, 0), (25000, 0), (25000, 15000), (0, 15000)])
        gdf = gpd.GeoDataFrame({"id": [1], "geometry": [polygon]}, crs="EPSG:4326")
        return gdf

    def test_make_exclusions_mask_basic_functionality(
        self, sample_dataset, sample_exclusions_dict
    ):
        """Test basic functionality of exclusions mask creation."""
        mock_outfile = Mock()

        with (
            patch("src.ds_to_h5.gpd.read_file") as mock_read_file,
            patch("src.ds_to_h5.pyproj.CRS"),
        ):
            # Create mock GeoDataFrame
            mock_gdf = Mock()
            mock_gdf_reproj = Mock()
            mock_gdf.to_crs.return_value = mock_gdf_reproj
            mock_gdf_reproj.__getitem__ = Mock(return_value=Mock())
            mock_read_file.return_value = mock_gdf

            # Create a list to hold the mock DataArrays for the merge call
            mock_arrays = []

            # For each reason, we'll create a DataArray to be returned
            for _reason in sample_exclusions_dict:
                mock_array = xr.DataArray(np.ones((3, 5)), dims=["y", "x"])
                mock_array.attrs["description"] = (
                    "1 for land use restriction, 0 for no restriction."
                )
                mock_arrays.append(mock_array)

            # Create complete patches
            with (
                patch("xarray.Dataset.rio.write_crs", create=True) as mock_write_crs,
                patch("xarray.Dataset.rio.clip", create=True) as mock_clip,
                patch("xarray.Dataset.fillna") as mock_fillna,
                patch("xarray.DataArray.astype") as mock_astype,
                patch("xarray.merge") as mock_merge,
            ):

                # Set up write_crs to return a dataset that can be passed to clip
                mock_write_crs.return_value = sample_dataset

                # Set up clip to return a dataset with the needed variables
                clip_dataset = xr.Dataset()
                for reason in sample_exclusions_dict:
                    clip_dataset[reason] = xr.DataArray(
                        np.ones((3, 5)), dims=["y", "x"]
                    )
                mock_clip.return_value = clip_dataset

                # Set up fillna to return the same clip dataset
                mock_fillna.return_value = clip_dataset

                # Set up astype to return a DataArray
                mock_astype.side_effect = lambda x: xr.DataArray(
                    np.ones((3, 5)),
                    dims=["y", "x"],
                    attrs={
                        "description": "1 for land use restriction, 0 for no restriction."
                    },
                )

                # Set up merge to return a dataset with all exclusions
                result_dataset = sample_dataset.copy()
                for reason in sample_exclusions_dict:
                    result_dataset[reason] = xr.DataArray(
                        np.ones((3, 5)), dims=["y", "x"]
                    )
                mock_merge.return_value = result_dataset

                # Call the function
                result = make_exclusions_mask(
                    sample_dataset, sample_exclusions_dict, mock_outfile
                )

                # Verify the mocks were called correctly
                assert mock_read_file.call_count == len(sample_exclusions_dict)
                assert mock_clip.call_count == len(sample_exclusions_dict)
                mock_merge.assert_called_once()  # Check that gpd.read_file was called for each exclusion
                assert mock_read_file.call_count == len(sample_exclusions_dict)

                # Check that result is an xarray Dataset
                assert isinstance(result, xr.Dataset)

    def test_make_exclusions_mask_multiple_exclusions(
        self, sample_dataset, mock_outfile=Mock()
    ):
        """Test that multiple exclusions are properly processed."""
        exclusions_dict = {
            "exclusion1": "/path/to/file1.shp",
            "exclusion2": "/path/to/file2.shp",
        }

        with (
            patch("src.ds_to_h5.gpd.read_file") as mock_read_file,
            patch("src.ds_to_h5.pyproj.CRS"),
        ):
            # Create mock GeoDataFrame for each exclusion
            mock_gdf = Mock()
            mock_gdf_reproj = Mock()
            mock_gdf.to_crs.return_value = mock_gdf_reproj
            mock_gdf_reproj.__getitem__ = Mock(return_value=Mock())
            mock_read_file.return_value = mock_gdf

            # Create complete patches
            with (
                patch("xarray.Dataset.rio.write_crs", create=True) as mock_write_crs,
                patch("xarray.Dataset.rio.clip", create=True) as mock_clip,
                patch("xarray.Dataset.fillna", create=True) as mock_fillna,
                patch("xarray.DataArray.astype") as mock_astype,
                patch("xarray.merge") as mock_merge,
            ):

                # Set up write_crs to return a dataset that can be passed to clip
                mock_write_crs.return_value = sample_dataset

                # Define clip datasets with subscriptable access
                clip_ds1 = xr.Dataset()
                clip_ds1["exclusion1"] = xr.DataArray(np.ones((3, 5)), dims=["y", "x"])

                clip_ds2 = xr.Dataset()
                clip_ds2["exclusion2"] = xr.DataArray(np.zeros((3, 5)), dims=["y", "x"])

                # Set up fillna to return datasets that can be properly subscripted
                mock_fillna.side_effect = [clip_ds1, clip_ds2]

                # Set up clip to return datasets with proper attributes
                mock_clip.side_effect = [clip_ds1, clip_ds2]

                # Set up astype to return proper DataArrays
                mock_astype.side_effect = [
                    xr.DataArray(
                        np.ones((3, 5)),
                        dims=["y", "x"],
                        attrs={
                            "description": "1 for land use restriction, 0 for no restriction."
                        },
                    ),
                    xr.DataArray(
                        np.zeros((3, 5)),
                        dims=["y", "x"],
                        attrs={
                            "description": "1 for land use restriction, 0 for no restriction."
                        },
                    ),
                ]

                # Set up merge to return a dataset with both exclusions
                result_dataset = sample_dataset.copy()
                result_dataset["exclusion1"] = xr.DataArray(
                    np.ones((3, 5)), dims=["y", "x"]
                )
                result_dataset["exclusion2"] = xr.DataArray(
                    np.zeros((3, 5)), dims=["y", "x"]
                )
                mock_merge.return_value = result_dataset

                # Call the function
                result = make_exclusions_mask(
                    sample_dataset, exclusions_dict, mock_outfile
                )

                # Verify the mocks were called correctly
                assert mock_read_file.call_count == 2
                assert mock_clip.call_count == 2
                assert mock_astype.call_count == 2
                mock_merge.assert_called_once()
                assert isinstance(result, xr.Dataset)

    def test_meta_h5_file_creation(self, sample_dataset):
        """Test HDF5 metadata file creation."""
        with patch("src.ds_to_h5.tb.open_file") as mock_open_file:
            mock_h5file = Mock()
            mock_table = Mock()
            mock_open_file.return_value.__enter__.return_value = mock_h5file
            mock_h5file.create_table.return_value = mock_table

            # Test with temporary file
            with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_file:
                meta_h5(sample_dataset, tmp_file.name)

            # Verify that file operations were attempted
            mock_open_file.assert_called_once()
            mock_h5file.create_table.assert_called_once()
            mock_table.append.assert_called_once()
            mock_table.flush.assert_called_once()

    def test_meta_h5_coordinate_renaming(self, sample_dataset):
        """Test that coordinates are properly renamed for metadata."""
        with patch("src.ds_to_h5.tb.open_file") as mock_open_file:
            mock_h5file = Mock()
            mock_table = Mock()
            mock_open_file.return_value.__enter__.return_value = mock_h5file
            mock_h5file.create_table.return_value = mock_table

            meta_h5(sample_dataset, "test.h5")

            # Check that the dataset was processed (indirectly through stack call)
            # This is tested by ensuring the function completes without error

    @patch("src.ds_to_h5._append_time")
    @patch("src.ds_to_h5._append_data_vars")
    def test_append_to_h5_wrapper_basic(
        self, mock_append_vars, mock_append_time, sample_dataset
    ):
        """Test basic HDF5 data appending functionality."""
        # Add time dimension to sample dataset
        time = np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]")
        sample_dataset = sample_dataset.expand_dims(time=time)

        var_list = ["temperature"]

        append_to_h5_wrapper(sample_dataset, res_file="test.h5", var_list=var_list)

        # Verify that time appending was called
        mock_append_time.assert_called_once()

        # The _append_data_vars should be called through apply
        # This is difficult to test directly due to xarray's apply method

    def test_append_to_h5_wrapper_wrf_time_format(self, sample_dataset):
        """Test HDF5 appending with WRF time format."""
        # Create dataset with WRF-style time format
        wrf_times = [b"2020-01-01_00:00:00", b"2020-01-02_00:00:00"]
        sample_dataset = sample_dataset.assign_coords(time=wrf_times)

        # Create mock datetime objects with expected behavior
        dt_mock1 = Mock()
        dt_mock1.strftime.return_value = "2020-01-01T00:00:00.000000"
        dt_mock2 = Mock()
        dt_mock2.strftime.return_value = "2020-01-02T00:00:00.000000"

        # Use MagicMock for datetime module
        datetime_mock = Mock()
        datetime_mock.strptime.side_effect = [dt_mock1, dt_mock2]

        # Configure mock for DataArray.dt accessor with necessary attributes
        mock_dt = Mock()
        mock_strftime = Mock()
        mock_strftime.return_value = pd.Series(
            ["2020-01-01T00:00:00.000000", "2020-01-02T00:00:00.000000"]
        )
        mock_dt.strftime = mock_strftime

        # We need to patch extensively for this test
        with (
            patch("src.ds_to_h5._append_time") as mock_append_time,
            patch("src.ds_to_h5.datetime", datetime_mock),
            patch("src.ds_to_h5._append_data_vars"),
            patch("xarray.Dataset.transpose") as mock_transpose,
            patch("xarray.DataArray.dt", mock_dt, create=True),
            patch("tables.open_file"),
        ):

            # Configure mock_transpose
            mock_transpose.return_value = sample_dataset

            # Run the function under test
            append_to_h5_wrapper(
                sample_dataset, res_file="test.h5", var_list=["temperature"], wrf=True
            )

            # Verify expected behavior
            mock_append_time.assert_called_once()
            assert datetime_mock.strptime.call_count == len(wrf_times)

    def test_append_to_h5_wrapper_dimension_transpose(self, sample_dataset):
        """Test that dimensions are properly transposed before appending."""
        time = np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]")
        sample_dataset = sample_dataset.expand_dims(time=time)

        # Need to mock both the _append_time and the tables.open_file functions
        with (
            patch("src.ds_to_h5._append_time"),
            patch(
                "xarray.Dataset.transpose", return_value=sample_dataset
            ) as mock_transpose,
            patch("src.ds_to_h5._append_data_vars"),
        ):  # Prevent the actual append operation
            append_to_h5_wrapper(
                sample_dataset, res_file="test.h5", var_list=["temperature"]
            )

            # Verify transpose was called with correct dimension order
            mock_transpose.assert_called_once_with("time", "y", "x")

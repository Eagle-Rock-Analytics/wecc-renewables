"""
Test zarr_to_pv_resource module.

This module contains unit tests for solar resource data processing functions
including variable derivation and data preprocessing for PV analysis.
"""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from src.zarr_to_pv_resource import (
    _compute_dewpointtemp,
    _derive_surface_albedo,
    _wind_10m_speed,
    preprocess_pv_wrapper,
)


class TestZarrToPvResource:
    """Test solar resource data processing functions."""

    @pytest.fixture
    def sample_pv_dataset(self):
        """Create a sample PV dataset for testing."""
        time = np.arange("2020-01-01T00", "2020-01-02T00", dtype="datetime64[h]")
        x = np.arange(5)
        y = np.arange(3)

        # Create sample solar and meteorological data
        ghi_data = np.random.rand(24, 3, 5) * 800  # Global horizontal irradiance
        dni_data = np.random.rand(24, 3, 5) * 900  # Direct normal irradiance
        dhi_data = np.random.rand(24, 3, 5) * 200  # Diffuse horizontal irradiance
        swupb_data = ghi_data * 0.2  # Upward shortwave (for albedo calc)
        t2_data = np.random.rand(24, 3, 5) * 20 + 280  # Temperature in Kelvin
        snow_data = np.random.rand(24, 3, 5) * 10  # Snow depth
        u10_data = np.random.rand(24, 3, 5) * 10  # U wind component
        v10_data = np.random.rand(24, 3, 5) * 10  # V wind component

        ds = xr.Dataset(
            {
                "swdnb": (["time", "y", "x"], ghi_data),
                "swddni": (["time", "y", "x"], dni_data),
                "swddif": (["time", "y", "x"], dhi_data),
                "swupb": (["time", "y", "x"], swupb_data),
                "t2": (["time", "y", "x"], t2_data),
                "snownc": (["time", "y", "x"], snow_data),
                "u10": (["time", "y", "x"], u10_data),
                "v10": (["time", "y", "x"], v10_data),
            },
            coords={"time": time, "x": x, "y": y},
        )

        ds.attrs = {"test_attr": "test_value"}
        return ds

    @pytest.fixture
    def dewpoint_dataset(self):
        """Create dataset for dew point calculation testing."""
        time = np.arange("2020-01-01T00", "2020-01-01T12", dtype="datetime64[h]")
        x = np.arange(3)
        y = np.arange(2)

        ds = xr.Dataset(
            {
                "air_temperature": (
                    ["time", "y", "x"],
                    np.random.rand(12, 2, 3) * 20 + 273.15,
                ),
                "q2": (
                    ["time", "y", "x"],
                    np.random.rand(12, 2, 3) * 0.02,
                ),  # Specific humidity
                "surface_pressure": (
                    ["time", "y", "x"],
                    np.random.rand(12, 2, 3) * 5000 + 98000,
                ),
            },
            coords={"time": time, "x": x, "y": y},
        )
        return ds

    def test_preprocess_pv_wrapper_variable_renaming(self, sample_pv_dataset):
        """Test that variables are correctly renamed in preprocessing."""
        with (
            patch("src.zarr_to_pv_resource._derive_surface_albedo") as mock_albedo,
            patch("src.zarr_to_pv_resource._wind_10m_speed") as mock_wind,
        ):

            # Mock the derived variable functions
            mock_albedo.return_value = xr.DataArray(
                np.random.rand(24, 3, 5), dims=["time", "y", "x"]
            )
            mock_wind.return_value = xr.DataArray(
                np.random.rand(24, 3, 5), dims=["time", "y", "x"]
            )

            result = preprocess_pv_wrapper(sample_pv_dataset)

            # Check that variables were renamed
            assert "ghi" in result.data_vars
            assert "dni" in result.data_vars
            assert "dhi" in result.data_vars
            assert "air_temperature" in result.data_vars
            assert "snow_depth" in result.data_vars

            # Check that original names are gone
            assert "swdnb" not in result.data_vars
            assert "swddni" not in result.data_vars
            assert "swddif" not in result.data_vars
            assert "t2" not in result.data_vars
            assert "snownc" not in result.data_vars

    def test_preprocess_pv_wrapper_temperature_conversion(self, sample_pv_dataset):
        """Test temperature conversion from Kelvin to Celsius."""
        with (
            patch("src.zarr_to_pv_resource._derive_surface_albedo") as mock_albedo,
            patch("src.zarr_to_pv_resource._wind_10m_speed") as mock_wind,
        ):

            mock_albedo.return_value = xr.DataArray(
                np.random.rand(24, 3, 5), dims=["time", "y", "x"]
            )
            mock_wind.return_value = xr.DataArray(
                np.random.rand(24, 3, 5), dims=["time", "y", "x"]
            )

            original_temp_k = sample_pv_dataset["t2"].values
            result = preprocess_pv_wrapper(sample_pv_dataset)

            # Temperature should be converted from K to C
            expected_temp_c = original_temp_k - 273.15
            np.testing.assert_array_almost_equal(
                result["air_temperature"].values, expected_temp_c, decimal=6
            )

    def test_preprocess_pv_wrapper_snow_depth_conversion(self, sample_pv_dataset):
        """Test snow depth unit conversion."""
        with (
            patch("src.zarr_to_pv_resource._derive_surface_albedo") as mock_albedo,
            patch("src.zarr_to_pv_resource._wind_10m_speed") as mock_wind,
        ):

            mock_albedo.return_value = xr.DataArray(
                np.random.rand(24, 3, 5), dims=["time", "y", "x"]
            )
            mock_wind.return_value = xr.DataArray(
                np.random.rand(24, 3, 5), dims=["time", "y", "x"]
            )

            original_snow = sample_pv_dataset["snownc"].values
            result = preprocess_pv_wrapper(sample_pv_dataset)

            # Snow depth should be multiplied by 0.1
            expected_snow = original_snow * 0.1
            np.testing.assert_array_almost_equal(
                result["snow_depth"].values, expected_snow, decimal=6
            )

    def test_preprocess_pv_wrapper_attributes_preserved(self, sample_pv_dataset):
        """Test that dataset attributes are preserved."""
        with (
            patch("src.zarr_to_pv_resource._derive_surface_albedo") as mock_albedo,
            patch("src.zarr_to_pv_resource._wind_10m_speed") as mock_wind,
        ):

            mock_albedo.return_value = xr.DataArray(
                np.random.rand(24, 3, 5), dims=["time", "y", "x"]
            )
            mock_wind.return_value = xr.DataArray(
                np.random.rand(24, 3, 5), dims=["time", "y", "x"]
            )

            result = preprocess_pv_wrapper(sample_pv_dataset)

            # Attributes should be preserved
            assert result.attrs == sample_pv_dataset.attrs

    def test_preprocess_pv_wrapper_dimension_order(self, sample_pv_dataset):
        """Test that dimensions are correctly ordered."""
        with (
            patch("src.zarr_to_pv_resource._derive_surface_albedo") as mock_albedo,
            patch("src.zarr_to_pv_resource._wind_10m_speed") as mock_wind,
        ):

            mock_albedo.return_value = xr.DataArray(
                np.random.rand(24, 3, 5), dims=["time", "y", "x"]
            )
            mock_wind.return_value = xr.DataArray(
                np.random.rand(24, 3, 5), dims=["time", "y", "x"]
            )

            result = preprocess_pv_wrapper(sample_pv_dataset)

            # Check dimension order
            for var in result.data_vars:
                if len(result[var].dims) == 3:
                    assert result[var].dims == ("time", "y", "x")

    def test_compute_dewpointtemp(self, dewpoint_dataset):
        """Test dew point temperature calculation."""
        with patch("src.zarr_to_pv_resource.relative_humidity") as mock_rh:
            # Mock relative humidity calculation
            mock_rh.return_value = xr.DataArray(
                np.random.rand(12, 2, 3) * 80 + 10, dims=["time", "y", "x"]  # 10-90% RH
            )

            result = _compute_dewpointtemp(dewpoint_dataset)

            # Check that result has correct properties
            assert isinstance(result, xr.DataArray)
            assert result.name == "dew_point"
            assert result.attrs["units"] == "C"
            assert result.shape == dewpoint_dataset["air_temperature"].shape

            # Dew point should be less than air temperature
            air_temp_c = dewpoint_dataset["air_temperature"] - 273.15
            assert np.all(result.values <= air_temp_c.values)

    def test_derive_surface_albedo(self, sample_pv_dataset):
        """Test surface albedo calculation."""
        # Need to rename the variable to 'ghi' as the function expects
        ds_renamed = sample_pv_dataset.rename({"swdnb": "ghi"})
        result = _derive_surface_albedo(ds_renamed)

        # Check basic properties
        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_pv_dataset["swdnb"].shape

        # Albedo should be between 0 and 1 (after fillna)
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 1)

        # Check that it's calculated as upward/downward radiation
        expected = sample_pv_dataset["swupb"] / sample_pv_dataset["swdnb"]
        expected = expected.fillna(0)
        np.testing.assert_array_equal(result.values, expected.values)

    def test_derive_surface_albedo_handles_nan(self):
        """Test that surface albedo handles NaN values correctly."""
        # Create dataset with some zero GHI values
        time = np.arange("2020-01-01T00", "2020-01-01T03", dtype="datetime64[h]")
        ds = xr.Dataset(
            {
                "ghi": (
                    ["time"],
                    [0, 100, 200],
                ),  # Zero GHI should cause division by zero
                "swupb": (["time"], [0, 20, 40]),
            },
            coords={"time": time},
        )

        result = _derive_surface_albedo(ds)

        # First value should be 0 (filled NaN), others should be calculated
        assert result.values[0] == 0
        assert result.values[1] == 0.2  # 20/100
        assert result.values[2] == 0.2  # 40/200

    def test_wind_10m_speed(self, sample_pv_dataset):
        """Test wind speed calculation."""
        with patch("src.zarr_to_pv_resource.uas_vas_2_sfcwind") as mock_wind_calc:
            # Mock wind speed calculation
            expected_speed = np.random.rand(24, 3, 5) * 15
            mock_wind_calc.return_value = (
                xr.DataArray(expected_speed, dims=["time", "y", "x"]),
                xr.DataArray(
                    np.random.rand(24, 3, 5) * 360, dims=["time", "y", "x"]
                ),  # direction
            )

            result = _wind_10m_speed(sample_pv_dataset)

            # Check that function was called with correct parameters
            # We need to check the call arguments manually since xarray DataArrays
            # can't be compared directly in assertions
            mock_wind_calc.assert_called_once()
            call_args = mock_wind_calc.call_args

            # Check keyword arguments
            assert "uas" in call_args.kwargs
            assert "vas" in call_args.kwargs
            assert "calm_wind_thresh" in call_args.kwargs
            assert call_args.kwargs["calm_wind_thresh"] == "0.5 m/s"

            # Check that the right data arrays were passed (by checking their values)
            np.testing.assert_array_equal(
                call_args.kwargs["uas"].values, sample_pv_dataset.u10.values
            )
            np.testing.assert_array_equal(
                call_args.kwargs["vas"].values, sample_pv_dataset.v10.values
            )

            # Check that only speed is returned (not direction)
            np.testing.assert_array_equal(result.values, expected_speed)

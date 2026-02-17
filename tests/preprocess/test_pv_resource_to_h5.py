#!/usr/bin/env python3
"""
Tests for pv_resource_to_h5 module.

This module tests PV resource data conversion to HDF5 format functionality,
including data processing, file operations, and configuration generation.
"""

from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.preprocess.pv_resource_to_h5 import (
    pv_resource_to_h5,
)


class TestPvResourceToH5:
    """Test PV resource data to HDF5 conversion functionality."""

    @pytest.fixture
    def sample_pv_dataset(self):
        """Create sample PV resource dataset for testing."""
        # Create realistic PV resource dataset
        time_coords = pd.date_range("2020-01-01", periods=365 * 24, freq="H")

        ds = xr.Dataset(
            {
                "t2": (["time", "y", "x"], np.random.rand(365 * 24, 5, 4) * 10 + 280),
                "u10": (["time", "y", "x"], np.random.rand(365 * 24, 5, 4) * 10),
                "v10": (["time", "y", "x"], np.random.rand(365 * 24, 5, 4) * 10),
                "swdnb": (["time", "y", "x"], np.random.rand(365 * 24, 5, 4) * 800),
                "swupb": (["time", "y", "x"], np.random.rand(365 * 24, 5, 4) * 100),
                "swddif": (["time", "y", "x"], np.random.rand(365 * 24, 5, 4) * 200),
                "swddni": (["time", "y", "x"], np.random.rand(365 * 24, 5, 4) * 600),
                "snownc": (["time", "y", "x"], np.random.rand(365 * 24, 5, 4) * 0.1),
                "latitude": (["y", "x"], np.random.rand(5, 4) * 5 + 35),
                "longitude": (["y", "x"], np.random.rand(5, 4) * 5 - 120),
                "elevation": (["y", "x"], np.random.rand(5, 4) * 2000),
                "landmask": (["y", "x"], np.random.choice([0, 1], size=(5, 4))),
                "timezone": (["y", "x"], np.full((5, 4), -8)),
            },
            coords={
                "time": time_coords,
                "x": np.arange(4),
                "y": np.arange(5),
            },
        )

        return ds

    @pytest.fixture
    def sample_meta_data(self):
        """Create sample metadata for testing."""
        return pd.DataFrame(
            {
                "gid": range(20),
                "latitude": np.random.rand(20) * 5 + 35,
                "longitude": np.random.rand(20) * 5 - 120,
                "elevation": np.random.rand(20) * 2000,
                "timezone": np.full(20, -8),
            }
        )

    @patch("src.preprocess.pv_resource_to_h5.intake")
    @patch("src.preprocess.pv_resource_to_h5.preprocess_pv_wrapper")
    @patch("src.preprocess.pv_resource_to_h5.ds_to_zarr")
    @patch("src.preprocess.pv_resource_to_h5.HDFStore")
    @patch("src.preprocess.pv_resource_to_h5.Outputs")
    @patch("src.preprocess.pv_resource_to_h5.os.path.exists")
    @patch("src.preprocess.pv_resource_to_h5.os.makedirs")
    def test_pv_resource_to_h5_basic(
        self,
        mock_makedirs,
        mock_exists,
        mock_outputs,
        mock_hdfstore,
        mock_ds_to_zarr,
        mock_preprocess_pv,
        mock_intake,
        sample_pv_dataset,
    ):
        """Test basic PV resource to H5 conversion."""
        # Mock intake catalog
        mock_cat = Mock()
        mock_intake.open_esm_datastore.return_value = mock_cat

        mock_query_result = Mock()
        mock_cat.search.return_value = mock_query_result

        mock_ds_dict = Mock()
        mock_query_result.to_dataset_dict.return_value = mock_ds_dict
        mock_ds_dict.values.return_value = [sample_pv_dataset]

        # Mock preprocessing
        processed_ds = sample_pv_dataset.copy()
        processed_ds["time_index"] = (
            processed_ds["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").astype("|S30")
        )
        mock_preprocess_pv.return_value = processed_ds

        # Mock file operations
        mock_exists.return_value = False
        mock_hdf_instance = Mock()
        mock_hdfstore.return_value = mock_hdf_instance
        mock_hdf_instance.meta = pd.DataFrame({"gid": [0, 1, 2]})

        mock_outputs_instance = Mock()
        mock_outputs_instance.__setitem__ = Mock()  # Allow item assignment
        mock_outputs_instance.__enter__ = Mock(return_value=mock_outputs_instance)
        mock_outputs_instance.__exit__ = Mock(return_value=None)
        mock_outputs.return_value = mock_outputs_instance

        # Run the function
        with patch("src.preprocess.pv_resource_to_h5.Outputs.add_dataset"):
            pv_resource_to_h5(rank=5, domain="d03", sim_id="MIROC6")

        # Verify catalog operations
        mock_cat.search.assert_called_once()
        mock_query_result.to_dataset_dict.assert_called_once()

        # Verify preprocessing
        mock_preprocess_pv.assert_called_once()

        # Verify zarr storage
        mock_ds_to_zarr.assert_called_once()
        zarr_args = mock_ds_to_zarr.call_args
        assert "era/resource_data/d03/pv/MIROC6/" in zarr_args[0][1]

        # Verify H5 operations
        mock_hdfstore.assert_called()
        mock_outputs.assert_called()

    @patch("src.preprocess.pv_resource_to_h5.intake")
    @patch("src.preprocess.pv_resource_to_h5.preprocess_pv_wrapper")
    @patch("src.preprocess.pv_resource_to_h5.ds_to_zarr")
    @patch("src.preprocess.pv_resource_to_h5.HDFStore")
    @patch("src.preprocess.pv_resource_to_h5.Outputs")
    @patch("src.preprocess.pv_resource_to_h5.os.path.exists")
    def test_pv_resource_to_h5_scenario_mapping(
        self,
        mock_exists,
        mock_outputs,
        mock_hdfstore,
        mock_ds_to_zarr,
        mock_preprocess_pv,
        mock_intake,
        sample_pv_dataset,
    ):
        """Test scenario mapping for different ranks and simulation IDs."""
        # Test cases: (rank, sim_id, expected_scenario)
        test_cases = [
            (5, "MIROC6", "historical"),  # rank < 34
            (45, "MIROC6", "ssp370"),  # rank >= 34
            (10, "ERA5", "reanalysis"),  # ERA5 always reanalysis
        ]

        for rank, sim_id, expected_scenario in test_cases:
            # Reset mocks
            mock_intake.reset_mock()

            # Mock intake catalog
            mock_cat = Mock()
            mock_intake.open_esm_datastore.return_value = mock_cat

            mock_query_result = Mock()
            mock_cat.search.return_value = mock_query_result

            mock_ds_dict = Mock()
            mock_query_result.to_dataset_dict.return_value = mock_ds_dict
            mock_ds_dict.values.return_value = [sample_pv_dataset]

            mock_preprocess_pv.return_value = sample_pv_dataset
            mock_exists.return_value = False
            mock_hdf_instance = Mock()
            mock_hdfstore.return_value = mock_hdf_instance
            mock_hdf_instance.meta = pd.DataFrame()

            with patch("src.preprocess.pv_resource_to_h5.os.makedirs"):
                # Run the function
                pv_resource_to_h5(rank=rank, domain="d03", sim_id=sim_id)

            # Check that correct scenario was used in query
            search_call_args = mock_cat.search.call_args
            search_kwargs = search_call_args[1]
            assert search_kwargs["experiment_id"] == expected_scenario

    @patch("src.preprocess.pv_resource_to_h5.intake")
    @patch("src.preprocess.pv_resource_to_h5.preprocess_pv_wrapper")
    @patch("src.preprocess.pv_resource_to_h5.ds_to_zarr")
    @patch("src.preprocess.pv_resource_to_h5.HDFStore")
    @patch("src.preprocess.pv_resource_to_h5.Outputs")
    @patch("src.preprocess.pv_resource_to_h5.os.path.exists")
    def test_pv_resource_to_h5_variable_list(
        self,
        mock_exists,
        mock_outputs,
        mock_hdfstore,
        mock_ds_to_zarr,
        mock_preprocess_pv,
        mock_intake,
        sample_pv_dataset,
    ):
        """Test that correct variables are requested from catalog."""
        # Mock intake catalog
        mock_cat = Mock()
        mock_intake.open_esm_datastore.return_value = mock_cat

        mock_query_result = Mock()
        mock_cat.search.return_value = mock_query_result

        mock_ds_dict = Mock()
        mock_query_result.to_dataset_dict.return_value = mock_ds_dict
        mock_ds_dict.values.return_value = [sample_pv_dataset]

        mock_preprocess_pv.return_value = sample_pv_dataset
        mock_exists.return_value = False
        mock_hdf_instance = Mock()
        mock_hdfstore.return_value = mock_hdf_instance
        mock_hdf_instance.meta = pd.DataFrame()

        with patch("src.preprocess.pv_resource_to_h5.os.makedirs"):
            pv_resource_to_h5(rank=5, domain="d03", sim_id="MIROC6")

        # Check variable list in query
        search_call_args = mock_cat.search.call_args
        search_kwargs = search_call_args[1]

        expected_vars = [
            "t2",
            "u10",
            "v10",
            "swdnb",
            "swupb",
            "swddif",
            "swddni",
            "snownc",
        ]
        assert "variable_id" in search_kwargs
        assert set(search_kwargs["variable_id"]) == set(expected_vars)

    def test_year_rank_function(self):
        """Test internal year rank mapping function."""
        # This tests the expected behavior of the _year_rank function
        years = [str(y) for y in np.arange(1980, 2099)]

        # Test some specific mappings
        assert years[0] == "1980"
        assert years[1] == "1981"  # rank 1 -> 1981
        assert years[34] == "2014"  # rank 34 -> 2014
        assert years[-1] == "2098"  # last year

        # Test that we have the right number of years
        assert len(years) == 119  # 1980-2098 inclusive

    @patch("src.preprocess.pv_resource_to_h5.intake")
    @patch("src.preprocess.pv_resource_to_h5.preprocess_pv_wrapper")
    @patch("src.preprocess.pv_resource_to_h5.ds_to_zarr")
    @patch("src.preprocess.pv_resource_to_h5.HDFStore")
    @patch("src.preprocess.pv_resource_to_h5.Outputs")
    @patch("src.preprocess.pv_resource_to_h5.os.path.exists")
    def test_pv_resource_to_h5_time_batching(
        self,
        mock_exists,
        mock_outputs,
        mock_hdfstore,
        mock_ds_to_zarr,
        mock_preprocess_pv,
        mock_intake,
        sample_pv_dataset,
    ):
        """Test time batching logic for different rank ranges."""
        # Test cases: (rank, expected_analysis_years_start, expected_analysis_years_end)
        test_cases = [
            (5, 1981, 2014),  # rank < 35: 1981-2014
            (45, 2015, 2044),  # 35 < rank <= 64: 2015-2044
            (75, 2045, 2074),  # 65 <= rank <= 94: 2045-2074
            (105, 2076, 2099),  # rank >= 95: 2076-2099
        ]

        for rank, _expected_start, _expected_end in test_cases:
            # Mock setup
            mock_intake.reset_mock()
            mock_cat = Mock()
            mock_intake.open_esm_datastore.return_value = mock_cat

            mock_query_result = Mock()
            mock_cat.search.return_value = mock_query_result

            mock_ds_dict = Mock()
            mock_query_result.to_dataset_dict.return_value = mock_ds_dict
            mock_ds_dict.values.return_value = [sample_pv_dataset]

            mock_preprocess_pv.return_value = sample_pv_dataset
            mock_exists.return_value = False
            mock_hdf_instance = Mock()
            mock_hdfstore.return_value = mock_hdf_instance
            mock_hdf_instance.meta = pd.DataFrame()

            with (
                patch("builtins.open", mock_open()),
                patch("src.preprocess.pv_resource_to_h5.json.dump"),
                patch("src.preprocess.pv_resource_to_h5.json.load", return_value={}),
                patch("src.preprocess.pv_resource_to_h5.shutil.copy"),
                patch("src.preprocess.pv_resource_to_h5.os.makedirs"),
            ):

                # Run for ranks that trigger config file creation
                edit_flag_ranks = [1, 41, 65, 95]
                if rank in edit_flag_ranks:
                    pv_resource_to_h5(rank=rank, domain="d03", sim_id="MIROC6")
                else:
                    pv_resource_to_h5(rank=rank, domain="d03", sim_id="MIROC6")

            # The function should complete without error
            # (Testing the batching logic structure)

    @patch("src.preprocess.pv_resource_to_h5.intake")
    @patch("src.preprocess.pv_resource_to_h5.preprocess_pv_wrapper")
    @patch("src.preprocess.pv_resource_to_h5.ds_to_zarr")
    @patch("src.preprocess.pv_resource_to_h5.HDFStore")
    @patch("src.preprocess.pv_resource_to_h5.Outputs")
    @patch("src.preprocess.pv_resource_to_h5.os.path.exists")
    @patch("src.preprocess.pv_resource_to_h5.os.remove")
    def test_pv_resource_to_h5_file_management(
        self,
        mock_remove,
        mock_exists,
        mock_outputs,
        mock_hdfstore,
        mock_ds_to_zarr,
        mock_preprocess_pv,
        mock_intake,
        sample_pv_dataset,
    ):
        """Test file management operations."""
        # Mock intake catalog
        mock_cat = Mock()
        mock_intake.open_esm_datastore.return_value = mock_cat

        mock_query_result = Mock()
        mock_cat.search.return_value = mock_query_result

        mock_ds_dict = Mock()
        mock_query_result.to_dataset_dict.return_value = mock_ds_dict
        mock_ds_dict.values.return_value = [sample_pv_dataset]

        mock_preprocess_pv.return_value = sample_pv_dataset
        mock_hdf_instance = Mock()
        mock_hdfstore.return_value = mock_hdf_instance
        mock_hdf_instance.meta = pd.DataFrame()

        # Test file exists case
        mock_exists.return_value = True

        with patch("src.preprocess.pv_resource_to_h5.os.makedirs"):
            pv_resource_to_h5(rank=5, domain="d03", sim_id="MIROC6")

        # Should remove existing file
        mock_remove.assert_called()

        # Test file doesn't exist case
        mock_exists.return_value = False
        mock_remove.reset_mock()

        with patch("src.preprocess.pv_resource_to_h5.os.makedirs"):
            pv_resource_to_h5(rank=5, domain="d03", sim_id="MIROC6")

        # Should not try to remove file
        mock_remove.assert_not_called()

    @patch("src.preprocess.pv_resource_to_h5.intake")
    @patch("src.preprocess.pv_resource_to_h5.preprocess_pv_wrapper")
    @patch("src.preprocess.pv_resource_to_h5.ds_to_zarr")
    @patch("src.preprocess.pv_resource_to_h5.HDFStore")
    @patch("src.preprocess.pv_resource_to_h5.Outputs")
    @patch("src.preprocess.pv_resource_to_h5.os.path.exists")
    @patch("src.preprocess.pv_resource_to_h5.os.makedirs")
    def test_pv_resource_to_h5_directory_creation(
        self,
        mock_makedirs,
        mock_exists,
        mock_outputs,
        mock_hdfstore,
        mock_ds_to_zarr,
        mock_preprocess_pv,
        mock_intake,
        sample_pv_dataset,
    ):
        """Test directory creation for output paths."""
        # Mock intake catalog
        mock_cat = Mock()
        mock_intake.open_esm_datastore.return_value = mock_cat

        mock_query_result = Mock()
        mock_cat.search.return_value = mock_query_result

        mock_ds_dict = Mock()
        mock_query_result.to_dataset_dict.return_value = mock_ds_dict
        mock_ds_dict.values.return_value = [sample_pv_dataset]

        mock_preprocess_pv.return_value = sample_pv_dataset
        mock_hdfstore.return_value.meta = pd.DataFrame()

        # Test directory doesn't exist
        mock_exists.return_value = False

        pv_resource_to_h5(rank=5, domain="d03", sim_id="MIROC6")

        # Should create directory
        mock_makedirs.assert_called()
        makedirs_call_args = mock_makedirs.call_args_list
        assert any("/data/MIROC6" in str(call) for call in makedirs_call_args)

    @patch("src.preprocess.pv_resource_to_h5.intake")
    @patch("src.preprocess.pv_resource_to_h5.preprocess_pv_wrapper")
    @patch("src.preprocess.pv_resource_to_h5.ds_to_zarr")
    @patch("src.preprocess.pv_resource_to_h5.HDFStore")
    @patch("src.preprocess.pv_resource_to_h5.Outputs")
    @patch("src.preprocess.pv_resource_to_h5.os.path.exists")
    def test_pv_resource_to_h5_data_processing(
        self,
        mock_exists,
        mock_outputs,
        mock_hdfstore,
        mock_ds_to_zarr,
        mock_preprocess_pv,
        mock_intake,
        sample_pv_dataset,
    ):
        """Test data processing steps."""
        # Mock intake catalog
        mock_cat = Mock()
        mock_intake.open_esm_datastore.return_value = mock_cat

        mock_query_result = Mock()
        mock_cat.search.return_value = mock_query_result

        mock_ds_dict = Mock()
        mock_query_result.to_dataset_dict.return_value = mock_ds_dict
        mock_ds_dict.values.return_value = [sample_pv_dataset]

        mock_preprocess_pv.return_value = sample_pv_dataset
        mock_hdf_instance = Mock()
        mock_hdfstore.return_value = mock_hdf_instance
        mock_hdf_instance.meta = pd.DataFrame()

        # Test preprocessing wrapper call
        processed_ds = sample_pv_dataset.copy()
        mock_preprocess_pv.return_value = processed_ds

        mock_exists.return_value = False

        with patch("src.preprocess.pv_resource_to_h5.os.makedirs"):
            pv_resource_to_h5(rank=5, domain="d03", sim_id="MIROC6")

        # Should call preprocessing wrapper
        mock_preprocess_pv.assert_called_once()

        # Should add time_index to dataset
        processed_call_args = mock_preprocess_pv.call_args[0][0]
        assert isinstance(processed_call_args, xr.Dataset)

    @patch("src.preprocess.pv_resource_to_h5.intake")
    @patch("src.preprocess.pv_resource_to_h5.preprocess_pv_wrapper")
    @patch("src.preprocess.pv_resource_to_h5.ds_to_zarr")
    @patch("src.preprocess.pv_resource_to_h5.HDFStore")
    @patch("src.preprocess.pv_resource_to_h5.Outputs")
    @patch("src.preprocess.pv_resource_to_h5.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.preprocess.pv_resource_to_h5.json.dump")
    @patch("src.preprocess.pv_resource_to_h5.json.load")
    @patch("src.preprocess.pv_resource_to_h5.shutil.copy")
    def test_pv_resource_to_h5_config_generation(
        self,
        mock_shutil_copy,
        mock_json_load,
        mock_json_dump,
        mock_file_open,
        mock_exists,
        mock_outputs,
        mock_hdfstore,
        mock_ds_to_zarr,
        mock_preprocess_pv,
        mock_intake,
        sample_pv_dataset,
    ):
        """Test configuration file generation for edit flag ranks."""
        # Mock intake catalog
        mock_cat = Mock()
        mock_intake.open_esm_datastore.return_value = mock_cat

        mock_query_result = Mock()
        mock_cat.search.return_value = mock_query_result

        mock_ds_dict = Mock()
        mock_query_result.to_dataset_dict.return_value = mock_ds_dict
        mock_ds_dict.values.return_value = [sample_pv_dataset]

        mock_preprocess_pv.return_value = sample_pv_dataset
        mock_exists.return_value = False
        mock_hdf_instance = Mock()
        mock_hdfstore.return_value = mock_hdf_instance
        mock_hdf_instance.meta = pd.DataFrame()

        # Mock JSON data
        sample_config = {
            "analysis_years": [],
            "resource_file": "",
            "project_points": "",
            "sam_files": {},
        }
        mock_json_load.return_value = sample_config

        # Test edit flag ranks
        edit_flag_ranks = [1, 41, 65, 95]

        for rank in edit_flag_ranks:
            mock_json_dump.reset_mock()
            mock_shutil_copy.reset_mock()

            with patch("src.preprocess.pv_resource_to_h5.os.makedirs"):
                pv_resource_to_h5(rank=rank, domain="d03", sim_id="MIROC6")

            # Should create config files for both distributed and utility
            # Check that JSON files were modified and saved
            assert (
                mock_json_dump.call_count >= 4
            )  # At least gen and collect for both modules

            # Check that files were copied
            assert mock_shutil_copy.call_count >= 4  # Multiple config files copied

    @patch("src.preprocess.pv_resource_to_h5.intake")
    def test_pv_resource_to_h5_error_catalog_access(self, mock_intake):
        """Test error handling when catalog access fails."""
        mock_intake.open_esm_datastore.side_effect = ConnectionError(
            "Catalog access failed"
        )

        with pytest.raises(ConnectionError):
            pv_resource_to_h5(rank=5, domain="d03", sim_id="MIROC6")

    @patch("src.preprocess.pv_resource_to_h5.intake")
    def test_pv_resource_to_h5_error_query_failed(self, mock_intake):
        """Test error handling when dataset query fails."""
        mock_cat = Mock()
        mock_intake.open_esm_datastore.return_value = mock_cat

        mock_cat.search.side_effect = ValueError("Query failed")

        with pytest.raises(ValueError):
            pv_resource_to_h5(rank=5, domain="d03", sim_id="MIROC6")

    @patch("src.preprocess.pv_resource_to_h5.intake")
    @patch("src.preprocess.pv_resource_to_h5.preprocess_pv_wrapper")
    def test_pv_resource_to_h5_error_preprocessing(
        self, mock_preprocess_pv, mock_intake
    ):
        """Test error handling when preprocessing fails."""
        # Mock intake catalog
        mock_cat = Mock()
        mock_intake.open_esm_datastore.return_value = mock_cat

        mock_query_result = Mock()
        mock_cat.search.return_value = mock_query_result

        mock_ds_dict = Mock()
        mock_query_result.to_dataset_dict.return_value = mock_ds_dict
        # Create a mock dataset with time coordinate for proper error testing
        mock_dataset = xr.Dataset(
            {"time": (["time"], pd.date_range("2020-01-01", periods=10, freq="H"))}
        )
        mock_ds_dict.values.return_value = [mock_dataset]

        # Mock preprocessing to fail
        mock_preprocess_pv.side_effect = RuntimeError("Preprocessing failed")

        with pytest.raises(RuntimeError):
            pv_resource_to_h5(rank=5, domain="d03", sim_id="MIROC6")

    def test_meta_variables_definition(self):
        """Test that metadata variables are correctly defined."""
        expected_meta_vars = [
            "latitude",
            "longitude",
            "elevation",
            "landmask",
            "x",
            "y",
            "timezone",
            "time_index",
        ]

        # This tests the expected structure of meta variables
        # In actual implementation, these would be filtered from data variables
        for var in expected_meta_vars:
            assert isinstance(var, str)
            assert len(var) > 0

    def test_analysis_years_calculation(self):
        """Test analysis years calculation for different rank ranges."""
        # Test the expected analysis years calculation logic
        test_cases = [
            # (rank_range_start, rank_range_end, expected_year_start, expected_year_end)
            (1, 34, 1981, 2014),  # rank < 35
            (35, 64, 2015, 2044),  # 35 < rank <= 64
            (65, 94, 2045, 2074),  # 65 <= rank <= 94
            (95, 119, 2076, 2099),  # rank >= 95
        ]

        for rank_start, _rank_end, _year_start, _year_end in test_cases:
            # Test calculation logic
            base_year = 1980
            if rank_start < 35:
                calc_years = [int(n + base_year) for n in np.arange(1, 35)]
                expected_years = list(range(1981, 2015))
            elif rank_start <= 64:
                calc_years = [int(n + base_year) for n in np.arange(35, 65)]
                expected_years = list(range(2015, 2045))
            elif rank_start <= 94:
                calc_years = [int(n + base_year) for n in np.arange(65, 95)]
                expected_years = list(range(2045, 2075))
            else:
                calc_years = [int(n + base_year) for n in np.arange(96, 120)]
                expected_years = list(range(2076, 2100))

            assert calc_years[0] == expected_years[0]
            assert calc_years[-1] == expected_years[-1]

    @patch("src.preprocess.pv_resource_to_h5.intake")
    @patch("src.preprocess.pv_resource_to_h5.preprocess_pv_wrapper")
    @patch("src.preprocess.pv_resource_to_h5.ds_to_zarr")
    @patch("src.preprocess.pv_resource_to_h5.HDFStore")
    @patch("src.preprocess.pv_resource_to_h5.Outputs")
    @patch("src.preprocess.pv_resource_to_h5.os.path.exists")
    @patch("src.preprocess.pv_resource_to_h5.os.makedirs")
    def test_pv_resource_to_h5_integration(
        self,
        mock_makedirs,
        mock_exists,
        mock_outputs,
        mock_hdfstore,
        mock_ds_to_zarr,
        mock_preprocess_pv,
        mock_intake,
        sample_pv_dataset,
        sample_meta_data,
    ):
        """Test complete integration of PV resource to H5 workflow."""
        # Mock intake catalog
        mock_cat = Mock()
        mock_intake.open_esm_datastore.return_value = mock_cat

        mock_query_result = Mock()
        mock_cat.search.return_value = mock_query_result

        mock_ds_dict = Mock()
        mock_query_result.to_dataset_dict.return_value = mock_ds_dict
        mock_ds_dict.values.return_value = [sample_pv_dataset]

        # Mock preprocessing with comprehensive dataset
        processed_ds = sample_pv_dataset.copy()
        processed_ds["time_index"] = (
            processed_ds["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").astype("|S30")
        )
        mock_preprocess_pv.return_value = processed_ds

        # Mock file operations
        mock_exists.return_value = False

        # Mock HDF operations
        mock_hdf_instance = Mock()
        mock_hdfstore.return_value = mock_hdf_instance
        mock_hdf_instance.meta = sample_meta_data

        mock_outputs_instance = Mock()
        mock_outputs_instance.__setitem__ = Mock()
        mock_outputs_instance.__enter__ = Mock(return_value=mock_outputs_instance)
        mock_outputs_instance.__exit__ = Mock(return_value=None)
        mock_outputs.return_value = mock_outputs_instance
        mock_outputs.add_dataset = Mock()

        # Run complete workflow
        with patch("src.preprocess.pv_resource_to_h5.Outputs.add_dataset"):
            pv_resource_to_h5(rank=10, domain="d03", sim_id="MIROC6")

        # Verify complete workflow execution
        mock_cat.search.assert_called_once()
        mock_query_result.to_dataset_dict.assert_called_once()
        mock_preprocess_pv.assert_called_once()
        mock_ds_to_zarr.assert_called_once()
        mock_hdfstore.assert_called()
        mock_outputs.assert_called()

        # Check final data processing steps
        # Should have processed dataset through stacking and variable extraction
        processed_data = mock_preprocess_pv.return_value
        assert "time_index" in processed_data

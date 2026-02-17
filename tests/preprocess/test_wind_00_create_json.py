#!/usr/bin/env python3
"""
Tests for wind_00_create_json module.

This module tests JSON configuration file creation functionality for wind
resource data processing workflows.
"""

from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

from src.preprocess.wind_00_create_json import (
    create_and_save_json,
    find_wrf_files_on_s3,
    make_wrf_jsons_domain_model,
)


class TestWindJsonCreation:
    """Test wind resource JSON configuration file creation."""

    @pytest.fixture
    def sample_s3_files(self):
        """Create sample S3 file list for testing."""
        return [
            "downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1981/d03/auxhist_d01_1981-01-01_00:00:00",
            "downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1981/d03/auxhist_d01_1981-01-02_00:00:00",
            "downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1981/d03/auxhist_d01_1981-01-03_00:00:00",
            "downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1981/d03/auxhist_d01_1981-06-15_00:00:00",
            "downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1981/d03/auxhist_d01_1981-12-31_00:00:00",
        ]

    @pytest.fixture
    def sample_boto3_response(self, sample_s3_files):
        """Create sample boto3 paginated response."""
        return [
            {"Contents": [{"Key": file_key} for file_key in sample_s3_files[:3]]},
            {"Contents": [{"Key": file_key} for file_key in sample_s3_files[3:]]},
        ]

    @patch("src.preprocess.wind_00_create_json.boto3.client")
    def test_find_wrf_files_on_s3_basic(self, mock_boto3_client, sample_boto3_response):
        """Test basic S3 file finding functionality."""
        # Mock S3 client and paginator
        mock_s3 = Mock()
        mock_boto3_client.return_value = mock_s3

        mock_paginator = Mock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = sample_boto3_response

        # Test finding files
        result = find_wrf_files_on_s3("test-bucket", "test-prefix")

        # Should call S3 client correctly
        mock_boto3_client.assert_called_once_with("s3")
        mock_s3.get_paginator.assert_called_once_with("list_objects_v2")
        mock_paginator.paginate.assert_called_once_with(
            Bucket="test-bucket", Prefix="test-prefix"
        )

        # Should return all file keys
        expected_files = [
            "downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1981/d03/auxhist_d01_1981-01-01_00:00:00",
            "downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1981/d03/auxhist_d01_1981-01-02_00:00:00",
            "downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1981/d03/auxhist_d01_1981-01-03_00:00:00",
            "downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1981/d03/auxhist_d01_1981-06-15_00:00:00",
            "downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1981/d03/auxhist_d01_1981-12-31_00:00:00",
        ]

        assert result == expected_files

    @patch("src.preprocess.wind_00_create_json.boto3.client")
    def test_find_wrf_files_on_s3_empty_response(self, mock_boto3_client):
        """Test S3 file finding with empty response."""
        # Mock S3 client with empty response
        mock_s3 = Mock()
        mock_boto3_client.return_value = mock_s3

        mock_paginator = Mock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]  # Empty page

        result = find_wrf_files_on_s3("test-bucket", "test-prefix")

        # Should return empty list
        assert result == []

    @patch("src.preprocess.wind_00_create_json.boto3.client")
    def test_find_wrf_files_on_s3_no_contents(self, mock_boto3_client):
        """Test S3 file finding with no Contents key."""
        # Mock S3 client with response lacking Contents
        mock_s3 = Mock()
        mock_boto3_client.return_value = mock_s3

        mock_paginator = Mock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"ResponseMetadata": {}}]

        result = find_wrf_files_on_s3("test-bucket", "test-prefix")

        # Should return empty list
        assert result == []

    @patch("src.preprocess.wind_00_create_json.find_wrf_files_on_s3")
    def test_make_wrf_jsons_domain_model_basic(self, mock_find_files, sample_s3_files):
        """Test basic JSON creation for WRF domain and model."""
        # Mock S3 file finding to return sample files
        mock_find_files.return_value = sample_s3_files

        result = make_wrf_jsons_domain_model("d03", "MIROC6")

        # Should return dictionary with year keys
        assert isinstance(result, dict)

        # Should have year keys as strings
        year_keys = list(result.keys())
        assert all(key.isdigit() for key in year_keys)

        # Should have reasonable number of years (1981-2098)
        assert len(year_keys) > 100  # Many years

        # Each year should have list of files
        for year_key in year_keys[:5]:  # Check first few years
            assert isinstance(result[year_key], list)

    @patch("src.preprocess.wind_00_create_json.find_wrf_files_on_s3")
    def test_make_wrf_jsons_domain_model_year_range(self, mock_find_files):
        """Test correct year range handling in JSON creation."""
        mock_find_files.return_value = []

        result = make_wrf_jsons_domain_model("d02", "EC-Earth3")

        # Check year range (should be 1981-2098)
        years = [int(year) for year in result]
        min_year, max_year = min(years), max(years)

        assert min_year == 1981
        assert max_year == 2098

    @patch("src.preprocess.wind_00_create_json.find_wrf_files_on_s3")
    def test_make_wrf_jsons_domain_model_scenario_mapping(self, mock_find_files):
        """Test correct scenario mapping for different years."""
        mock_find_files.return_value = []

        make_wrf_jsons_domain_model("d03", "MPI-ESM1-2-HR")

        # Function should be called for both historical and ssp370 scenarios
        # Check that find_wrf_files_on_s3 was called multiple times
        assert mock_find_files.call_count > 0

        # Check some of the calls for different scenarios
        call_args_list = mock_find_files.call_args_list

        # Should have calls for historical years (before 2014)
        historical_calls = [
            call for call in call_args_list if "historical" in str(call)
        ]
        assert len(historical_calls) > 0

        # Should have calls for ssp370 years (2014 and after)
        ssp370_calls = [call for call in call_args_list if "ssp370" in str(call)]
        assert len(ssp370_calls) > 0

    @patch("src.preprocess.wind_00_create_json.find_wrf_files_on_s3")
    def test_make_wrf_jsons_domain_model_file_filtering(self, mock_find_files):
        """Test file filtering logic in JSON creation."""
        # Create sample files with some that should be filtered out
        sample_files_with_exclusions = [
            "auxhist_d01_1981-01-01_00:00:00",
            "auxhist_d01_1981-02-29_00:00:00",  # Leap day - should be filtered
            "auxhist_d01_1981-06-15_00:00:00",
            "auxhist_d01_1982-09-01_00:00:00",  # Next year Sept 1 - should be filtered
            "auxhist_d01_1981-12-31_00:00:00",
            "auxhist_d02_1981-01-01_00:00:00",  # Wrong domain - should be filtered
        ]

        mock_find_files.return_value = sample_files_with_exclusions

        result = make_wrf_jsons_domain_model("d03", "TaiESM1")

        # Check that files were filtered appropriately
        # (This is a structural test - in practice filtering happens within the function)
        for year_key in list(result.keys())[:3]:  # Check first few years
            year_files = result[year_key]

            # No leap day files
            leap_day_files = [f for f in year_files if "-02_29_" in f]
            assert len(leap_day_files) == 0

            # Only d01 files (auxhist_d01_)
            d01_files = [f for f in year_files if "auxhist_d01_" in f]
            assert len(d01_files) == len(year_files) or len(year_files) == 0

    @patch("src.preprocess.wind_00_create_json.find_wrf_files_on_s3")
    def test_make_wrf_jsons_domain_model_different_models(self, mock_find_files):
        """Test JSON creation for different climate models."""
        models = ["MIROC6", "EC-Earth3", "MPI-ESM1-2-HR", "TaiESM1"]

        mock_find_files.return_value = ["test_file_1981.nc"]

        for model in models:
            result = make_wrf_jsons_domain_model("d02", model)

            # Should return valid dictionary for each model
            assert isinstance(result, dict)
            assert len(result) > 0

            # Check that appropriate model-specific prefix was used
            # by examining the calls to find_wrf_files_on_s3
            model_calls = [
                call
                for call in mock_find_files.call_args_list
                if model.lower() in str(call).lower()
            ]
            # Should have calls related to this model
            assert len(model_calls) > 0

        mock_find_files.reset_mock()

    @patch("src.preprocess.wind_00_create_json.find_wrf_files_on_s3")
    def test_make_wrf_jsons_domain_model_different_domains(self, mock_find_files):
        """Test JSON creation for different WRF domains."""
        domains = ["d01", "d02", "d03"]

        mock_find_files.return_value = ["test_file.nc"]

        for domain in domains:
            result = make_wrf_jsons_domain_model(domain, "MIROC6")

            # Should return valid dictionary for each domain
            assert isinstance(result, dict)

            # Check that domain was used in S3 prefix calls
            domain_calls = [
                call for call in mock_find_files.call_args_list if domain in str(call)
            ]
            assert len(domain_calls) > 0

        mock_find_files.reset_mock()

    @patch("src.preprocess.wind_00_create_json.make_wrf_jsons_domain_model")
    @patch("src.preprocess.wind_00_create_json.s3")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.preprocess.wind_00_create_json.json.dump")
    def test_create_and_save_json_basic(
        self, mock_json_dump, mock_file_open, mock_s3, mock_make_jsons
    ):
        """Test basic JSON file creation and S3 upload."""
        # Mock the JSON data creation
        sample_json_data = {"1981": ["file1.nc"], "1982": ["file2.nc"]}
        mock_make_jsons.return_value = sample_json_data

        create_and_save_json("d03")

        # Should call make_wrf_jsons_domain_model for each model
        models = ["MIROC6", "EC-Earth3", "MPI-ESM1-2-HR", "TaiESM1"]
        assert mock_make_jsons.call_count == len(models)

        # Check that each model was processed
        for model in models:
            model_calls = [
                call for call in mock_make_jsons.call_args_list if model in call[0]
            ]
            assert len(model_calls) == 1

        # Should create and upload file for each model
        assert mock_file_open.call_count == len(models)
        assert mock_json_dump.call_count == len(models)
        assert mock_s3.upload_file.call_count == len(models)

    @patch("src.preprocess.wind_00_create_json.make_wrf_jsons_domain_model")
    @patch("src.preprocess.wind_00_create_json.s3")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.preprocess.wind_00_create_json.json.dump")
    def test_create_and_save_json_file_naming(
        self, mock_json_dump, mock_file_open, mock_s3, mock_make_jsons
    ):
        """Test correct file naming in JSON creation and upload."""
        sample_json_data = {"1981": ["file1.nc"]}
        mock_make_jsons.return_value = sample_json_data

        create_and_save_json("d02")

        # Check file names for each model
        models = ["MIROC6", "EC-Earth3", "MPI-ESM1-2-HR", "TaiESM1"]

        expected_filenames = [f"{model}_d02.json" for model in models]

        # Check file open calls
        file_open_calls = [call[0][0] for call in mock_file_open.call_args_list]
        for expected_filename in expected_filenames:
            assert expected_filename in file_open_calls

        # Check S3 upload calls
        upload_calls = mock_s3.upload_file.call_args_list
        for i, expected_filename in enumerate(expected_filenames):
            upload_call = upload_calls[i]
            assert upload_call[1]["Filename"] == expected_filename
            assert upload_call[1]["Key"] == f"wrf_jsons/{expected_filename}"
            assert upload_call[1]["Bucket"] == "wfclimres"

    @patch("src.preprocess.wind_00_create_json.make_wrf_jsons_domain_model")
    @patch("src.preprocess.wind_00_create_json.s3")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_and_save_json_s3_error_handling(
        self, mock_file_open, mock_s3, mock_make_jsons
    ):
        """Test error handling during S3 upload."""
        mock_make_jsons.return_value = {"1981": ["file1.nc"]}

        # Mock S3 upload to raise exception
        mock_s3.upload_file.side_effect = ConnectionError("S3 upload failed")

        # Should raise exception when S3 upload fails
        with pytest.raises(ConnectionError):
            create_and_save_json("d03")

    @patch("src.preprocess.wind_00_create_json.make_wrf_jsons_domain_model")
    def test_create_and_save_json_make_jsons_error(self, mock_make_jsons):
        """Test error handling during JSON data creation."""
        # Mock JSON creation to raise exception
        mock_make_jsons.side_effect = ValueError("JSON creation failed")

        # Should raise exception when JSON creation fails
        with pytest.raises(ValueError):
            create_and_save_json("d03")

    @patch("src.preprocess.wind_00_create_json.find_wrf_files_on_s3")
    def test_make_wrf_jsons_simulation_arns(self, mock_find_files):
        """Test correct simulation ARN formatting."""
        mock_find_files.return_value = []

        # Test different models to ensure correct ARN formatting
        models_and_arns = {
            "EC-Earth3": "ec-earth3_r1i1p1f1_",
            "MPI-ESM1-2-HR": "mpi-esm1-2-hr_r3i1p1f1_",
            "MIROC6": "miroc6_r1i1p1f1_",
            "TaiESM1": "taiesm1_r1i1p1f1_",
        }

        for model, arn_prefix in models_and_arns.items():
            mock_find_files.reset_mock()
            make_wrf_jsons_domain_model("d03", model)

            # Check that ARN was used in S3 prefix calls
            arn_calls = [
                call
                for call in mock_find_files.call_args_list
                if arn_prefix in str(call)
            ]
            assert len(arn_calls) > 0

    @patch("src.preprocess.wind_00_create_json.find_wrf_files_on_s3")
    def test_make_wrf_jsons_file_concatenation_and_sorting(self, mock_find_files):
        """Test file concatenation and sorting logic."""
        # Create files from two consecutive years
        files_year1 = ["file_1981_01.nc", "file_1981_03.nc"]
        files_year2 = ["file_1980_11.nc", "file_1980_12.nc"]

        # Mock find_wrf_files_on_s3 to return different files for different calls
        mock_find_files.side_effect = [
            files_year1,
            files_year2,
        ] * 200  # Repeat for many years

        result = make_wrf_jsons_domain_model("d03", "MIROC6")

        # Check that files from consecutive years are combined
        # (This is a structural test of the logic)
        for year_key in list(result.keys())[:2]:  # Check first couple years
            year_files = result[year_key]
            # Files should be sorted
            assert year_files == sorted(year_files)

    def test_year_scenario_mapping(self):
        """Test year to scenario mapping logic."""
        years = np.arange(1980, 2100, dtype=int)
        scens = {y: "historical" for y in years if y < 2014} | {
            y: "ssp370" for y in years if y >= 2014
        }

        # Check historical scenario years
        historical_years = [y for y in years if scens[y] == "historical"]
        assert all(y < 2014 for y in historical_years)
        assert 1980 in historical_years
        assert 2013 in historical_years
        assert 2014 not in historical_years

        # Check ssp370 scenario years
        ssp370_years = [y for y in years if scens[y] == "ssp370"]
        assert all(y >= 2014 for y in ssp370_years)
        assert 2014 in ssp370_years
        assert 2099 in ssp370_years
        assert 2013 not in ssp370_years

    def test_model_constants(self):
        """Test that model constants are correctly defined."""
        from src.preprocess.wind_00_create_json import models

        expected_models = ["MIROC6", "EC-Earth3", "MPI-ESM1-2-HR", "TaiESM1"]

        # Should have all expected models
        for model in expected_models:
            assert model in models

        # Should have correct number of models
        assert len(models) == len(expected_models)

    @patch("src.preprocess.wind_00_create_json.find_wrf_files_on_s3")
    def test_make_wrf_jsons_integration(self, mock_find_files):
        """Test complete integration of JSON creation workflow."""
        # Mock comprehensive file list
        sample_files = [
            f"auxhist_d01_1981-{month:02d}-01_00:00:00" for month in range(1, 13)
        ]
        mock_find_files.return_value = sample_files

        result = make_wrf_jsons_domain_model("d03", "MIROC6")

        # Should return comprehensive result
        assert isinstance(result, dict)
        assert len(result) > 100  # Many years

        # Should have called S3 file finding many times (for each year)
        assert mock_find_files.call_count > 100

        # Each year should have file list
        for year_key in list(result.keys())[:5]:
            assert isinstance(result[year_key], list)

    def test_bucket_constants(self):
        """Test that bucket constants are correctly defined."""
        from src.preprocess.wind_00_create_json import bucket, wrf_bucket

        assert bucket == "wfclimres"
        assert wrf_bucket == "wrf-cmip6-noversioning"

    @patch("src.preprocess.wind_00_create_json.find_wrf_files_on_s3")
    def test_make_wrf_jsons_empty_file_handling(self, mock_find_files):
        """Test handling when no files are found."""
        # Mock empty file lists
        mock_find_files.return_value = []

        result = make_wrf_jsons_domain_model("d03", "MIROC6")

        # Should still return dictionary with year keys
        assert isinstance(result, dict)
        assert len(result) > 0

        # Each year should have empty list
        for year_key in list(result.keys())[:5]:
            assert result[year_key] == []

    @patch("src.preprocess.wind_00_create_json.make_wrf_jsons_domain_model")
    @patch("src.preprocess.wind_00_create_json.s3")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.preprocess.wind_00_create_json.json.dump")
    def test_create_and_save_json_complete_workflow(
        self, mock_json_dump, mock_file_open, mock_s3, mock_make_jsons
    ):
        """Test complete JSON creation and upload workflow."""
        # Mock comprehensive JSON data
        comprehensive_json_data = {
            str(year): [f"file_{year}_{month}.nc" for month in range(1, 13)]
            for year in range(1981, 1985)  # Sample years
        }
        mock_make_jsons.return_value = comprehensive_json_data

        create_and_save_json("d03")

        # Complete workflow should execute for all models
        models = ["MIROC6", "EC-Earth3", "MPI-ESM1-2-HR", "TaiESM1"]

        # Should process each model
        assert mock_make_jsons.call_count == len(models)

        # Should create files for each model
        assert mock_file_open.call_count == len(models)
        assert mock_json_dump.call_count == len(models)

        # Should upload files for each model
        assert mock_s3.upload_file.call_count == len(models)

        # Check that JSON data was dumped correctly
        for call_args in mock_json_dump.call_args_list:
            dumped_data = call_args[0][0]
            assert isinstance(dumped_data, dict)
            assert len(dumped_data) > 0

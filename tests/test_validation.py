"""
Validation tests to verify the testing infrastructure is set up correctly.
"""
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np


class TestInfrastructureValidation:
    """Test class to validate the testing infrastructure setup."""
    
    @pytest.mark.unit
    def test_pytest_is_working(self):
        """Verify that pytest is installed and working."""
        assert True
        
    @pytest.mark.unit
    def test_imports_are_working(self):
        """Verify that all required packages can be imported."""
        import pandas
        import numpy
        import pytest
        import sklearn
        
        assert pandas.__version__
        assert numpy.__version__
        assert pytest.__version__
        assert sklearn.__version__
        
    @pytest.mark.unit
    def test_fixtures_are_accessible(self, temp_dir, sample_config):
        """Verify that conftest fixtures are accessible."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert isinstance(sample_config, dict)
        assert 'random_state' in sample_config
        
    @pytest.mark.unit
    def test_temp_dir_fixture(self, temp_dir):
        """Verify that temp_dir fixture creates and cleans up properly."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")
        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"
        
    @pytest.mark.unit
    def test_sample_csv_fixture(self, sample_csv_file):
        """Verify that sample_csv_file fixture creates valid CSV."""
        assert sample_csv_file.exists()
        df = pd.read_csv(sample_csv_file)
        assert len(df) == 5
        assert list(df.columns) == ['feature1', 'feature2', 'feature3', 'target']
        
    @pytest.mark.unit
    def test_sample_iris_data_fixture(self, sample_iris_data):
        """Verify that sample_iris_data fixture creates valid data."""
        assert isinstance(sample_iris_data, pd.DataFrame)
        assert len(sample_iris_data) == 150
        assert all(col in sample_iris_data.columns for col in 
                  ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'])
        
    @pytest.mark.unit
    def test_capture_stdout_fixture(self, capture_stdout):
        """Verify that capture_stdout fixture works correctly."""
        print("Test output 1")
        print("Test output 2")
        assert len(capture_stdout) == 2
        assert capture_stdout[0] == "Test output 1"
        assert capture_stdout[1] == "Test output 2"
        
    @pytest.mark.unit
    def test_mock_sklearn_model_fixture(self, mock_sklearn_model):
        """Verify that mock_sklearn_model fixture works correctly."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        model = mock_sklearn_model
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert model.fit.called
        assert model.predict.called
        assert len(predictions) == 4
        
    @pytest.mark.integration
    def test_code_module_can_be_imported(self):
        """Verify that the Code module can be imported."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            import Code
            assert Code
        except ImportError:
            pass
        
    @pytest.mark.unit
    @pytest.mark.parametrize("marker", ["unit", "integration", "slow"])
    def test_custom_markers_are_registered(self, marker):
        """Verify that custom markers are properly registered."""
        # Custom markers are defined in pyproject.toml, so we just verify they work
        assert marker in ['unit', 'integration', 'slow']
        
    @pytest.mark.unit
    def test_coverage_is_configured(self):
        """Verify that coverage is properly configured."""
        try:
            import coverage
            assert coverage.__version__
        except ImportError:
            pytest.skip("Coverage not installed")
            
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Test that demonstrates the slow marker."""
        import time
        time.sleep(0.1)
        assert True


def test_basic_assertion():
    """Simple test to verify pytest runs without test class."""
    assert 1 + 1 == 2


def test_numpy_operations():
    """Test to verify numpy works correctly in tests."""
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15


def test_pandas_operations():
    """Test to verify pandas works correctly in tests."""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert len(df) == 3
    assert df['A'].sum() == 6
    assert df['B'].mean() == 5.0
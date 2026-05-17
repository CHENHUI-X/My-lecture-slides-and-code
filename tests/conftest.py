"""
Shared pytest fixtures and configuration for all tests.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Any
from unittest.mock import Mock

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_csv_file(temp_dir: Path) -> Path:
    """Create a sample CSV file for testing."""
    csv_path = temp_dir / "test_data.csv"
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2.5, 3.5, 4.5, 5.5, 6.5],
        'feature3': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })
    data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_excel_file(temp_dir: Path) -> Path:
    """Create a sample Excel file for testing."""
    excel_path = temp_dir / "test_data.xlsx"
    data = pd.DataFrame({
        'SepalLength': [5.1, 4.9, 4.7, 4.6, 5.0],
        'SepalWidth': [3.5, 3.0, 3.2, 3.1, 3.6],
        'PetalLength': [1.4, 1.4, 1.3, 1.5, 1.4],
        'PetalWidth': [0.2, 0.2, 0.2, 0.2, 0.2],
        'Species': [0, 0, 0, 0, 0]
    })
    data.to_excel(excel_path, index=False)
    return excel_path


@pytest.fixture
def sample_iris_data() -> pd.DataFrame:
    """Create sample iris dataset for testing."""
    np.random.seed(42)
    n_samples = 150
    
    data = pd.DataFrame({
        'SepalLength': np.random.normal(5.8, 0.8, n_samples),
        'SepalWidth': np.random.normal(3.0, 0.4, n_samples),
        'PetalLength': np.random.normal(3.7, 1.7, n_samples),
        'PetalWidth': np.random.normal(1.2, 0.7, n_samples),
        'Species': np.random.choice([0, 1, 2], n_samples)
    })
    return data


@pytest.fixture
def sample_boston_data() -> tuple[np.ndarray, np.ndarray]:
    """Create sample Boston housing dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 13
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 10 + 20
    
    return X, y


@pytest.fixture
def mock_sklearn_model() -> Mock:
    """Create a mock scikit-learn model."""
    model = Mock()
    model.fit = Mock(return_value=model)
    model.predict = Mock(return_value=np.array([0, 1, 0, 1]))
    model.score = Mock(return_value=0.85)
    return model


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Create a sample configuration dictionary."""
    return {
        'random_state': 42,
        'test_size': 0.2,
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
    }


@pytest.fixture
def mock_data_path(monkeypatch, temp_dir: Path) -> Path:
    """Mock data paths to use temporary directory."""
    data_path = temp_dir / "data"
    data_path.mkdir(exist_ok=True)
    monkeypatch.setattr('os.getcwd', lambda: str(temp_dir))
    return data_path


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
    

@pytest.fixture
def capture_stdout(monkeypatch) -> list[str]:
    """Capture stdout output during tests."""
    captured = []
    
    def mock_print(*args, **kwargs):
        captured.append(' '.join(str(arg) for arg in args))
    
    monkeypatch.setattr('builtins.print', mock_print)
    return captured


@pytest.fixture
def mock_file_operations(monkeypatch) -> dict[str, list]:
    """Mock file read/write operations and track them."""
    operations = {
        'reads': [],
        'writes': []
    }
    
    original_read_csv = pd.read_csv
    original_read_excel = pd.read_excel
    
    def mock_read_csv(filepath_or_buffer, **kwargs):
        operations['reads'].append(('csv', str(filepath_or_buffer)))
        return original_read_csv(filepath_or_buffer, **kwargs)
    
    def mock_read_excel(io, **kwargs):
        operations['reads'].append(('excel', str(io)))
        return original_read_excel(io, **kwargs)
    
    monkeypatch.setattr('pandas.read_csv', mock_read_csv)
    monkeypatch.setattr('pandas.read_excel', mock_read_excel)
    
    return operations


@pytest.mark.unit
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
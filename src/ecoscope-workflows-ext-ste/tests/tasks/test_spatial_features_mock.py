from wt_task.testing import _find_example_return_path, _load_example_return

from ecoscope_workflows_ext_ste.tasks.spatial_operations import get_spatial_features


def test_get_spatial_features_is_mockable_io():
    """Mock-IO test runs replace get_spatial_features with its example return."""
    path = _find_example_return_path("ecoscope_workflows_ext_ste.tasks.spatial_operations", "get_spatial_features")
    assert _load_example_return(path) == []
    assert get_spatial_features([], layers=None) == []

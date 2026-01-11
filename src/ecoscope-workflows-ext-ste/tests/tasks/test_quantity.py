from ecoscope_workflows_ext_ste.tasks._quantity import to_quantity
from ecoscope_workflows_core.tasks.transformation._unit import Quantity


def test_convert_integer_with_unit():
    """Test converting an integer value to Quantity."""
    result = to_quantity(100, "km²")

    assert isinstance(result, Quantity)
    assert result.value == 100
    assert result.unit == "km²"


def test_convert_float_with_unit():
    """Test converting a float value to Quantity."""
    result = to_quantity(25.75, "km/h")

    assert isinstance(result, Quantity)
    assert result.value == 25.75
    assert result.unit == "km/h"


def test_convert_string_with_unit():
    """Test converting a string value to Quantity."""
    result = to_quantity("Male", "gender")

    assert isinstance(result, Quantity)
    assert result.value == "Male"
    assert result.unit == "gender"

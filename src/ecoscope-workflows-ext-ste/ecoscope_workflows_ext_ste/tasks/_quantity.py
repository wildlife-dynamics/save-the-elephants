from typing import Any
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.transformation._unit import Quantity


@task
def to_quantity(value: Any, unit: str) -> Quantity:
    """Converts a value and unit to a Quantity object."""
    return Quantity(value=value, unit=unit)

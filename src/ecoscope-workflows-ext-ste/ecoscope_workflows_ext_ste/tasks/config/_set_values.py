from pydantic import Field
from wt_registry import register
from typing import Annotated, Union


@register()
def set_numerical_var(
    var: Annotated[Union[int, float], Field(title="")],
) -> Union[int, float]:
    return var

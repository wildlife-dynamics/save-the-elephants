from typing import Annotated, Any

from pydantic import Field
from wt_registry import register
from wt_task.skip import SKIP_SENTINEL, SkipSentinel
from .._log import count, log

# One slot per group: a (key, value) pair, where the value is a SkipSentinel when
# the step that produced it was skipped for that group. Bare None slots and
# pairs with a None key are tolerated and ignored.
KeyedIterableWithSkips = list[tuple[Any, Any] | None]


def _label(key: Any) -> str:
    """'subject_name=Scout 3' from a group key like (('subject_name', '=', 'Scout 3'),)."""
    try:
        return ", ".join(f"{column}={value}" for column, _, value in key)
    except (TypeError, ValueError):
        return str(key)


def _combine(
    iterables: list[KeyedIterableWithSkips],
) -> tuple[list[Any], list[dict[Any, Any]]]:
    """Group keys in first-seen order, and for each iterable its real values by key."""
    values_by_key: list[dict[Any, Any]] = []
    order: list[Any] = []
    for iterable in iterables:
        values: dict[Any, Any] = {}
        for slot in iterable:
            if slot is None:
                continue
            key, value = slot
            if key is None:
                continue
            if key not in order:
                order.append(key)
            if value is None or isinstance(value, SkipSentinel):
                continue
            values[key] = value
        values_by_key.append(values)
    return order, values_by_key


@register()
def groupbykey_complete(
    iterables: Annotated[
        list[KeyedIterableWithSkips],
        Field(description="List of keyed iterables, which may hold skipped values."),
    ],
) -> Annotated[
    list[tuple[Any, list[Any]]] | SkipSentinel,
    Field(description="The groups that have a value in every iterable, with their values combined in order."),
]:
    """Combine keyed iterables, keeping only the groups that are complete.

    Works like ``groupbykey``, but a group is kept only when every iterable has a
    real value for it. A group whose value was skipped (or is missing) in any
    iterable is left out, and the other groups go through untouched.

    Use it instead of ``groupbykey`` with the ``any_keyed_iterables_are_skips``
    skip condition, which skips the step for every group as soon as one group has
    a skipped value. With this task one subject without data for a map no longer
    removes that map for all the other subjects.

    Returns a SkipSentinel when no group is complete, so the steps after it are
    skipped as a whole.
    """
    order, values_by_key = _combine(iterables)
    complete = [key for key in order if all(key in values for values in values_by_key)]
    if len(complete) < len(order):
        left_out = [key for key in order if key not in complete]
        log(
            "setup",
            f"combine groups: {len(left_out)} of {count(len(order), 'group')} left out, "
            f"a step before this was skipped for them: {[_label(key) for key in left_out]}",
        )
    if not complete:
        return SKIP_SENTINEL
    return [(key, [values[key] for values in values_by_key]) for key in complete]


@register()
def groupbykey_keep_skips(
    iterables: Annotated[
        list[KeyedIterableWithSkips],
        Field(description="List of keyed iterables, which may hold skipped values."),
    ],
) -> Annotated[
    list[tuple[Any, list[Any]]] | SkipSentinel,
    Field(description="Every group, with one value per iterable and a skip marker for each missing value."),
]:
    """Combine keyed iterables, keeping every group even when some values were skipped.

    Each group gets one value per iterable, in order. A value that was skipped
    (or is missing) becomes a SkipSentinel, so the values still line up with the
    next step's arguments.

    For steps that can work with part of their inputs, such as a report page
    that shows N/A for what is missing. That step must accept skipped values and
    must not use the ``any_dependency_skipped`` condition.

    Returns a SkipSentinel when there are no groups at all.
    """
    order, values_by_key = _combine(iterables)
    if not order:
        return SKIP_SENTINEL
    partial = [key for key in order if not all(key in values for values in values_by_key)]
    if partial:
        log(
            "setup",
            f"combine groups: {len(partial)} of {count(len(order), 'group')} kept with missing values: "
            f"{[_label(key) for key in partial]}",
        )
    return [(key, [values.get(key, SKIP_SENTINEL) for values in values_by_key]) for key in order]

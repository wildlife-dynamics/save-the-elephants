from ecoscope_workflows_core.decorators import task
from typing import Sequence, TypeVar, Tuple, Union, List, Dict, Any
from collections import defaultdict
from ecoscope_workflows_core.skip import SkipSentinel, SKIP_SENTINEL

GroupKey = Tuple  # normalized group key as a tuple
V = Union[tuple, list, str, int, float, object]

K = TypeVar("K")  # no Hashable bound
L = TypeVar("L")
R = TypeVar("R")
T = TypeVar("T")
T = TypeVar("T")
U = TypeVar("U")


JsonPrimitive = Union[str, int, float, bool, None]


@task
def flatten_tuple(nested: tuple) -> Tuple[JsonPrimitive, ...]:
    """
    Recursively flatten a (possibly deeply nested) tuple into a flat tuple
    of JSON-safe primitives. Filters out SkipSentinel values.

    Note:
    - We annotate the input as `tuple` (built-in) to avoid Pydantic trying to
      resolve a recursive type annotation at import time.
    - At runtime we enforce that leaf values are JSON-safe primitives and
      raise TypeError for unsupported leaf types (avoids silently accepting objects).
    - SkipSentinel values are silently filtered out.
    """
    # If input is not a tuple then it's an invalid call for this function.
    if not isinstance(nested, tuple):
        raise TypeError("flatten_tuple expects a tuple (possibly nested).")

    flat_list: list[JsonPrimitive] = []

    for item in nested:
        # Skip SkipSentinel values using identity check
        if item is SKIP_SENTINEL or isinstance(item, SkipSentinel):
            continue

        if isinstance(item, tuple):
            flat_list.extend(flatten_tuple(item))
            continue

        # Accept JSON-safe primitives
        if isinstance(item, (str, int, float, bool)) or item is None:
            flat_list.append(item)
            continue

        # Reject everything else explicitly
        raise TypeError(
            f"Unsupported leaf type in nested tuple: {type(item)!r}. "
            "Allowed leaf types: str, int, float, bool, None."
        )

    return tuple(flat_list)


@task
def zip_grouped_by_key(
    left: Sequence[tuple[K, L]],
    right: Sequence[tuple[K, R]],
) -> list[tuple[K, tuple[L, R]]]:
    """
    Zips two grouped sequences by key.

    - Preserves order from `left`
    - Keeps only keys present in both
    - Returns [(key, (left_value, right_value)), ...]
    """
    right_dict: dict[K, R] = dict(right)  # runtime still requires hashable keys
    out: list[tuple[K, tuple[L, R]]] = []
    seen: set[K] = set()

    for k, lv in left:
        if k in seen:
            raise ValueError(f"Duplicate key in 'left': {k!r}")
        seen.add(k)
        if k in right_dict:
            out.append((k, (lv, right_dict[k])))

    return out


@task
def zip_lists(left: List[T], right: List[U]) -> List[Tuple[T, U]]:
    """
    Zip two lists together into a list of tuples.

    Args:
        left: First list
        right: Second list

    Returns:
        List of tuples combining elements from both lists

    Example:
        >>> zip_lists(['MNP', 'WDH East'], [data1, data2])
        [('MNP', data1), ('WDH East', data2)]
    """
    if len(left) != len(right):
        raise ValueError(f"Lists must have the same length. Got {len(left)} and {len(right)}")

    return list(zip(left, right))


@task
def zip_groupbykey(sequences: List[Any]) -> List[Tuple[Any, Tuple[Any, ...]]]:
    """
    Improved version that returns [(key, (val1, val2, ...)), ...]
    - Handles mixed input types safely (grouped tuples, dict, broadcast single value)
    - Compatible with the stricter zip_grouped_by_key style
    """
    print(f"DEBUG: Received {len(sequences)} sequences")

    if not sequences:
        return []

    # Normalize each sequence
    normalized: List[Dict[Any, Any]] = []
    broadcast_values: List[Any] = []

    for seq in sequences:
        if isinstance(seq, dict):
            normalized.append({k: v if isinstance(v, list) else [v] for k, v in seq.items()})
            broadcast_values.append(None)
        elif isinstance(seq, list) and seq and isinstance(seq[0], tuple) and len(seq[0]) == 2:
            d = defaultdict(list)
            for k, v in seq:
                d[k].append(v)
            normalized.append(d)
            broadcast_values.append(None)
        else:
            # Single value → broadcast to all keys
            broadcast_values.append(seq)
            normalized.append(None)

    # Find the first grouped sequence to use as key source
    key_source = None
    for i, norm in enumerate(normalized):
        if norm is not None:
            key_source = norm
            break

    if key_source is None:
        # All broadcast → return single entry with a dummy or None key if needed
        # Or decide: maybe raise/error, or return [(None, tuple(broadcast_values))]
        return [(None, tuple(broadcast_values))]

    # Ordered keys from the first grouped sequence
    common_keys_in_order = list(key_source.keys())
    print(f"DEBUG: Common keys: {common_keys_in_order}")

    # Build result: include the key in each tuple
    result: List[Tuple[Any, Tuple[Any, ...]]] = []
    for k in common_keys_in_order:
        values: List[Any] = []
        for i, norm in enumerate(normalized):
            if norm is not None:
                v_list = norm[k]
                # Unwrap singleton lists
                # values.append(v_list[0] if len(v_list) == 1 else v_list)
                if not v_list:
                    # No value for this key → semantic None
                    values.append(None)
                elif len(v_list) == 1:
                    values.append(v_list[0])
                else:
                    values.append(v_list)

            else:
                values.append(broadcast_values[i])  # broadcast value
        result.append((k, tuple(values)))

    print(f"DEBUG: Returning {len(result)} tuples with keys")
    return result

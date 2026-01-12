from ecoscope_workflows_core.decorators import task
from typing import Tuple, List, Dict, Any
from collections import defaultdict


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

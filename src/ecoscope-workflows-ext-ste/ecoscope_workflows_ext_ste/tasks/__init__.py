# write test functions for these functions
from ._ste_utils import (
    get_duration,
    label_quarter_status,
    retrieve_feature_gdf,
    create_seasonal_labels,  # upstream
    generate_ecograph_raster,
    assign_quarter_status_colors,
    modify_quarter_status_colors,
    calculate_seasonal_home_range,
    build_mapbook_report_template,
    dataframe_column_first_unique_str,
    create_context_page,
    create_mapbook_context,
    merge_docx_files,
    get_split_group_names,
    get_split_group_column,
    create_report_context_from_tuple,
)
from ._example import add_one_thousand
from ._downloader import get_file_path, fetch_and_persist_file
from ._zip import zip_grouped_by_key, flatten_tuple, zip_lists
from ._tabular import split_gdf_by_column, generate_mcp_gdf, round_off_values

from ._custom_ecomap import (
    create_view_state_from_gdf,
    get_gdf_geom_type,
    annotate_gdf_dict_with_geom_type,
    create_layer_from_gdf,
    create_layers_from_gdf_dict,
    combine_map_layers,
    make_text_layer,
)

__all__ = [
    # _custom_ecomap
    "get_gdf_geom_type",
    "annotate_gdf_dict_with_geom_type",
    "create_layers_from_gdf_dict",
    "create_view_state_from_gdf",
    "create_layer_from_gdf",
    "combine_map_layers",
    "make_text_layer",
    # _ste_utils
    "create_report_context_from_tuple",
    "get_split_group_column",
    "get_split_group_names",
    "get_duration",
    "label_quarter_status",
    "retrieve_feature_gdf",
    "create_seasonal_labels",
    "generate_ecograph_raster",
    "modify_quarter_status_colors",
    "assign_quarter_status_colors",
    "calculate_seasonal_home_range",
    "build_mapbook_report_template",
    "dataframe_column_first_unique_str",
    "create_context_page",
    "create_mapbook_context",
    "merge_docx_files",
    "round_off_values",
    # _downloader
    "fetch_and_persist_file",
    "get_file_path",
    # _tabular
    "split_gdf_by_column",
    "generate_mcp_gdf",
    # _zip
    "zip_grouped_by_key",
    "flatten_tuple",
    "zip_lists",
    # _example
    "add_one_thousand",
]

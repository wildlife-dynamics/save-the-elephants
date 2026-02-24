from ._downloader import fetch_and_persist_file, get_file_path
from ._ecograph import generate_ecograph_raster, retrieve_feature_gdf
from ._path_utils import get_local_geo_path
from ._seasons import create_seasonal_labels, calculate_seasonal_home_range, custom_determine_season_windows
from ._tabular import (
    split_gdf_by_column,
    generate_mcp_gdf,
    round_off_values,
    dataframe_column_first_unique_str,
    get_duration,
    filter_df_cols,
    create_column,
    convert_to_str,
)
from ._zip import zip_groupbykey
from ._time_comparison import determine_previous_period
from ._mapdeck_utils import (
    view_state_deck_gdf,
    combine_deckgl_map_layers,
    order_deckgl_layers_by_type,
    get_gdf_geom_type,
    annotate_gdf_dict_with_geom_type,
    create_custom_text_layer,
    create_deckgl_layer_from_gdf,
    create_deckgl_layers_from_gdf_dict,
    custom_view_state_from_gdf,
    get_image_zoom_value,
    envelope_gdf,
    generate_protected_column,
)
from ._groupers import (
    get_split_group_column,
    get_split_group_values,
    get_split_group_names,
    extract_index_names,
    set_custom_groupers,
    get_split_group_value,
)

from ._merge import merge_multiple_df
from ._status import modify_status_colors, assign_season_colors
from ._mapbook_context import (
    create_context_page,
    create_mapbook_ctx_cover,
    create_mapbook_grouper_ctx,
    create_grouper_page,
    merge_mapbook_files,
)

from ._preprocess import custom_trajectory_segment_filter
from ._filter import filter_groups_by_value_criteria, filter_df_values
from ._aerial_lines import validate_polygon_geometry, draw_survey_lines, transform_gdf_crs
from ._custom_html_png import adjust_map_zoom_and_screenshot

from ._hex_rgba import convert_hex_to_rgba
from ._day_night import get_grid_night_fixes, get_day_night_dominance
from ._plot import plot_fix_protection_status
from ._general_context import general_template_context


__all__ = [
    "general_template_context",
    "get_split_group_value",
    "plot_fix_protection_status",
    "get_grid_night_fixes",
    "get_day_night_dominance",
    "generate_protected_column",
    "filter_df_values",
    "convert_hex_to_rgba",
    "envelope_gdf",
    "get_image_zoom_value",
    "adjust_map_zoom_and_screenshot",
    "fetch_and_persist_file",
    "get_file_path",
    "generate_ecograph_raster",
    "retrieve_feature_gdf",
    "get_local_geo_path",
    "create_seasonal_labels",
    "calculate_seasonal_home_range",
    "split_gdf_by_column",
    "generate_mcp_gdf",
    "round_off_values",
    "dataframe_column_first_unique_str",
    "get_duration",
    "filter_df_cols",
    "create_column",
    "convert_to_str",
    "zip_groupbykey",
    "determine_previous_period",
    "view_state_deck_gdf",
    "combine_deckgl_map_layers",
    "order_deckgl_layers_by_type",
    "get_gdf_geom_type",
    "annotate_gdf_dict_with_geom_type",
    "create_custom_text_layer",
    "create_deckgl_layer_from_gdf",
    "create_deckgl_layers_from_gdf_dict",
    "get_split_group_column",
    "get_split_group_values",
    "get_split_group_names",
    "extract_index_names",
    "merge_multiple_df",
    "modify_status_colors",
    "assign_season_colors",
    "create_context_page",
    "create_mapbook_ctx_cover",
    "create_mapbook_grouper_ctx",
    "create_grouper_page",
    "merge_mapbook_files",
    "custom_trajectory_segment_filter",
    "filter_groups_by_value_criteria",
    "set_custom_groupers",
    "validate_polygon_geometry",
    "draw_survey_lines",
    "custom_determine_season_windows",
    "transform_gdf_crs",
    "custom_view_state_from_gdf",
]

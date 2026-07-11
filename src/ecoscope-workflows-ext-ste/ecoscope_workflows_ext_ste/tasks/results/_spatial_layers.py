import re
import requests
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
from wt_registry import register
from typing_extensions import Self
import matplotlib.colors as mcolors
from typing import Annotated, Any, Literal, cast
from pydantic.json_schema import SkipJsonSchema
from ecoscope.platform.annotations import AnyGeoDataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ecoscope_workflows_ext_custom.tasks.results._map import (
    GeoJSONLayerStyle,
    IconLayerStyle,
    LayerDefinition,
    LegendSegment,
    LegendValue,
    _color_tuple_to_css,
)

from ..spatial_operations._spatial_features import (
    EarthRangerQuery,
    LocalSpatialLayer,
    _apply_column_style,
)


class _UnifiedLegend:
    GREY = "rgba(128, 128, 128, 1.0)"

    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        self.gdf = gdf
        self._cache: dict[str, str] = {}

    def __call__(self) -> LegendSegment | None:
        if "legend_label" not in self.gdf.columns:
            return None
        return LegendSegment(title=self._title(), values=self._values())

    def _title(self) -> str:
        return str(self.gdf["legend_title"].iloc[0]) if "legend_title" in self.gdf.columns else ""

    def _values(self) -> list[LegendValue]:
        seen: dict[str, str] = {}
        for _, row in self.gdf.iterrows():
            label = str(row["legend_label"])
            if label not in seen:
                seen[label] = self._color(row)
        return [LegendValue(label=k, color=v) for k, v in seen.items()]

    def _color(self, row: pd.Series) -> str:  # type: ignore[type-arg]
        return self._icon_tint(row) or self._geom_color(row) or self._svg_color(row) or self.GREY

    def _icon_tint(self, row: pd.Series) -> str | None:  # type: ignore[type-arg]
        ic = row.get("icon_color")
        return self._css(ic) if ic is not None else None

    def _geom_color(self, row: pd.Series) -> str | None:  # type: ignore[type-arg]
        geom = row.geometry.geom_type if row.geometry is not None else ""
        fill = row.get("get_fill_color")
        line = row.get("get_line_color")
        color = line if ("LineString" in geom or not self._opaque(fill)) else fill
        return self._css(color) if self._opaque(color) else None

    def _svg_color(self, row: pd.Series) -> str | None:  # type: ignore[type-arg]
        if (url := row.get("icon_url")) is None:
            return None
        url = str(url)
        if url not in self._cache:
            self._cache[url] = self._fetch_svg(url) or self.GREY
        return self._cache[url]

    @staticmethod
    def _opaque(color: object) -> bool:
        return isinstance(color, list) and len(color) == 4 and color[3] != 0

    @staticmethod
    def _css(rgba: object) -> str:
        return _color_tuple_to_css(tuple(int(c) for c in rgba))  # type: ignore[attr-defined, arg-type]

    @staticmethod
    def _fetch_svg(url: str) -> str | None:
        try:
            resp = requests.get(url, timeout=5, verify=False)
            if match := re.search(r'fill="(#[0-9a-fA-F]{3,8})"', resp.text):
                from ecoscope.base.utils import (  # type: ignore[import-untyped]
                    hex_to_rgba,
                )

                return _color_tuple_to_css(
                    tuple(int(c) for c in hex_to_rgba(match.group(1)))  # type: ignore[arg-type]
                )
        except Exception:
            pass
        return None


class _GeoJsonLayer:
    _COLS = frozenset({"get_fill_color", "get_line_color", "get_line_width", "get_point_radius"})

    def __init__(self, gdf: gpd.GeoDataFrame, legend: LegendSegment | None) -> None:
        self.gdf = gdf
        self.legend = legend

    def __call__(self) -> LayerDefinition:
        present = self._COLS & set(self.gdf.columns)
        return LayerDefinition(
            layer_type="GeoJsonLayer",
            layer_style=GeoJSONLayerStyle(**{c: c for c in present}),
            legend=self.legend,
            geodataframe=self.gdf,  # type: ignore[arg-type]
        )


class _IconLayer:
    DEFAULT_SIZE = 15.0

    def __init__(self, gdf: gpd.GeoDataFrame, legend: LegendSegment | None) -> None:
        self.gdf = gdf
        self.legend = legend

    def __call__(self) -> LayerDefinition:
        has_custom = "icon_color" in self.gdf.columns
        gdf = self.gdf.copy()
        gdf["_icon_data"] = self._icon_data(gdf, has_custom)  # type: ignore[assignment]
        return LayerDefinition(
            layer_type="IconLayer",
            layer_style=IconLayerStyle(
                get_icon="_icon_data",
                get_size="icon_size" if "icon_size" in gdf.columns else self.DEFAULT_SIZE,
                get_color="icon_color" if has_custom else None,
            ),
            legend=self.legend,
            geodataframe=gdf,  # type: ignore[arg-type]
        )

    def _icon_data(self, gdf: gpd.GeoDataFrame, has_custom: bool) -> list[dict]:  # type: ignore[type-arg]
        sizes = (
            pd.to_numeric(gdf["icon_size"], errors="coerce").fillna(self.DEFAULT_SIZE).tolist()
            if "icon_size" in gdf.columns
            else [self.DEFAULT_SIZE] * len(gdf)
        )
        return [
            {
                "url": str(u),
                "width": int(s),
                "height": int(s),
                **({"mask": True} if has_custom else {}),
            }
            for u, s in zip(gdf["icon_url"], sizes)
        ]


def _layers_for_gdf(gdf: gpd.GeoDataFrame) -> list[LayerDefinition]:
    """Build the GeoJsonLayer / IconLayer definitions for a single styled GDF.

    The legend is rebuilt per GDF, so every source contributes its own legend
    segment (carried on the first non-empty layer for that source).
    """
    if gdf is None or gdf.empty:
        return []

    if "icon_url" in gdf.columns and gdf["icon_url"].notna().any():
        mask = gdf["icon_url"].notna()
        icon_gdf = gdf[mask].explode(index_parts=False).reset_index(drop=True)
        other_gdf = gdf[~mask].copy()
    else:
        icon_gdf, other_gdf = gpd.GeoDataFrame(), gdf

    legend = _UnifiedLegend(gdf)()
    layers: list[LayerDefinition] = []

    if not other_gdf.empty:
        layers.append(_GeoJsonLayer(other_gdf, legend)())
    if not icon_gdf.empty:
        layers.append(_IconLayer(icon_gdf, legend if other_gdf.empty else None)())

    return layers


@register()
def geodataframe_from_layers(
    layers: Annotated[
        list[LayerDefinition],
        Field(
            description="Map overlay layers (e.g. from select_map_overlay) whose "
            "underlying GeoDataFrames should be combined into one.",
            exclude=True,
        ),
    ],
) -> Annotated[AnyGeoDataFrame, Field()]:
    """Combine the GeoDataFrames backing one or more layers into a single GeoDataFrame.

    TextLayer entries (map labels, e.g. the LandDx name labels) are skipped since
    their geometry is a centroid rather than the underlying feature, and layers
    rendered from a `data_url` (no in-memory geodataframe) are skipped since there's
    no local data to extract. Useful for feeding a map overlay (e.g. from
    `select_map_overlay`) into `spatial_join` as `right_df`.
    """
    gdfs = [
        layer.geodataframe for layer in layers if layer.geodataframe is not None and layer.layer_type != "TextLayer"
    ]
    if not gdfs:
        return cast(AnyGeoDataFrame, gpd.GeoDataFrame())
    return cast(AnyGeoDataFrame, pd.concat(gdfs, ignore_index=True))


@register()
def create_spatial_features_layer(
    geodataframes: Annotated[
        list[AnyGeoDataFrame],
        Field(
            description="One styled spatial-features GDF per source.",
            exclude=True,
        ),
    ],
) -> Annotated[list[LayerDefinition], Field()]:
    """Create GeoJsonLayer / IconLayer definitions for one or more styled GDFs.

    Accepts either a list of GeoDataFrames (one per source) or, for backwards
    compatibility, a single GeoDataFrame. Each source is turned into its own
    layer(s) with its own legend.
    """
    # Back-compat: allow a single GeoDataFrame to be passed directly.
    if isinstance(geodataframes, gpd.GeoDataFrame):
        geodataframes = [geodataframes]

    layers: list[LayerDefinition] = []
    for gdf in geodataframes or []:
        layers.extend(_layers_for_gdf(gdf))  # type: ignore[arg-type]
    return layers


@register()
def create_column_styled_layer(
    geodataframes: Annotated[
        list[AnyGeoDataFrame],
        Field(
            description="One or more GeoDataFrames whose rows already carry colour "
            "values in a column (e.g. 'type_hex_colors' / 'type_rgba_colors').",
            exclude=True,
        ),
    ],
    fill_color_col: Annotated[
        str | None,
        Field(
            default=None,
            description="Column holding per-row fill colour (hex like '#a6b697' or "
            "rgba like '(166, 182, 151, 255)'). Applies to polygons and points.",
        ),
    ] = None,
    line_color_col: Annotated[
        str | None,
        Field(
            default=None,
            description="Column holding per-row stroke colour. Defaults to " "fill_color_col if omitted.",
        ),
    ] = None,
    line_width: Annotated[float, Field(default=2.0, description="Border / line width in pixels.")] = 2.0,
    point_radius: Annotated[float, Field(default=8.0, description="Point radius in pixels.")] = 8.0,
    fill_opacity: Annotated[
        float | None,
        Field(default=None, description="Override fill alpha (0-1)."),
    ] = None,
    line_opacity: Annotated[
        float | None,
        Field(default=None, description="Override line alpha (0-1)."),
    ] = None,
    group_by: Annotated[
        str,
        Field(
            default="type_name",
            description="Column used to group features in the legend e.g. 'type'.",
        ),
    ] = "type_name",
    legend_title: Annotated[
        str,
        Field(default="", description="Label shown in the map legend."),
    ] = "",
) -> Annotated[list[LayerDefinition], Field()]:
    """Style features from existing colour columns and build LayerDefinitions.

    Each input GeoDataFrame is styled via _apply_column_style, then converted to
    layer definitions with the shared _layers_for_gdf helper — so the output is
    the same flat list[LayerDefinition] (one legend per source) produced by
    create_spatial_features_layer.
    """
    # Back-compat: allow a single GeoDataFrame to be passed directly.
    if isinstance(geodataframes, gpd.GeoDataFrame):
        geodataframes = [geodataframes]

    layers: list[LayerDefinition] = []
    for gdf in geodataframes or []:
        if gdf is None or gdf.empty:
            continue
        styled = _apply_column_style(
            gdf,
            fill_color_col=fill_color_col,
            line_color_col=line_color_col,
            line_width=line_width,
            point_radius=point_radius,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
            group_by=group_by,
            legend_title=legend_title,
        )
        layers.extend(_layers_for_gdf(styled))  # type: ignore[arg-type]
    return layers


# --------------------------------------------------------------------------- #
# Categorical layer: one spec = source (ER or local file) + colour-by-column.  #
# --------------------------------------------------------------------------- #

ColormapName = Literal[tuple(mpl.colormaps)]  # type: ignore[valid-type]


class ColormapPalette(BaseModel):
    type_: Literal["palette"] = "palette"
    name: Annotated[  # type: ignore[valid-type]
        ColormapName,
        Field(description="Matplotlib colormap name."),
    ] = "tab10"


class CustomPalette(BaseModel):
    type_: Literal["custom"] = "custom"
    colors: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Hex colors e.g. ['#e63946', '#457b9d']. Cycles if more categories than colors.",
        ),
    ]


ColorPalette = Annotated[ColormapPalette | CustomPalette, Field(discriminator="type_")]


def _palette_colors(palette: ColormapPalette | CustomPalette, n: int) -> list[str]:
    """Return n hex colours from a palette spec.

    Custom palettes and small (qualitative) colormaps cycle; continuous
    colormaps are sampled evenly across their range.
    """
    if n <= 0:
        return []
    if isinstance(palette, CustomPalette):
        pool = palette.colors or ["#808080"]
        return [pool[i % len(pool)] for i in range(n)]

    cmap = mpl.colormaps[palette.name]
    if getattr(cmap, "N", 256) <= 20:  # qualitative (tab10, Set2, ...) -> cycle
        pool = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]
        return [pool[i % len(pool)] for i in range(n)]
    # continuous (viridis, plasma, ...) -> evenly spaced samples
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def _validate_unmapped(unmapped: str) -> None:
    if unmapped not in ("auto", "drop") and not unmapped.startswith("#"):
        raise ValueError("`unmapped` must be 'auto', 'drop', or a hex colour like '#808080'.")


def _resolve_category_colors(
    values: list[str],
    color_mapping: dict[str, str],
    unmapped: str,
    palette: ColormapPalette | CustomPalette,
) -> dict[str, str]:
    """Return a complete value -> hex mapping for the given category values.

    User-supplied entries in `color_mapping` win; values not covered are
    handled per `unmapped` ('auto' = palette colours avoiding any manually
    taken ones, 'drop' = left out of the mapping, '#rrggbb' = fixed fallback).
    """
    mapping = dict(color_mapping)
    missing = [v for v in values if v not in mapping]
    if missing and unmapped != "drop":
        if unmapped == "auto":
            taken = {c.lower() for c in mapping.values()}
            pool = _palette_colors(palette, len(missing) + len(taken))
            fresh = [c for c in pool if c.lower() not in taken]
            # If the palette cycles (fewer unique colours than categories),
            # reuse non-taken colours rather than leaving values uncoloured
            # or duplicating a manually-assigned colour.
            base = list(dict.fromkeys(fresh)) or pool
            i = 0
            while len(fresh) < len(missing):
                fresh.append(base[i % len(base)])
                i += 1
            mapping.update(dict(zip(missing, fresh)))
        else:  # fixed hex fallback
            mapping.update({v: unmapped for v in missing})
    return mapping


class CategoricalLayer(BaseModel):
    """One map layer coloured by the values of a column.

    The source is either EarthRanger (set `query`) or a local file (set
    `file_path`) — exactly one must be provided. Colours come from
    `color_mapping` (value -> hex); any values not covered are handled per
    `unmapped`: assigned automatic palette colours, dropped, or given a fixed
    hex colour. With no mapping at all, every value gets an automatic colour.
    """

    model_config = ConfigDict(title="Categorical Layer", str_strip_whitespace=True)

    # --- source: choose exactly one -------------------------------------- #
    query: Annotated[
        EarthRangerQuery | None,
        Field(default=None, description="EarthRanger query (featureset / type / id)."),
    ] = None
    file_path: Annotated[
        str,
        Field(
            default="",
            description="Path to a local GeoJSON / GeoPackage / GeoParquet file.",
        ),
    ] = ""
    layer: Annotated[
        str | None,
        Field(default=None, description="Layer name inside a GeoPackage, if needed."),
    ] = None

    # --- colouring -------------------------------------------------------- #
    color_by: Annotated[
        str,
        Field(description="Column whose values drive the colours e.g. 'type'."),
    ]
    color_mapping: Annotated[
        dict[str, str],
        Field(
            default={},
            description="Value -> hex colour e.g. {'National Park': '#115631'}. "
            "Leave empty to auto-assign colours to every value.",
        ),
    ] = {}
    unmapped: Annotated[
        str,
        Field(
            default="auto",
            description="What to do with values missing from color_mapping: "
            "'auto' assigns palette colours, 'drop' removes those rows, or a "
            "hex colour like '#808080' to use for all of them.",
        ),
    ] = "auto"
    palette: Annotated[
        ColorPalette,
        Field(
            default_factory=ColormapPalette,
            description="Palette used for auto-assigned colours: a matplotlib "
            "colormap (default 'tab10') or a custom list of hex colours.",
        ),
    ]

    # --- rendering --------------------------------------------------------- #
    fill_opacity: Annotated[float | None, Field(default=None, ge=0.0, le=1.0)] = None
    line_opacity: Annotated[float | None, Field(default=None, ge=0.0, le=1.0)] = None
    line_width: Annotated[float, Field(default=2.0)] = 2.0
    point_radius: Annotated[float, Field(default=8.0)] = 8.0
    group_by: Annotated[
        str,
        Field(default="", description="Legend grouping column. Defaults to color_by."),
    ] = ""
    legend_title: Annotated[str, Field(default="")] = ""

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if bool(self.query) == bool(self.file_path):
            raise ValueError("Provide exactly one source: either `query` (EarthRanger) " "or `file_path` (local file).")
        _validate_unmapped(self.unmapped)
        return self

    # --- pipeline ----------------------------------------------------------- #
    def _load(self, client: Any = None) -> gpd.GeoDataFrame:
        if self.query is not None:
            if client is None:
                raise ValueError("An EarthRanger `client` is required when `query` is used.")
            return self.query.get(client)
        return LocalSpatialLayer(file_path=self.file_path, layer=self.layer)._read()

    def _resolve_mapping(self, values: list[str]) -> dict[str, str]:
        return _resolve_category_colors(values, self.color_mapping, self.unmapped, self.palette)

    def build(self, client: Any = None) -> list[LayerDefinition]:
        """Load, colour, style, and convert to LayerDefinitions."""
        gdf = self._load(client)
        if gdf is None or gdf.empty:
            return []
        if self.color_by not in gdf.columns:
            raise KeyError(
                f"color_by column {self.color_by!r} not found. " f"Available: {sorted(map(str, gdf.columns))}"
            )

        gdf = gdf.copy()
        values = gdf[self.color_by].astype(str)
        mapping = self._resolve_mapping(list(pd.unique(values.dropna())))

        gdf["_cat_color"] = values.map(mapping)
        if self.unmapped == "drop":
            gdf = gdf[gdf["_cat_color"].notna()].reset_index(drop=True)
            if gdf.empty:
                return []

        styled = _apply_column_style(
            gdf,
            fill_color_col="_cat_color",
            line_color_col="_cat_color",
            line_width=self.line_width,
            point_radius=self.point_radius,
            fill_opacity=self.fill_opacity,
            line_opacity=self.line_opacity,
            group_by=self.group_by or self.color_by,
            legend_title=self.legend_title,
        )
        return _layers_for_gdf(styled)  # type: ignore[arg-type]


@register()
def create_categorical_layers(
    layers: Annotated[
        list[CategoricalLayer],
        Field(
            description="One or more categorical layer specs (EarthRanger or "
            "local file), each coloured by a column.",
        ),
    ],
    client: Annotated[
        Any,
        Field(
            default=None, description="EarthRanger client, required only for " "layers that use `query`.", exclude=True
        ),
    ] = None,
) -> Annotated[list[LayerDefinition], Field()]:
    """Build LayerDefinitions for one or more column-coloured layers."""
    out: list[LayerDefinition] = []
    for spec in layers or []:
        spec = CategoricalLayer.model_validate(spec)
        out.extend(spec.build(client))
    return out


@register()
def create_categorical_styled_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(
            description="A GeoDataFrame (e.g. from get_spatial_features "
            "or load_local_spatial_file) to colour by the values of a column.",
            exclude=True,
        ),
    ],
    color_by: Annotated[
        str,
        Field(description="Column whose values drive the colours e.g. 'type'."),
    ],
    color_mapping: Annotated[
        dict[str, str],
        Field(
            default={},
            description="Value -> hex colour e.g. {'National Park': '#115631'}. "
            "Leave empty to auto-assign colours to every value.",
        ),
    ] = {},
    unmapped: Annotated[
        str,
        Field(
            default="auto",
            description="What to do with values missing from color_mapping: "
            "'auto' assigns palette colours, 'drop' removes those rows, or a "
            "hex colour like '#808080' to use for all of them.",
        ),
    ] = "auto",
    palette: Annotated[
        ColorPalette | SkipJsonSchema[None],
        Field(
            default=None,
            description="Palette used for auto-assigned colours: a matplotlib "
            "colormap or a custom list of hex colours. Defaults to 'tab10'.",
        ),
    ] = None,
    fill_opacity: Annotated[
        float | None,
        Field(default=None, ge=0.0, le=1.0, description="Override fill alpha (0-1)."),
    ] = None,
    line_opacity: Annotated[
        float | None,
        Field(default=None, ge=0.0, le=1.0, description="Override line alpha (0-1)."),
    ] = None,
    line_width: Annotated[float, Field(default=2.0, description="Border / line width in pixels.")] = 2.0,
    point_radius: Annotated[float, Field(default=2.0, description="Point radius in pixels.")] = 2.0,
    group_by: Annotated[
        str,
        Field(
            default="",
            description="Column used to group features in the legend. " "Defaults to color_by.",
        ),
    ] = "",
    legend_title: Annotated[
        str,
        Field(default="", description="Label shown in the map legend."),
    ] = "",
) -> Annotated[list[LayerDefinition], Field()]:
    """Colour features by the values of a column and build LayerDefinitions.

    A flat-parameter counterpart to CategoricalLayer that plugs into existing
    pipelines: sources are loaded upstream (get_spatial_features /
    load_local_spatial_file) and passed in as a GeoDataFrame.
    """
    _validate_unmapped(unmapped)
    if isinstance(palette, dict):  # direct notebook calls bypass task validation
        from pydantic import TypeAdapter

        palette = TypeAdapter(ColorPalette).validate_python(palette)  # type: ignore[arg-type]
    palette = palette if palette is not None else ColormapPalette()

    gdf = geodataframe.copy()
    if gdf is None or gdf.empty:
        return []
    if color_by not in gdf.columns:
        raise KeyError(f"color_by column {color_by!r} not found. " f"Available: {sorted(map(str, gdf.columns))}")

    all_values = gdf[color_by].astype(str).unique().tolist()
    mapping = _resolve_category_colors(all_values, color_mapping, unmapped, palette)

    gdf = gdf.copy()
    gdf["_cat_color"] = gdf[color_by].astype(str).map(mapping)
    if unmapped == "drop":
        gdf = gdf[gdf["_cat_color"].notna()].reset_index(drop=True)
        if gdf.empty:
            return []
    styled = _apply_column_style(
        gdf,
        fill_color_col="_cat_color",
        line_color_col="_cat_color",
        line_width=line_width,
        point_radius=point_radius,
        fill_opacity=fill_opacity,
        line_opacity=line_opacity,
        group_by=group_by or color_by,
        legend_title=legend_title,
    )
    return list(_layers_for_gdf(styled))  # type: ignore[arg-type]

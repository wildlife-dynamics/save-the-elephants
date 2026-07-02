import ast
import pandas as pd
import geopandas as gpd
from wt_registry import register
from typing import Annotated, Any, TypeAlias, cast
from ecoscope.platform.annotations import (
    AdvancedField,
    AnyGeoDataFrame,
    EmptyDataFrame,
)
from ecoscope.platform.connections import EarthRangerClient
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

RGBA: TypeAlias = list[int]  # [R, G, B, A]


class FeatureStyle(BaseModel):
    """Base for per-geometry-type style classes. Provides hex-to-RGBA conversion."""

    @staticmethod
    def _rgba(colors: str | list[str] | list[RGBA]) -> RGBA | list[RGBA]:
        from ecoscope.base.utils import hex_to_rgba  # type: ignore[import-untyped]

        if isinstance(colors, str):
            return list(hex_to_rgba(colors))
        return [c if isinstance(c, list) else list(hex_to_rgba(c)) for c in colors]


class PolygonStyle(FeatureStyle):
    model_config = ConfigDict(title="")
    fill_color: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=[],
            description="Fill hex colour(s) e.g. ['#FFA500']. Cycles across rows.",
        ),
    ] = []
    stroke_color: Annotated[
        str | SkipJsonSchema[None],
        Field(default=None, description="Border hex colour."),
    ] = None
    fill_opacity: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Fill opacity 0–1. Works with or without fill_color: leave "
            "fill_color empty to keep native EarthRanger colours and only adjust "
            "their opacity. Leave empty to keep the native opacity.",
        ),
    ] = None
    stroke_width: Annotated[float, Field(default=2.0, description="Border width in pixels.")] = 2.0

    @model_validator(mode="after")
    def _convert(self) -> Self:
        if self.fill_color:
            self.fill_color = self._rgba(self.fill_color)  # type: ignore[assignment]
        if isinstance(self.stroke_color, str):
            self.stroke_color = self._rgba(self.stroke_color)  # type: ignore[assignment]
        return self


class LineStyle(FeatureStyle):
    model_config = ConfigDict(title="")
    color: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=[],
            description="Line hex colour(s) e.g. ['#E63946']. Cycles across rows.",
        ),
    ] = []
    opacity: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Line opacity 0–1. Works with or without color: leave color "
            "empty to keep native EarthRanger colours and only adjust their "
            "opacity. Leave empty to keep the native opacity.",
        ),
    ] = None
    width: Annotated[float, Field(default=2.0, description="Line width in pixels.")] = 2.0

    @model_validator(mode="after")
    def _convert(self) -> Self:
        if self.color:
            self.color = self._rgba(self.color)  # type: ignore[assignment]
        return self


class PointStyle(FeatureStyle):
    model_config = ConfigDict(title="")
    color: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=[],
            description="Fill hex colour(s). For SVG icons this tints the marker. Cycles across rows.",
        ),
    ] = []
    size: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            description="Point radius / icon size in pixels. Leave empty to use the size set in EarthRanger.",
        ),
    ] = None

    @model_validator(mode="after")
    def _convert(self) -> Self:
        if self.color:
            self.color = self._rgba(self.color)  # type: ignore[assignment]
        return self


class LayerStyle(BaseModel):
    model_config = ConfigDict(title="")
    polygon: Annotated[
        list[PolygonStyle],
        Field(
            default=[],
            max_length=1,
            description="Polygon styling. Add one entry to override ER native colours.",
        ),
    ] = []
    line: Annotated[
        list[LineStyle],
        Field(
            default=[],
            max_length=1,
            description="Line styling. Add one entry to override ER native colours.",
        ),
    ] = []
    point: Annotated[
        list[PointStyle],
        Field(
            default=[],
            max_length=1,
            description="Point and icon marker styling. Add one entry to override ER native colours.",
        ),
    ] = []


def _apply_geo_style(
    gdf: AnyGeoDataFrame,
    style: LayerStyle,
    group_by: str = "type_name",
    legend_title: str = "",
    server_url: str = "",
) -> AnyGeoDataFrame:
    gdf = gdf.copy()
    n = len(gdf)

    # Rows with an `image` value are SVG icon markers; all others use geometry fill/stroke.
    is_icon = gdf["image"].notna() if "image" in gdf.columns else pd.Series(False, index=gdf.index)
    geom_types = gdf.geometry.geom_type
    is_polygon = geom_types.str.contains("Polygon") & ~is_icon
    is_line = geom_types.str.contains("LineString") & ~is_icon
    is_point = ~is_icon & ~is_polygon & ~geom_types.str.contains("LineString")

    def _ensure(col: str) -> None:
        if col not in gdf.columns:
            gdf[col] = [None] * n

    def _set_icon_url() -> None:
        gdf["icon_url"] = None
        gdf.loc[is_icon, "icon_url"] = (server_url + gdf.loc[is_icon, "image"]).values

    ps = style.polygon[0] if style.polygon else None
    ls = style.line[0] if style.line else None
    pts = style.point[0] if style.point else None

    # ---- 1) Base: native EarthRanger styling from feature properties ---- #
    if "fill" in gdf.columns:
        opacity = gdf.get("fill-opacity", pd.Series(1.0, index=gdf.index))
        gdf["get_fill_color"] = [
            (rgba[:3] + [int((op if pd.notna(op) else 1.0) * 255)])
            if (rgba := _coerce_rgba(f)) is not None
            else [0, 0, 0, 0]
            for f, op in zip(gdf["fill"], opacity)
        ]
    if "stroke" in gdf.columns:
        gdf["get_line_color"] = [_coerce_rgba(s) or [0, 0, 0, 0] for s in gdf["stroke"]]
    if "stroke-width" in gdf.columns:
        gdf["get_line_width"] = pd.to_numeric(gdf["stroke-width"], errors="coerce").fillna(2.0)

    if "width" in gdf.columns and is_point.any():
        gdf.loc[is_point, "get_point_radius"] = pd.to_numeric(gdf.loc[is_point, "width"], errors="coerce").fillna(8.0)

    if is_icon.any():
        _set_icon_url()
        gdf["icon_size"] = pd.to_numeric(gdf.get("width", pd.Series(dtype=float)), errors="coerce").fillna(15.0)

    # ---- 2) Explicit overrides from LayerStyle ---- #
    def _norm(rgba: object) -> list:  # ensure [R, G, B, A]
        rgba = list(rgba)  # type: ignore[call-overload]
        return rgba + [255] if len(rgba) == 3 else rgba

    if ps and is_polygon.any():
        if ps.fill_color:
            _ensure("get_fill_color")
            colors = ps.fill_color
            for i, idx in enumerate(gdf.index[is_polygon]):
                gdf.at[idx, "get_fill_color"] = _norm(colors[i % len(colors)])
        if ps.stroke_color:
            _ensure("get_line_color")
            for idx in gdf.index[is_polygon]:
                gdf.at[idx, "get_line_color"] = _norm(ps.stroke_color)
        gdf.loc[is_polygon, "get_line_width"] = ps.stroke_width

    if ls and is_line.any():
        if ls.color:
            _ensure("get_line_color")
            colors = ls.color
            for i, idx in enumerate(gdf.index[is_line]):
                gdf.at[idx, "get_line_color"] = _norm(colors[i % len(colors)])
        gdf.loc[is_line, "get_line_width"] = ls.width

    if pts:
        if is_point.any():
            if pts.color:
                _ensure("get_fill_color")
                colors = pts.color
                for i, idx in enumerate(gdf.index[is_point]):
                    gdf.at[idx, "get_fill_color"] = _norm(colors[i % len(colors)])
            gdf.loc[is_point, "get_point_radius"] = pts.size if pts.size is not None else 8.0
        if is_icon.any():
            if pts.size is not None:
                gdf["icon_size"] = float(pts.size)
            if pts.color:
                gdf["icon_color"] = None
                colors = pts.color
                for i, idx in enumerate(gdf.index[is_icon]):
                    gdf.at[idx, "icon_color"] = _norm(colors[i % len(colors)])

    # ---- 3) Opacity adjustments (apply to native OR explicit colours) ---- #
    def _set_alpha(col: str, mask: "pd.Series[bool]", opacity_01: float) -> None:
        if col not in gdf.columns:
            return
        alpha = int(opacity_01 * 255)
        for idx in gdf.index[mask]:
            c = gdf.at[idx, col]
            if isinstance(c, list) and len(c) == 4 and c[3] != 0:
                gdf.at[idx, col] = c[:3] + [alpha]

    if ps and ps.fill_opacity is not None:
        _set_alpha("get_fill_color", is_polygon, ps.fill_opacity)
    if ls and ls.opacity is not None:
        _set_alpha("get_line_color", is_line, ls.opacity)

    if any(c in gdf.columns for c in ("get_fill_color", "get_line_color", "icon_url")):
        col = group_by if (group_by == "geom_type" or group_by in gdf.columns) else "geom_type"
        gdf["legend_title"] = legend_title
        gdf["legend_label"] = gdf.geometry.geom_type.astype(str) if col == "geom_type" else gdf[col].astype(str)

    return gdf


def _featuresets_from_response(
    response: dict[str, Any] | list[Any],
) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        return response.get("features", [])
    return response


@register()
def get_featureset(
    client: EarthRangerClient,
    featureset_id: Annotated[
        str,
        Field(description="Unique identifier of the featureset."),
    ],
) -> AnyGeoDataFrame | EmptyDataFrame:
    """Retrieve all spatial features belonging to an EarthRanger featureset."""
    response = client._get(f"featureset/{featureset_id}/")  # type: ignore[attr-defined]
    if not isinstance(response, dict):
        return cast(EmptyDataFrame, pd.DataFrame())
    if not (features := response.get("features", [])):
        return cast(EmptyDataFrame, pd.DataFrame())
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    return cast(AnyGeoDataFrame, gdf)


# --------------------------------------------------------------------------- #
# Queries: each knows only how to LOAD a GeoDataFrame. Styling / grouping /    #
# legends are owned by SpatialFeatureLayer so they can vary per layer.         #
# --------------------------------------------------------------------------- #
class FeatureSetQuery(BaseModel):
    """Load all features from a named EarthRanger featureset."""

    model_config = ConfigDict(title="Feature Set", str_strip_whitespace=True)
    featureset_name: Annotated[
        str,
        Field(
            min_length=1,
            description="Display name of the featureset exactly as it appears in EarthRanger e.g. 'Boundaries'.",
        ),
    ]

    def get(self, client: EarthRangerClient) -> AnyGeoDataFrame:
        response = client._get("featureset/")  # type: ignore[attr-defined]
        featuresets = _featuresets_from_response(response)
        featureset = next((fs for fs in featuresets if fs["name"] == self.featureset_name), None)
        if featureset is None:
            raise ValueError(
                f"Featureset {self.featureset_name!r} not found. " f"Available: {[fs['name'] for fs in featuresets]}"
            )
        result = get_featureset(client, featureset["id"])
        if not isinstance(result, gpd.GeoDataFrame):
            return cast(AnyGeoDataFrame, gpd.GeoDataFrame())
        return cast(AnyGeoDataFrame, result)


class FeatureTypeQuery(BaseModel):
    """Load all features of a given type across EarthRanger featuresets."""

    model_config = ConfigDict(title="Feature Type", str_strip_whitespace=True)
    feature_type: Annotated[
        str,
        Field(
            min_length=1,
            description="Feature type name as shown in EarthRanger e.g. 'Conservancy'.",
        ),
    ]

    def get(self, client: EarthRangerClient) -> AnyGeoDataFrame:
        feature_classes: list[dict[str, Any]] = client._get("featureclass/")  # type: ignore[attr-defined]
        feature_class = next((fc for fc in feature_classes if fc["name"] == self.feature_type), None)
        if feature_class is None:
            raise ValueError(f"Feature type {self.feature_type!r} not found.")
        if not feature_class.get("feature_set_id"):
            raise ValueError(f"Feature type {self.feature_type!r} is not linked to a featureset.")
        result = get_featureset(client, feature_class["feature_set_id"])
        if not isinstance(result, gpd.GeoDataFrame):
            return cast(AnyGeoDataFrame, gpd.GeoDataFrame())
        return cast(
            AnyGeoDataFrame,
            result[result["type_name"] == self.feature_type].reset_index(drop=True),
        )


class FeatureIdQuery(BaseModel):
    """Load a single spatial feature by its EarthRanger UUID."""

    model_config = ConfigDict(title="Feature ID", str_strip_whitespace=True)
    feature_id: Annotated[
        str,
        Field(description="UUID of a specific spatial feature available on EarthRanger."),
    ]

    def get(self, client: EarthRangerClient) -> AnyGeoDataFrame:
        response = client._get(f"feature/{self.feature_id}/")  # type: ignore[attr-defined]
        if not isinstance(response, dict):
            return cast(AnyGeoDataFrame, gpd.GeoDataFrame())
        if not (features := response.get("features", [])):
            return cast(AnyGeoDataFrame, gpd.GeoDataFrame())
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        featuresets = _featuresets_from_response(client._get("featureset/"))  # type: ignore[attr-defined]
        type_map: dict[str, str] = {}
        for featureset in featuresets:
            for entry in featureset.get("types", []):
                type_map[entry["id"]] = entry["name"]
        gdf["type_name"] = gdf["feature_type"].map(type_map)
        return cast(AnyGeoDataFrame, gdf)


EarthRangerQuery: TypeAlias = FeatureSetQuery | FeatureTypeQuery | FeatureIdQuery


# --------------------------------------------------------------------------- #
# Per-layer spec: one query + how to render it. A list of these = many layers. #
# --------------------------------------------------------------------------- #
class SpatialFeatureLayer(BaseModel):
    """A single EarthRanger map layer: what to load and how to render it."""

    model_config = ConfigDict(title="Spatial Feature Layer", str_strip_whitespace=True)
    query: Annotated[
        EarthRangerQuery,
        Field(description="What to load from EarthRanger for this layer."),
    ]
    style: Annotated[
        list[LayerStyle],
        Field(
            default=[],
            max_length=1,
            description="Optional: Override how EarthRanger spatial features are rendered on the map.",
        ),
    ] = []
    group_by: Annotated[
        str,
        AdvancedField(
            default="type_name",
            description="Column used to group features in this layer's legend.",
            json_schema_extra={
                "oneOf": [
                    {"const": "type_name", "title": "Feature Type"},
                    {"const": "title", "title": "Feature Name"},
                ]
            },
        ),
    ] = "type_name"
    legend_title: Annotated[
        str,
        AdvancedField(
            default="",
            description="Label shown in this layer's legend e.g. 'Park Boundary'.",
        ),
    ] = ""

    def load(self, client: EarthRangerClient, server_url: str) -> AnyGeoDataFrame:
        gdf = self.query.get(client)
        if gdf.empty:
            return cast(AnyGeoDataFrame, gdf)
        style = self.style[0] if self.style else LayerStyle()
        return _apply_geo_style(gdf, style, self.group_by, self.legend_title, server_url)


@register()
def get_spatial_features(
    client: EarthRangerClient,
    layers: Annotated[
        list[SpatialFeatureLayer] | SkipJsonSchema[None],
        Field(
            default=None,
            description="One or more EarthRanger layers to load and style.",
        ),
    ] = None,
) -> list[AnyGeoDataFrame]:
    """Load one or more spatial-feature layers from EarthRanger and apply styling.

    Returns one styled GeoDataFrame per requested layer (empty results skipped),
    so downstream layer creation can render each as an independent map layer.
    """
    if not layers:
        return []
    server_url = getattr(client, "server", "").rstrip("/")
    out: list[AnyGeoDataFrame] = []
    for spec in layers:
        spec = SpatialFeatureLayer.model_validate(spec)
        styled = spec.load(client, server_url)
        if styled.empty:
            continue
        out.append(styled)
    return out


# --------------------------------------------------------------------------- #
# Local / custom files: same pattern — a list of file specs => many layers.    #
# --------------------------------------------------------------------------- #
class LocalSpatialLayer(BaseModel):
    """A single local-file map layer: what to read and how to render it."""

    model_config = ConfigDict(title="Local Spatial File", str_strip_whitespace=True)
    file_path: Annotated[
        str,
        AdvancedField(
            default="",
            description="Path to the geospatial file.",
        ),
    ] = ""
    layer: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Layer name within a GeoPackage file. Only required when the file contains multiple layers.",
        ),
    ] = None
    split_by: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Optional: split a single file into one map layer per distinct value of this column.",
        ),
    ] = None
    group_by: Annotated[
        str,
        AdvancedField(
            default="",
            description="Column used to group features in the legend e.g. 'name', 'category'.",
        ),
    ] = ""
    legend_title: Annotated[
        str,
        AdvancedField(
            default="",
            description="Label shown in the map legend e.g. 'Park Boundary'.",
        ),
    ] = ""
    style: Annotated[
        list[LayerStyle] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            max_length=1,
            description="Optional: Customise how features are rendered on the map.",
        ),
    ] = None

    def _read(self) -> gpd.GeoDataFrame:
        path_lower = self.file_path.lower().split("?")[0]
        if path_lower.endswith((".parquet", ".geoparquet")):
            return gpd.read_parquet(self.file_path)
        if self.layer:
            return gpd.read_file(self.file_path, layer=self.layer)
        return gpd.read_file(self.file_path)

    def load(self) -> list[AnyGeoDataFrame]:
        if not self.file_path:
            return []
        gdf = self._read()
        if gdf.empty:
            return []

        style_obj = self.style[0] if self.style else LayerStyle()
        group_by = self.group_by or "geom_type"

        # One file -> one layer (default), or split into one layer per group value.
        if self.split_by and self.split_by in gdf.columns:
            chunks = [cast(gpd.GeoDataFrame, sub.reset_index(drop=True)) for _, sub in gdf.groupby(self.split_by)]
        else:
            chunks = [gdf]

        out: list[AnyGeoDataFrame] = []
        for chunk in chunks:
            if chunk.empty:
                continue
            title = self.legend_title
            if self.split_by and self.split_by in chunk.columns and not title:
                title = str(chunk[self.split_by].iloc[0])
            out.append(_apply_geo_style(cast(AnyGeoDataFrame, chunk), style_obj, group_by, title))
        return out


@register()
def load_local_spatial_file(
    files: Annotated[
        list[LocalSpatialLayer] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="One or more local geospatial files to load and style. Each file becomes its own map layer.",
        ),
    ] = None,
) -> list[AnyGeoDataFrame]:
    """Load one or more local geospatial files and apply styling.

    Returns one styled GeoDataFrame per layer (per file, or per `split_by`
    group within a file), so downstream layer creation renders each separately.
    """
    if not files:
        return []
    out: list[AnyGeoDataFrame] = []
    for spec in files:
        spec = LocalSpatialLayer.model_validate(spec)
        out.extend(spec.load())
    return out


def _coerce_rgba(value: Any, opacity: float | None = None) -> list[int] | None:
    """Turn a single cell into an [R, G, B, A] list.

    Accepts:
      - hex string:            "#a6b697" / "#a6b697ff"
      - real list/tuple:       [166, 182, 151, 255] / (166, 182, 151)
      - stringified tuple:     "(166, 182, 151, 255)"  (common after CSV/Parquet round-trip)
    `opacity` (0-1), if given, overrides the alpha channel.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    rgba: list[int] | None = None

    if isinstance(value, str):
        s = value.strip()
        if s.startswith("#"):
            hex_part = s.lstrip("#")
            # Expand CSS shorthand (#abc / #abcd) to full form.
            if len(hex_part) in (3, 4):
                hex_part = "".join(ch * 2 for ch in hex_part)
            try:
                from ecoscope.base.utils import (  # type: ignore[import-untyped]
                    hex_to_rgba,
                )

                rgba = list(hex_to_rgba("#" + hex_part))
            except (ValueError, TypeError):
                return None
        elif s.startswith("(") or s.startswith("["):
            try:
                rgba = [int(c) for c in ast.literal_eval(s)]
            except (ValueError, SyntaxError, TypeError):
                return None
        else:
            # 'none', 'transparent', named colours, rgb(...) strings, etc.
            return None
    elif isinstance(value, (list, tuple)):
        try:
            rgba = [int(c) for c in value]
        except (ValueError, TypeError):
            return None
    else:
        return None

    if rgba is not None and (len(rgba) not in (3, 4) or any(not (0 <= c <= 255) for c in rgba)):
        return None

    if rgba is None:
        return None
    if len(rgba) == 3:
        rgba = rgba + [255]
    if opacity is not None:
        rgba = rgba[:3] + [int(opacity * 255)]
    return rgba


def _apply_column_style(
    gdf: "AnyGeoDataFrame",  # type: ignore[name-defined] # noqa: F821
    fill_color_col: str | None = None,
    line_color_col: str | None = None,
    line_width: float = 2.0,
    point_radius: float = 8.0,
    fill_opacity: float | None = None,
    line_opacity: float | None = None,
    group_by: str = "type_name",
    legend_title: str = "",
) -> "AnyGeoDataFrame":  # type: ignore[name-defined] # noqa: F821
    """Style a GeoDataFrame from colour values already present in its columns.

    Use this instead of the LayerStyle/SVG paths when each row carries its own
    colour, e.g. a `type_hex_colors` ("#a6b697") or `type_rgba_colors`
    ("(166, 182, 151, 255)") column. Maps those values onto the deck.gl
    accessors (`get_fill_color`, `get_line_color`, `get_line_width`,
    `get_point_radius`) and wires up the same legend columns as _apply_geo_style.

    Parameters
    ----------
    fill_color_col : name of the column holding per-row fill colour (polygons + points).
    line_color_col : name of the column holding per-row stroke colour.
                     Defaults to `fill_color_col` if not given.
    """
    gdf = gdf.copy()

    if line_color_col is None:
        line_color_col = fill_color_col

    geom_types = gdf.geometry.geom_type
    is_polygon = geom_types.str.contains("Polygon")
    is_line = geom_types.str.contains("LineString")
    is_point = ~is_polygon & ~is_line

    # Fill colour -> polygons and points.
    if fill_color_col and fill_color_col in gdf.columns:
        fill = gdf[fill_color_col].map(lambda v: _coerce_rgba(v, opacity=fill_opacity))
        mask = (is_polygon | is_point) & fill.notna()
        if mask.any():
            gdf["get_fill_color"] = None
            gdf.loc[mask, "get_fill_color"] = fill[mask]

    # Line / stroke colour -> polygons (border) and lines.
    if line_color_col and line_color_col in gdf.columns:
        line = gdf[line_color_col].map(lambda v: _coerce_rgba(v, opacity=line_opacity))
        mask = (is_polygon | is_line) & line.notna()
        if mask.any():
            gdf["get_line_color"] = None
            gdf.loc[mask, "get_line_color"] = line[mask]

    if (is_polygon | is_line).any():
        gdf.loc[is_polygon | is_line, "get_line_width"] = line_width
    if is_point.any():
        gdf.loc[is_point, "get_point_radius"] = point_radius

    # Legend wiring — identical contract to _apply_geo_style.
    if any(c in gdf.columns for c in ("get_fill_color", "get_line_color")):
        col = group_by if (group_by == "geom_type" or group_by in gdf.columns) else "geom_type"
        gdf["legend_title"] = legend_title
        gdf["legend_label"] = gdf.geometry.geom_type.astype(str) if col == "geom_type" else gdf[col].astype(str)

    return gdf

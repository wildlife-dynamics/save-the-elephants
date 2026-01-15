# Aerial Survey Lines Workflow

This workflow generates **aerial survey lines** from polygon or
multipolygon spatial inputs and visualizes them on an interactive map
and dashboard. It supports downloading spatial data from a URL or
loading it from a local path and allows users to control survey
direction and spacing.

------------------------------------------------------------------------

## Workflow Overview

The Aerial Survey Lines Workflow is designed to: - Ingest polygon or
multipolygon spatial data - Generate evenly spaced aerial survey lines -
Render the original input data and generated survey lines on a map -
Export results in multiple geospatial formats for downstream use

> **Note:** This workflow only processes **Polygon** and
> **MultiPolygon** geometries. Other geometry types are not supported.

------------------------------------------------------------------------

## Workflow Configuration

### 1. Set Workflow Details

Provide information that helps distinguish this workflow from others.

-   **Workflow Name**\
    A unique name for this workflow run.

-   **Workflow Description**\
    A short description explaining the purpose or context of the
    workflow.

------------------------------------------------------------------------

### 2. Set Groupers (Optional)

Groupers define how data is grouped to create separate views in the
dashboard.

-   If groupers are provided, the dashboard will generate separate views
    per group.
-   If left blank, all data will appear in a single view.
-   Groupers are **not required** to run the workflow.

------------------------------------------------------------------------

### 3. Configure Base Map Layers

Specify the base map layers that will be used for visualization in the
output map and dashboard.

------------------------------------------------------------------------

## Advanced Configuration

### Retrieve Input Shapefile

#### Input Method

The workflow supports two input methods:

-   Download from URL
-   Load from Local Path

#### Download from URL

When using this option, provide:

-   **URL**\
    A direct link to a spatial file. Supported formats include:
    -   `.gpkg`
    -   `.shp`
    -   `.geoparquet`

The file will be downloaded and processed automatically.

------------------------------------------------------------------------

### Geometry Requirements

-   Only **Polygon** and **MultiPolygon** geometries are supported.
-   Layers already loaded in a GeoDataFrame (`gdf`) are **not yet
    supported**.

------------------------------------------------------------------------

## Draw Aerial Survey Lines

Configure how the aerial survey lines are generated.

### Direction

Specify the orientation of the survey lines:

-   North--South
-   East--West

### Spacing

Define the distance between adjacent survey lines.

------------------------------------------------------------------------

## Outputs

This workflow produces the following outputs:

-   **Aerial survey lines** exported as:
    -   GeoPackage (`.gpkg`, EPSG:3857)
    -   GeoParquet
-   **HTML map** displaying:
    -   The loaded input polygons
    -   Generated aerial survey lines
-   **Dashboard map rendering** for interactive exploration and review

------------------------------------------------------------------------

## Notes & Limitations

-   Groupers are optional.
-   Only polygon-based geometries are supported.
-   Input layers provided directly as a GeoDataFrame are not currently
    supported.
-   All generated spatial outputs use **EPSG:3857**.

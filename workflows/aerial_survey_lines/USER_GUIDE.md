# Aerial Survey Workflow — Quick Guide

## Outputs

The workflow generates the following files:

- `aerial_survey.gpkg` – Survey flight lines (spatial data)
- `aerial_survey.html` – Interactive web map
- `aerial_survey.parquet` – Optimized data format
- `community-conservancy.gpkg` – Area boundary data
- `result.json` – Workflow run summary


## Inputs

### Workflow Details
- **Workflow Name** (required)
- **Workflow Description** (optional)

### Time Range
- **Since** – Start time
- **Until** – End time

### Optional Settings
- **Groupers** – Control how data is grouped
- **Base Map Layers** – Background map style

### Spatial Input
You can provide boundary data in two ways:

1. **Download from URL**
   - Supports: `.gpkg`, `.shp`, `.geoparquet`

2. **Local File Path**
   - Example:
     `/Users/username/data/boundary.gpkg`

### Aerial Survey Lines
- **Direction** – Orientation of flight lines
- **Spacing** – Distance between lines
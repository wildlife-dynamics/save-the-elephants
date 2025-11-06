# Aerial Survey Lines Workflow

The **Aerial Survey Lines Workflow** generates aerial survey line features from a user-provided Region of Interest (ROI) and produces geospatial datasets, an interactive ecomap, and a dashboard summarizing the surveyed area.

---

## Overview

This workflow:

1. Accepts a **URL link** to a shapefile or GeoPackage containing the ROI.
2. Downloads and loads the ROI.
3. Generates **aerial survey lines** inside the ROI.
4. Saves the output survey lines as:
   - **GeoPackage (.gpkg)**
   - **GeoParquet (.geoparquet)**
5. Creates:
   - An **HTML ecomap**
   - A **widget map view**
   - A **dashboard** aggregating metadata and visualization

---

## Workflow Steps

| Step | Purpose |
|------|---------|
| **Initialize Workflow Metadata** | Sets workflow identification and metadata. |
| **Define Time Range** | Captures the workflow time window. |
| **Configure Grouping Strategy** | Defines how results should be grouped (none, temporal, or by column). |
| **Configure Base Map Layers** | Sets map tile layers used in the final map. |
| **Download ROI** | Downloads the user-provided ROI dataset. |
| **Generate Aerial Survey Lines** | Builds line geometry from the ROI. |
| **Persist Outputs** | Saves the generated lines as `.gpkg` and `.geoparquet`. |
| **Create Polyline Map Layer** | Styles the generated aerial survey lines. |
| **Compute View State** | Auto-centers and zooms the map to the data. |
| **Create Ecomap** | Generates an interactive HTML map. |
| **Create Widgets & Dashboard** | Packages the visualization into a dashboard for reporting. |

---

## User Inputs

### Required Input  
**public or accessible URL** pointing to a shapefile ZIP or GeoPackage:

**Accepted formats:**
- `.zip` containing shapefile components  
- `.gpkg`  

---

## Outputs

All final outputs are saved into:

```
$ECOSCOPE_WORKFLOWS_RESULTS/
```

### Geospatial Data  
- `aerial_survey_lines.gpkg`  
- `aerial_survey_lines.geoparquet`

### Map Output  
- `aerial_survey_lines.html` — interactive ecomap

### Visualization Components  
- Aerial Survey Map Widget  
- Dashboard containing:
  - Workflow metadata  
  - Time range  
  - Grouping strategy  
  - Aerial survey map widget  

---

## Running the Workflow

1. Launch the workflow using your ecoscope workflows runner.
2. Provide the **ROI download URL** when prompted.
3. Let the workflow complete processing.
4. Retrieve:
   - Generated geospatial files  
   - Ecomap HTML  
   - Dashboard output  
   from your results directory.

---

## Future Enhancements

### Local File Support  
Allow users to load ROI files from a local directory path.

### Layer / Column Selection  
Expose parameters like `layer_name` and `roi_column` for finer control.

### Multi‑ROI Support  
Enable processing multiple ROI areas in a single run.


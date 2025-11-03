# Mapbook Workflow

## Overview
The **Mapbook Workflow** is designed to showcase the movement of collared elephants over a specified time period.  
It integrates data from **EarthRanger** and **Google Earth Engine (GEE)** to generate both analytical and visual outputs — including **maps, reports, and interactive dashboards**.

---

## Workflow Description
```yaml
initialize_workflow_metadata:
  name: "Mapbook workflow"
  description: "Workflow to showcase movement of collared elephants that entails various metrics"

define_time_range:
  since: "2025-05-01T00:00:00Z"
  until: "2025-10-31T23:59:59Z"

er_client_name:
  data_source:
    name: "mep-dev"

gee_project_name:
  data_source:
    name: "ecoscope_poc"

configure_grouping_strategy:
  groupers: 
     - subject_name

subject_observations:
  subject_group_name: "Elephants"

convert_to_trajectories:
  trajectory_segment_filter:
    min_length_meters: 0.001
    max_length_meters: 5000.0
    max_time_secs: 7200.0
    min_time_secs: 1.0
    max_speed_kmhr: 9.0
    min_speed_kmhr: 0.01

configure_base_maps:
  base_maps:
    - url: "https://server.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}"
      opacity: 1.0
      max_zoom: 15
      min_zoom: null

download_logo_path:
  url: "https://www.dropbox.com/scl/fi/se63iljd7h9zdi4szv2pt/MEP-logo-dark-linear.png?rlkey=usqs76f4ejd2nyqrpmwxmks0o&st=oewz7ftm&dl=0"
```

---

## Required Parameters

| Parameter | Description | Example |
|------------|--------------|----------|
| `since` | Start date for the analysis period | `"2025-05-01T00:00:00Z"` |
| `until` | End date for the analysis period | `"2025-10-31T23:59:59Z"` |
| `er_client_name` | EarthRanger data source name | `"mep-dev"` |
| `gee_project_name` | Google Earth Engine project ID | `"ecoscope_poc"` |
| `subject_group_name` | Target subject group to visualize | `"Elephants"` |
| `base_maps` | Basemap layer URLs and properties | ArcGIS Hillshade |
| `trajectory_segment_filter` | Motion and distance filtering thresholds | *(See YAML above)* |
| `download_logo_path` | URL of the logo for report branding | Dropbox link |

---

## Maps Produced

The workflow automatically generates **6 analytical maps** for each tracked subject:

1. **Current vs. Previous Quarter Movement**  
   - Compares spatial movement patterns across quarters.

2. **Subject Speed Map**  
   - Visualizes spatial speed variations along trajectories.

3. **Night vs. Day Movement**  
   - Differentiates between nocturnal and diurnal movement.

4. **Home Range (MCP)**  
   - Calculates the Minimum Convex Polygon representing home range.

5. **Mean Raster Value**  
   - Computes average raster (e.g., vegetation index) values over the subject’s path.

6. **Movement by Seasons**  
   - Displays seasonal variations in movement behavior.

---

## Additional Metrics Displayed

Each subject’s report also includes:
- **Tracking duration (months)**
- **Grid area covered**
- **Home range area (MCP)**
- **Subject gender**
- **Seasonal movement summaries**

---

## Preset Configurations

| Setting | Value |
|----------|--------|
| Grid size | `2000 x 2000` |
| Temporal grouping | `subject_name` |
| Text layer | Included on all basemaps |

---

## Outputs

### 1. Generated Documents
The workflow will download and compile:
- A **cover page** for the Mapbook report  
- Individual **Word document pages** for each subject  
- A **combined Mapbook Word document** containing all subjects and metrics  

### 2. Interactive Dashboard
An interactive dashboard is also generated, allowing users to:
- Explore movement and metrics visually  
- Filter, group, and analyze based on configured groupers  

---

## Notes

- Requires valid access to **EarthRanger** and **Google Earth Engine (GEE)**.  
- Internet connectivity is required for logo download ,base map access, earthranger interaction and gee connectivity.  
- Text layers are automatically included on basemaps for contextual labeling.  

---

## Output Summary

| Output Type | Description | Format |
|--------------|-------------|---------|
| Maps | Analytical and visual map outputs | `.png`, `.geojson` |
| Reports | Mapbook-style summaries per subject | `.docx` |
| Dashboard | Interactive exploration and visualization tool | `.html` |

---

**Author:** Wildlife Dynamics
**Last Updated:** November 2025  

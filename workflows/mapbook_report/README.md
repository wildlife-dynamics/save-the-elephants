# STE Mapbook Workflow

This document describes how to configure and run the **STE Mapbook workflow**, including analysis setup, data sources, comparison periods, and generated outputs.

---

## Overview

The STE Mapbook workflow produces movement analysis maps, spatial datasets, dashboards, and Word reports for a selected **subject group**.  
It supports **time-based comparisons** by analyzing a *current period* alongside a *previous period*.

---

## 1. Set Workflow Details

Provide metadata to help identify and distinguish this workflow run.

### Workflow Name *(required)*
A short, descriptive name for the workflow.

**Example**  
`STE Mapbook – Elephant Movement Q4 2025`

### Workflow Description
Optional description of the analysis purpose.

---

## 2. Define Analysis Time Range

Specify the **current analysis period**.

### Since *(required)*
Start date and time for the analysis.

### Until *(required)*
End date and time for the analysis.

This period defines the **current trajectories and relocations** used across all outputs.

---

## 3. Configure Grouping Strategy (Optional)

Specify how data should be grouped to create separate views in dashboards and reports.

- Leave empty to analyze all data in a single view
- Add grouping fields to split outputs (e.g. by subject name)

---

## 4. Configure Base Map Layers

Select the base map layers to be used in all generated visualizations.

---

## 5. Advanced Configurations

### 5.1 Set Previous Period Range

Define the **comparison period** used alongside the current analysis period.

The previous period is used to generate:
- Previous trajectories
- Previous relocations

---

### Option

Choose how the previous period dates should be derived.

---

### PreviousPeriodType *(required)*

Select one of the predefined previous period calculation strategies:

| Value | Description |
|------|-------------|
| `Same as current` | Uses the same date range as the current analysis period |
| `Previous month` | One month immediately preceding the current period |
| `Previous 3 months` | Three months immediately preceding the current period |
| `Previous 6 months` | Six months immediately preceding the current period |
| `Previous year` | One year immediately preceding the current period |

---

### Custom Previous Period

In addition to predefined options, the workflow supports a **Custom Previous Period**.

When **Custom Period** is selected:
- The user manually specifies:
  - Custom **Since** date
  - Custom **Until** date
- These dates are used directly as the previous comparison range

---

## 6. Connect Data Sources

### 6.1 Connect to EarthRanger

#### Data Source *(required)*
Select the configured **EarthRanger** data source that provides subject relocations and trajectories.

---

### 6.2 Connect to Earth Engine

#### Data Source *(required)*
Select the configured **Earth Engine** data source used for raster-based analyses.

---

## 7. Subject Group Selection

### Subject Group Name *(required)*
Choose the subject group to analyze.

All movement, spatial, and raster outputs are generated for this group.

---

## 8. Load LandDx Database

Specify how the LandDx dataset should be provided.

### Input Method *(required)*
- **Download from URL**
- **Load local file**

#### Download from URL
**LandDx Download URL**
```
https://www.dropbox.com/scl/fi/v9maw2jeg1zptv68qtpv3/landDx.gpkg?rlkey=kez5vsbxkgha2emfy5kzwa5n1&st=98v4anq3&dl=0
```

---

## 9. Trajectory Segment Filters *(required)*

Apply constraints to trajectory segments included in the analysis:

- Min Length (meters)
- Max Length (meters)
- Min Time (seconds)
- Max Time (seconds)
- Min Speed (km/hr)
- Max Speed (km/hr)

---

## 10. Outputs

### Maps
- Speed maps
- Movement tracks maps
- Home range maps (includes MCP)
- Day / Night movement maps
- Seasonal maps
- Mean speed raster map

### Spatial Data
- Trajectories (current period)
- Trajectories (previous period)
- Relocations (current period)
- Relocations (previous period)
- Mean speed raster `.tif` files

### Reports
- Grouper-based context Word documents
- Combined Word document

### Dashboard
- Interactive dashboard summarizing maps, metrics, and grouped views

---

## Notes

- Ensure EarthRanger and Earth Engine data sources are configured before running the workflow.
- Grouping strategies affect both report generation and dashboard structure.
- Selecting an appropriate previous period is critical for meaningful comparisons.

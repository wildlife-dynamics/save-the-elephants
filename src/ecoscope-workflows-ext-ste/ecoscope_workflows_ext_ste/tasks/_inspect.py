from pydantic import Field
from typing import Annotated
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame


@task
def view_df(gdf: Annotated[AnyGeoDataFrame, Field(description="A GeoDataFrame to inspect")], name: str) -> None:
    print(f"\nDisplaying data for {name}")
    print("\n--- GeoDataFrame Summary ---")

    if gdf.empty:
        print("The GeoDataFrame is empty.")
        return

    print(f"Number of rows: {gdf.shape[0]}")
    print(f"Number of columns: {gdf.shape[1]}")
    print(f"Geometry column: {gdf.geometry.name}")

    print("\n--- Column Details ---")
    for col in gdf.columns:
        dtype = gdf[col].dtype
        unique_vals = gdf[col].unique()
        n_unique = len(unique_vals)
        print(f"\n- {col} ({dtype})")
        print(f"  Unique values ({n_unique}):")
        if n_unique <= 10:
            for val in unique_vals:
                print(f"    - {val}")
        else:
            print(f"    - [Showing first 10 of {n_unique} unique values]")
            for val in unique_vals[:10]:
                print(f"    - {val}")

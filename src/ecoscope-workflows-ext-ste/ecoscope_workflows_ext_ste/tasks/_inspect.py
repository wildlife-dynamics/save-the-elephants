from pydantic import Field
from typing import Annotated
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame

from typing import TypeVar

T = TypeVar("T")


@task
def print_output(value: T) -> None:
    print("\n--- Print Output Task ---")
    print(f"Output value: {value}")


@task
def view_df(gdf: Annotated[AnyDataFrame, Field(description="A GeoDataFrame to inspect")], name: str) -> None:
    print(f"\nDisplaying data for {name}")
    print("\n--- GeoDataFrame Summary ---")

    if gdf.empty:
        print("The GeoDataFrame is empty.")
        return

    print(f"Number of rows: {gdf.shape[0]}")
    print(f"Number of columns: {gdf.shape[1]}")
    # print(f"Geometry column: {gdf.geometry.name}")

    for col in gdf.columns:
        print(f"column name: {col}")


@task
def view_gdf(gdf: Annotated[AnyGeoDataFrame, Field(description="A GeoDataFrame to inspect")], name: str) -> None:
    print(f"\nDisplaying data for {name}")
    print("\n--- GeoDataFrame Summary ---")

    if gdf.empty:
        print("The GeoDataFrame is empty.")
        return

    print(f"Number of rows: {gdf.shape[0]}")
    print(f"Number of columns: {gdf.shape[1]}")
    print(f"Geometry column: {gdf.geometry.name}")

    print("\n--- Column Types ---")
    for col in gdf.columns:
        print(f"column name: {col}")
        print(f"  Type: {gdf[col].dtype}")  # Changed from col.dtype to gdf[col].dtype

    print("\n--- First Five Rows ---")
    print(gdf.head())  # Removed f-string, just print the dataframe

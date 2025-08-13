from pydantic import Field
from typing import Annotated
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame


@task
def view_df(gdf: Annotated[AnyDataFrame, Field(description="A GeoDataFrame to inspect")], name: str) -> AnyDataFrame:
    print(f"\nDisplaying data for {name}")
    print("\n--- GeoDataFrame Summary ---")

    if gdf.empty:
        print("The GeoDataFrame is empty.")
        return gdf

    print(f"Rows: {gdf.shape[0]}")
    print(f"Columns: {gdf.shape[1]}")
    try:
        print(f"CRS: {gdf.crs}")
    except Exception:
        pass

    print("\n--- Column Details ---")
    print(f"Column names: {gdf.columns.tolist()}")

    print("\n--- Column Types and Unique Values ---")
    for col in gdf.columns:
        col_type = str(gdf[col].dtype)
        null_count = gdf[col].isnull().sum()
        non_null_count = gdf[col].notnull().sum()

        print(f"\nColumn: '{col}'")
        print(f"  Type: {col_type}")
        print(f"  Non-null: {non_null_count}, Null: {null_count}")

        # Get unique values (limit to reasonable number for display)
        unique_vals = gdf[col].dropna().unique()
        unique_count = len(unique_vals)

        print(f"  Unique values: {unique_count}")

        # Display unique values (limit to first 10 for readability)
        if unique_count == 0:
            print("  Values: [All null]")
        elif unique_count <= 10:
            print(f"  Values: {unique_vals.tolist()}")
        else:
            print(f"  Values (first 10): {unique_vals[:10].tolist()}")
            print(f"  ... and {unique_count - 10} more unique values")

    print("\n--- First 5 Rows ---")
    print(gdf.head())

    return gdf


# @task
# def view_data(rand_inf:List[ViewState,LayerDefinition])->str:
#    print("Data: {rand_inf}")

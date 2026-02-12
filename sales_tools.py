import pandas as pd
import duckdb
from pydantic import BaseModel, Field
from langsmith import traceable

from settings import (
    MODEL,
    TRANSACTION_DATA_FILE_PATH,
    SQL_GENERATION_PROMPT,
    DATA_ANALYSIS_PROMPT,
    CHART_CONFIGURATION_PROMPT,
    CREATE_CHART_PROMPT,
    client,
)


@traceable
def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate an SQL query based on a prompt."""
    formatted_prompt = SQL_GENERATION_PROMPT.format(
        prompt=prompt,
        columns=", ".join(columns),
        table_name=table_name,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    return response.choices[0].message.content


@traceable
def lookup_sales_data(prompt: str) -> str:
    """Implementation of sales data lookup from parquet file using SQL."""
    try:
        # Define the table name
        table_name = "sales"

        # Step 1: read the parquet file into a DuckDB table
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

        # Step 2: generate the SQL code
        sql_query = generate_sql_query(prompt, df.columns.tolist(), table_name)

        # Clean the response to make sure it only includes the SQL code
        sql_query = sql_query.strip()
        sql_query = sql_query.replace("```sql", "")

        # Step 3: execute the SQL query
        result = duckdb.sql(sql_query).df()

        return result.to_string()

    except Exception as e:
        return (
            f"Error accessing data: {str(e)}\n"
            f"Attempted SQL: {sql_query if 'sql_query' in locals() else 'N/A'}"
        )


@traceable
def analyze_sales_data(prompt: str, data: str) -> str:
    """Implementation of AI-powered sales data analysis."""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    analysis = response.choices[0].message.content
    return analysis if analysis else "No analysis could be generated"


class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")


@traceable
def extract_chart_config(data: str, visualization_goal: str) -> dict:
    """Generate chart visualization configuration."""
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
        data=data,
        visualization_goal=visualization_goal,
    )

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        response_format=VisualizationConfig,
    )

    try:
        # Extract axis and title info from response
        content = response.choices[0].message.content

        # Return structured chart config
        return {
            "chart_type": content.chart_type,
            "x_axis": content.x_axis,
            "y_axis": content.y_axis,
            "title": content.title,
            "data": data,
        }
    except Exception:
        return {
            "chart_type": "line",
            "x_axis": "date",
            "y_axis": "value",
            "title": visualization_goal,
            "data": data,
        }


@traceable
def create_chart(config: dict) -> str:
    """Create a chart based on the configuration."""
    formatted_prompt = CREATE_CHART_PROMPT.format(config=config)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    code = response.choices[0].message.content
    code = code.replace("```python", "").replace("```", "")
    code = code.strip()

    return code


def generate_visualization(data: str, visualization_goal: str) -> str:
    """Generate a visualization based on the data and goal."""
    config = extract_chart_config(data, visualization_goal)
    code = create_chart(config)
    return code


from openai import OpenAI

from helper import get_openai_api_key

# Model configuration
MODEL = "gpt-4o-mini"

# Path to the transactional data
TRANSACTION_DATA_FILE_PATH = "/content/Store_Sales_Price_Elasticity_Promotions_Data.parquet"

# Initialize the OpenAI client
openai_api_key = get_openai_api_key()
client = OpenAI(api_key=openai_api_key)


# Prompt templates
SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
Do NOT include any explanations, markdown formatting, or JSON.
Do NOT wrap the query in code blocks or backticks.
Return ONLY raw SQL code.

Task: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""

DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}

Provide a clear, concise analysis addressing the question.
"""

CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""

CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
config: {config}
"""


SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the Store Sales Price Elasticity Promotions dataset.
You have access to the following tools:
1. lookup_sales_data: Query the sales database
2. analyze_sales_data: Analyze sales data to extract insights
3. generate_visualization: Generate Python code for visualizations

Use these tools to help answer user questions.
"""


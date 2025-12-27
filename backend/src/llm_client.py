"""Mistral AI API client for natural language processing of data queries."""

import json
import logging
import time
from typing import Dict, Any, Optional, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralClient:
    """Client for interacting with Mistral AI API."""
    
    def __init__(self):
        """Initialize the Mistral client."""
        self.api_key = settings.MISTRAL_API_KEY
        self.api_url = settings.MISTRAL_API_URL
        self.model = settings.MISTRAL_MODEL
        self.timeout = settings.REQUEST_TIMEOUT
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=settings.MAX_RETRIES,
            backoff_factor=settings.RETRY_DELAY,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def create_chat_prompt(self, columns: List[str], user_query: str, sample_data) -> str:
        """Create a prompt for conversational responses about the automobile dataset."""
        columns_str = ", ".join(columns)
        
        # Convert sample data to string for context
        sample_str = ""
        if sample_data is not None and len(sample_data) > 0:
            sample_str = "\nSample data:\n" + sample_data.to_string(index=False)
        
        prompt = f"""You are a friendly and knowledgeable assistant for an AUTOMOBILE DATASET.

Dataset Information:
- This dataset contains information about {len(columns)} different aspects of automobiles
- Available columns: {columns_str}
{sample_str}

User Query: "{user_query}"

Instructions:
1. If the query is related to automobiles, cars, vehicles, or this dataset, provide a helpful, conversational response.
2. You can explain data insights, answer questions about car specifications, pricing trends, etc.
3. If the user asks about specific data (like "show me BMW cars"), explain what information is available but suggest they use the dataset viewer or create charts for detailed data.
4. If the query is completely unrelated to automobiles, politely redirect them to ask about cars/vehicles.
5. Keep responses friendly, informative, and focused on the automobile domain.
6. Suggest using charts when appropriate: "You might want to create a [chart type] to visualize this data."

Respond in a conversational, helpful manner. No JSON format needed - just natural text."""
        
        return prompt
    
    def generate_chat_response(self, columns: List[str], user_query: str, sample_data=None) -> Dict[str, Any]:
        """
        Generate a conversational response about the automobile dataset.
        
        Args:
            columns: List of available dataset columns
            user_query: Natural language query from user
            sample_data: Sample data for context
            
        Returns:
            Dict containing response or error message
        """
        if not self.api_key:
            return {"error": "Mistral API key not configured"}
            
        try:
            prompt = self.create_chat_prompt(columns, user_query, sample_data)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specializing in automobile data analysis. Provide conversational, informative responses."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            logger.info(f"Sending chat request to Mistral API: {user_query}")
            
            response = self.session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"].strip()
                    
                    return {
                        "response": content
                    }
                else:
                    return {"error": "No response received from Mistral AI"}
            else:
                logger.error(f"Mistral API error: {response.status_code}")
                return {"error": f"Mistral AI request failed: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            logger.error("Mistral API request timed out")
            return {"error": "Request timed out. Please try again."}
        except requests.exceptions.RequestException as e:
            logger.error(f"Mistral API request failed: {str(e)}")
            return {"error": f"API request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Mistral response: {str(e)}")
            return {"error": "Invalid response format from AI"}
        except Exception as e:
            logger.error(f"Unexpected error in chat generation: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}

    def create_analysis_prompt(self, columns: List[str], user_query: str) -> str:
        """Create a structured prompt for data analysis."""
        columns_str = ", ".join(columns)
        
        prompt = f"""You are a Data Analysis and Visualization Assistant for an AUTOMOBILE DATASET ONLY.

You are given:
1. A dataset containing automobile information. The columns include:
{columns_str}

2. A natural language query from the user: "{user_query}"

IMPORTANT: This dataset is ONLY about automobiles/cars. If the user's query is about anything else (weather, sports, food, politics, general questions, etc.), you MUST respond with an error.

Your task:
1. FIRST: Check if the query is related to automobiles, cars, vehicles, or data analysis of this automotive dataset.
2. If the query is NOT about automobiles/cars/vehicles, respond with:
{{"error": "This query appears to be unrelated to automobile data. Please ask questions about cars, vehicle specifications, pricing, fuel efficiency, or other automotive topics."}}

3. If the query IS about automobiles, then:
- Understand the query.
- Determine the best chart type (choose from: bar, scatter, histogram, line, pie, box, violin, area, heatmap).
- Determine which columns should be used as x-axis and y-axis.
- If the query requires aggregation (e.g., average, sum), include that in the output.
- If the query mentions sorting (e.g., "sorted by", "order by"), include the sort column and direction.
- If the query asks for specific columns only (e.g., "show brand only"), specify which columns to display.
- Always respond in **valid JSON** format, with no additional text, following this structure:

{{
  "chart": "bar",
  "x": "brand",
  "y": "price",
  "agg": "mean",
  "sort_by": "engine_size",
  "sort_order": "desc",
  "columns_only": ["brand"]
}}

Important data type rules:
- "chart", "x", "y", "agg", "sort_by", "sort_order" must be strings (not arrays)
- "columns_only" must be an array of strings
- Use null for optional fields, not empty strings

Rules:
- Only use column names that exist in the dataset: {columns_str}
- "agg" can be: mean, sum, count, median, or null if not needed.
- "sort_by" should be a column name, "sort_order" can be "asc" or "desc".
- "columns_only" should be an array of column names when user asks for specific columns only.
- For histogram charts, only specify "x" column, "y" should be null.
- For scatter plots, both "x" and "y" are required.
- If the automobile query is unclear or cannot be fulfilled, respond with:
{{"error": "Cannot determine appropriate chart type or columns for this automobile data query"}}

Examples of VALID automobile queries:
- "Show average price by brand" → {{"chart": "bar", "x": "brand", "y": "price", "agg": "mean"}}
- "All brands sorted by engine size" → {{"chart": "bar", "x": "brand", "y": "engine_size", "sort_by": "engine_size", "sort_order": "desc"}}
- "Show brand only" → {{"chart": "bar", "x": "brand", "columns_only": ["brand"]}}
- "Plot fuel efficiency vs engine size" → {{"chart": "scatter", "x": "engine_size", "y": "city_mpg"}}
- "Display horsepower distribution" → {{"chart": "histogram", "x": "horsepower"}}
- "Compare sedan vs hatchback prices" → {{"chart": "bar", "x": "body_style", "y": "price", "agg": "mean"}}

Examples of INVALID (unrelated) queries:
- "What's the weather today?"
- "Tell me a joke"
- "How to cook pasta?"
- "What's the capital of France?"
- "Explain quantum physics"

Query: "{user_query}"

Respond with JSON only:"""
        
        return prompt
    
    def analyze_query(self, columns: List[str], user_query: str) -> Dict[str, Any]:
        """
        Analyze user query and return chart specifications.
        
        Args:
            columns: List of available dataset columns
            user_query: Natural language query from user
            
        Returns:
            Dict containing chart specifications or error message
        """
        if not self.api_key:
            return {"error": "Mistral API key not configured"}
            
        try:
            prompt = self.create_analysis_prompt(columns, user_query)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a data analysis assistant. Always respond with valid JSON only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.1
            }
            
            logger.info(f"Sending request to Mistral API: {user_query}")
            
            response = self.session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"].strip()
                    
                    # Try to parse JSON response
                    try:
                        # Clean the response - remove markdown code blocks if present
                        content = content.strip()
                        if content.startswith('```json'):
                            content = content.replace('```json', '').replace('```', '').strip()
                        elif content.startswith('```'):
                            content = content.replace('```', '').strip()
                        
                        analysis_result = json.loads(content)
                        logger.info(f"Successfully parsed analysis: {analysis_result}")
                        return self._validate_analysis_result(analysis_result, columns)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {content}")
                        return {"error": f"Invalid JSON response from LLM: {str(e)}"}
                else:
                    return {"error": "No response from LLM"}
                    
            elif response.status_code == 401:
                return {"error": "Invalid API key"}
            elif response.status_code == 429:
                return {"error": "Rate limit exceeded. Please try again later."}
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {"error": f"API request failed: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            return {"error": "Request timeout. Please try again."}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection error. Please check your internet connection."}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}
    
    def _validate_analysis_result(self, result: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """
        Validate the analysis result from LLM.
        
        Args:
            result: Analysis result from LLM
            columns: Available dataset columns
            
        Returns:
            Validated result or error message
        """
        if "error" in result:
            return result
            
        # Fix common LLM response format issues
        # Convert arrays to strings for single-value fields
        for field in ["chart", "x", "y", "agg", "sort_by", "sort_order"]:
            if field in result and isinstance(result[field], list):
                if len(result[field]) > 0:
                    result[field] = result[field][0]
                else:
                    result[field] = None
        
        # Ensure columns_only is always an array
        if "columns_only" in result and isinstance(result["columns_only"], str):
            result["columns_only"] = [result["columns_only"]]
            
        # Validate required fields
        if "chart" not in result:
            return {"error": "Missing chart type in response"}
            
        valid_charts = ["bar", "scatter", "histogram", "line", "pie", "box", "violin", "area", "heatmap"]
        if result["chart"] not in valid_charts:
            return {"error": f"Invalid chart type: {result['chart']}"}
            
        # Validate columns exist in dataset
        if "x" in result and result["x"] and result["x"] not in columns:
            return {"error": f"Column '{result['x']}' not found in dataset"}
            
        if "y" in result and result["y"] and result["y"] not in columns:
            return {"error": f"Column '{result['y']}' not found in dataset"}
            
        # Validate aggregation method
        if "agg" in result and result["agg"]:
            valid_agg = ["mean", "sum", "count", "median", "max", "min"]
            if result["agg"] not in valid_agg:
                return {"error": f"Invalid aggregation method: {result['agg']}"}
        
        # Validate sort column
        if "sort_by" in result and result["sort_by"]:
            sort_col = result["sort_by"]
            # Handle case where sort_by might be a list
            if isinstance(sort_col, list):
                if sort_col:
                    result["sort_by"] = sort_col[0]  # Take first column
                    sort_col = sort_col[0]
                else:
                    result["sort_by"] = None
                    sort_col = None
            
            if sort_col and sort_col not in columns:
                return {"error": f"Sort column '{sort_col}' not found in dataset"}
        
        # Validate sort order
        if "sort_order" in result and result["sort_order"]:
            valid_orders = ["asc", "desc", "ascending", "descending"]
            if result["sort_order"].lower() not in valid_orders:
                result["sort_order"] = "asc"  # Default to ascending
        
        # Validate columns_only
        if "columns_only" in result and result["columns_only"]:
            if not isinstance(result["columns_only"], list):
                return {"error": "columns_only must be a list"}
            # Check if requested columns exist
            invalid_cols = [col for col in result["columns_only"] if col not in columns]
            if invalid_cols:
                return {"error": f"Columns not found: {invalid_cols}"}
        
        return result

# Global client instance
mistral_client = MistralClient()
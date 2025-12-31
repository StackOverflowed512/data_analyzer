"""Chart generation module for creating visualizations based on LLM analysis."""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
# import seaborn as sns  # Temporarily disabled due to scipy dependency issues
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('default')  # Changed from seaborn style
# sns.set_palette("husl")  # Temporarily disabled

class ChartGenerator:
    """Generates charts based on LLM analysis results."""
    
    def __init__(self):
        """Initialize the chart generator."""
        self.figure_size = settings.DEFAULT_CHART_SIZE
        self.dpi = settings.DEFAULT_DPI
        
    def generate_chart(self, 
                      data: pd.DataFrame, 
                      chart_specs: Dict[str, Any], 
                      library: str = "matplotlib",
                      save_path: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
        """
        Generate a chart based on specifications.
        
        Args:
            data: Processed dataframe
            chart_specs: Chart specifications from LLM
            library: Chart library to use ("matplotlib" or "plotly")
            save_path: Optional path to save the chart
            
        Returns:
            Tuple of (chart_object, error_message)
        """
        try:
            chart_type = chart_specs.get("chart")
            x_col = chart_specs.get("x")
            y_col = chart_specs.get("y")
            
            if library == "matplotlib":
                return self._generate_matplotlib_chart(data, chart_type, x_col, y_col, save_path)
            elif library == "plotly":
                return self._generate_plotly_chart(data, chart_type, x_col, y_col, save_path)
            else:
                return None, f"Unsupported chart library: {library}"
                
        except Exception as e:
            logger.error(f"Chart generation failed: {str(e)}")
            return None, f"Chart generation error: {str(e)}"
    
    def _generate_matplotlib_chart(self, 
                                 data: pd.DataFrame, 
                                 chart_type: str, 
                                 x_col: str, 
                                 y_col: Optional[str],
                                 save_path: Optional[str] = None) -> Tuple[Optional[plt.Figure], Optional[str]]:
        """Generate chart using matplotlib."""
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            if chart_type == "bar":
                if y_col:
                    ax.bar(data[x_col], data[y_col])
                    ax.set_ylabel(y_col.replace("_", " ").title())
                else:
                    value_counts = data[x_col].value_counts()
                    ax.bar(value_counts.index, value_counts.values)
                    ax.set_ylabel("Count")
                
                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.tick_params(axis='x', rotation=45)
                
            elif chart_type == "pie":
                if y_col:
                    # Group by x_col and sum y_col values
                    pie_data = data.groupby(x_col)[y_col].sum()
                else:
                    # Use value counts for categorical data
                    pie_data = data[x_col].value_counts()
                
                wedges, texts, autotexts = ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                ax.set_aspect('equal')
                
            elif chart_type == "scatter":
                if not y_col:
                    return None, "Scatter plot requires both x and y columns"
                
                ax.scatter(data[x_col], data[y_col], alpha=0.6)
                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.set_ylabel(y_col.replace("_", " ").title())
                
            elif chart_type == "histogram":
                ax.hist(data[x_col], bins=20, alpha=0.7, edgecolor='black')
                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.set_ylabel("Frequency")
                
            elif chart_type == "line":
                if not y_col:
                    return None, "Line plot requires both x and y columns"
                
                # Sort data by x column for proper line plotting
                sorted_data = data.sort_values(x_col)
                ax.plot(sorted_data[x_col], sorted_data[y_col], marker='o')
                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.set_ylabel(y_col.replace("_", " ").title())
                
            elif chart_type == "box" or chart_type == "boxplot":
                if y_col:
                    # Box plot for continuous variable grouped by categorical
                    unique_categories = data[x_col].unique()
                    box_data = [data[data[x_col] == cat][y_col].dropna() for cat in unique_categories]
                    ax.boxplot(box_data, labels=unique_categories)
                    ax.set_xlabel(x_col.replace("_", " ").title())
                    ax.set_ylabel(y_col.replace("_", " ").title())
                    ax.tick_params(axis='x', rotation=45)
                else:
                    # Single box plot for one variable
                    ax.boxplot(data[x_col].dropna())
                    ax.set_ylabel(x_col.replace("_", " ").title())
                    ax.set_xticklabels([x_col.replace("_", " ").title()])
                    
            elif chart_type == "violin":
                if y_col:
                    unique_categories = data[x_col].unique()
                    violin_data = [data[data[x_col] == cat][y_col].dropna() for cat in unique_categories]
                    parts = ax.violinplot(violin_data, positions=range(1, len(unique_categories) + 1))
                    ax.set_xticks(range(1, len(unique_categories) + 1))
                    ax.set_xticklabels(unique_categories, rotation=45)
                    ax.set_xlabel(x_col.replace("_", " ").title())
                    ax.set_ylabel(y_col.replace("_", " ").title())
                else:
                    return None, "Violin plot requires both x and y columns"
                    
            elif chart_type == "area":
                if not y_col:
                    return None, "Area plot requires both x and y columns"
                
                sorted_data = data.sort_values(x_col)
                ax.fill_between(sorted_data[x_col], sorted_data[y_col], alpha=0.7)
                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.set_ylabel(y_col.replace("_", " ").title())
                
            elif chart_type == "heatmap":
                # Create correlation heatmap for numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    return None, "Heatmap requires at least 2 numeric columns"
                
                correlation_matrix = data[numeric_cols].corr()
                im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                ax.set_xticks(range(len(correlation_matrix.columns)))
                ax.set_yticks(range(len(correlation_matrix.columns)))
                ax.set_xticklabels(correlation_matrix.columns, rotation=45)
                ax.set_yticklabels(correlation_matrix.columns)
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
                
                # Add correlation values to cells
                for i in range(len(correlation_matrix.columns)):
                    for j in range(len(correlation_matrix.columns)):
                        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black")
                
            else:
                return None, f"Unsupported chart type: {chart_type}"
            
            # Set title
            title = self._generate_chart_title(chart_type, x_col, y_col)
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Improve layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"Chart saved to: {save_path}")
            
            return fig, None
            
        except Exception as e:
            logger.error(f"Matplotlib chart generation failed: {str(e)}")
            return None, str(e)
    
    def _generate_plotly_chart(self, 
                             data: pd.DataFrame, 
                             chart_type: str, 
                             x_col: str, 
                             y_col: Optional[str],
                             save_path: Optional[str] = None) -> Tuple[Optional[go.Figure], Optional[str]]:
        """Generate chart using plotly."""
        try:
            if chart_type == "bar":
                if y_col:
                    fig = px.bar(data, x=x_col, y=y_col)
                else:
                    value_counts = data[x_col].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values)
                    fig.update_layout(yaxis_title="Count")
                    
            elif chart_type == "pie":
                if y_col:
                    # Group by x_col and sum y_col values
                    pie_data = data.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.pie(pie_data, values=y_col, names=x_col)
                else:
                    # Use value counts for categorical data
                    value_counts = data[x_col].value_counts().reset_index()
                    value_counts.columns = [x_col, 'count']
                    fig = px.pie(value_counts, values='count', names=x_col)
                    
            elif chart_type == "scatter":
                if not y_col:
                    return None, "Scatter plot requires both x and y columns"
                
                fig = px.scatter(data, x=x_col, y=y_col)
                
            elif chart_type == "histogram":
                fig = px.histogram(data, x=x_col, nbins=20)
                
            elif chart_type == "line":
                if not y_col:
                    return None, "Line plot requires both x and y columns"
                
                # Sort data by x column for proper line plotting
                sorted_data = data.sort_values(x_col)
                fig = px.line(sorted_data, x=x_col, y=y_col, markers=True)
                
            elif chart_type == "box" or chart_type == "boxplot":
                if y_col:
                    fig = px.box(data, x=x_col, y=y_col)
                else:
                    fig = px.box(data, y=x_col)
                    
            elif chart_type == "violin":
                # Relaxed validation: Allow violin with just x or y (distribution)
                if y_col:
                    fig = px.violin(data, x=x_col if x_col else None, y=y_col, box=True)
                elif x_col:
                    fig = px.violin(data, x=x_col, box=True, orientation='h')
                else:
                    return None, "Violin plot requires at least one column (x or y)"
                
            elif chart_type == "area":
                if not y_col:
                    # If only x is present, we can plot counts if it is categorical or histogram-like
                    # Or treat it as a line chart of counts
                    if x_col:
                        # Assuming we want area under the curve of counts?
                        # Or if the user asked for area chart of a single variable, they might mean distribution
                        # Let's try to plot freq/value counts as area
                       value_counts = data[x_col].value_counts().sort_index()
                       fig = px.area(x=value_counts.index, y=value_counts.values)
                       fig.update_layout(xaxis_title=x_col, yaxis_title="Count")
                    else:
                        return None, "Area plot requires at least x column"
                else:
                    sorted_data = data.sort_values(x_col)
                    fig = px.area(sorted_data, x=x_col, y=y_col)
                
            elif chart_type == "heatmap":
                # Create correlation heatmap for numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    return None, "Heatmap requires at least 2 numeric columns"
                
                correlation_matrix = data[numeric_cols].corr()
                fig = px.imshow(correlation_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              color_continuous_scale='RdBu_r')
                              
            elif chart_type == "sunburst":
                if not y_col:
                    return None, "Sunburst chart requires both x and y columns"
                fig = px.sunburst(data, path=[x_col], values=y_col)
                
            elif chart_type == "treemap":
                if not y_col:
                    return None, "Treemap requires both x and y columns"
                fig = px.treemap(data, path=[x_col], values=y_col)
                
            elif chart_type == "funnel":
                if not y_col:
                    return None, "Funnel chart requires both x and y columns"
                fig = px.funnel(data, x=y_col, y=x_col)
                
            elif chart_type == "density":
                if y_col:
                    fig = px.density_contour(data, x=x_col, y=y_col)
                else:
                    fig = px.histogram(data, x=x_col, marginal="rug", histnorm="density")
                    
            else:
                return None, f"Unsupported chart type: {chart_type}"
            
            # Update layout
            title = self._generate_chart_title(chart_type, x_col, y_col)
            fig.update_layout(
                title=title,
                title_font_size=16,
                xaxis_title=x_col.replace("_", " ").title(),
                yaxis_title=y_col.replace("_", " ").title() if y_col else "Count"
            )
            
            # Save if path provided
            if save_path:
                if save_path.endswith('.html'):
                    fig.write_html(save_path)
                else:
                    fig.write_image(save_path)
                logger.info(f"Chart saved to: {save_path}")
            
            return fig, None
            
        except Exception as e:
            logger.error(f"Plotly chart generation failed: {str(e)}")
            return None, str(e)
    
    def _generate_chart_title(self, chart_type: str, x_col: str, y_col: Optional[str]) -> str:
        """Generate appropriate chart title."""
        x_title = x_col.replace("_", " ").title()
        
        if chart_type == "histogram":
            return f"Distribution of {x_title}"
        elif chart_type == "bar":
            if y_col:
                y_title = y_col.replace("_", " ").title()
                return f"{y_title} by {x_title}"
            else:
                return f"Count by {x_title}"
        elif chart_type == "scatter":
            y_title = y_col.replace("_", " ").title() if y_col else "Y"
            return f"{y_title} vs {x_title}"
        elif chart_type == "line":
            y_title = y_col.replace("_", " ").title() if y_col else "Y"
            return f"{y_title} vs {x_title}"
        else:
            return f"{chart_type.title()} Chart"
    
    def get_supported_chart_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all supported chart types with their descriptions and requirements.
        
        Returns:
            Dictionary of chart types with metadata
        """
        return {
            "bar": {
                "description": "Bar chart for categorical data comparison",
                "requires_x": True,
                "requires_y": False,
                "best_for": "Comparing categories, showing counts or values"
            },
            "pie": {
                "description": "Pie chart for showing proportions",
                "requires_x": True,
                "requires_y": False,
                "best_for": "Showing parts of a whole, proportions"
            },
            "line": {
                "description": "Line chart for trends over time",
                "requires_x": True,
                "requires_y": True,
                "best_for": "Showing trends, time series data"
            },
            "scatter": {
                "description": "Scatter plot for correlation analysis",
                "requires_x": True,
                "requires_y": True,
                "best_for": "Showing relationships between two variables"
            },
            "histogram": {
                "description": "Histogram for distribution analysis",
                "requires_x": True,
                "requires_y": False,
                "best_for": "Showing data distribution, frequency analysis"
            },
            "box": {
                "description": "Box plot for statistical summary",
                "requires_x": True,
                "requires_y": False,
                "best_for": "Showing quartiles, outliers, data spread"
            },
            "violin": {
                "description": "Violin plot for distribution shape",
                "requires_x": True,
                "requires_y": True,
                "best_for": "Showing distribution shape and density"
            },
            "area": {
                "description": "Area chart for cumulative data",
                "requires_x": True,
                "requires_y": True,
                "best_for": "Showing cumulative values over time"
            },
            "heatmap": {
                "description": "Heatmap for correlation matrix",
                "requires_x": False,
                "requires_y": False,
                "best_for": "Showing correlations between multiple variables"
            },
            "sunburst": {
                "description": "Sunburst chart for hierarchical data",
                "requires_x": True,
                "requires_y": True,
                "best_for": "Showing hierarchical proportions"
            },
            "treemap": {
                "description": "Treemap for hierarchical data with rectangles",
                "requires_x": True,
                "requires_y": True,
                "best_for": "Showing hierarchical data with size proportions"
            },
            "funnel": {
                "description": "Funnel chart for process analysis",
                "requires_x": True,
                "requires_y": True,
                "best_for": "Showing conversion rates, process flows"
            },
            "density": {
                "description": "Density plot for continuous distributions",
                "requires_x": True,
                "requires_y": False,
                "best_for": "Showing probability density of continuous data"
            }
        }

    def display_chart(self, fig, library: str = "matplotlib"):
        """
        Display the chart.
        
        Args:
            fig: Chart figure object
            library: Chart library used
        """
        try:
            if library == "matplotlib":
                plt.show()
            elif library == "plotly":
                fig.show()
                
        except Exception as e:
            logger.error(f"Failed to display chart: {str(e)}")
    
    def get_chart_summary(self, data: pd.DataFrame, chart_specs: Dict[str, Any]) -> str:
        """
        Generate a summary of the chart data.
        
        Args:
            data: Chart data
            chart_specs: Chart specifications
            
        Returns:
            Text summary of the chart
        """
        try:
            chart_type = chart_specs.get("chart")
            x_col = chart_specs.get("x")
            y_col = chart_specs.get("y")
            
            summary = f"Generated {chart_type} chart with {len(data)} data points.\n"
            
            if x_col:
                summary += f"X-axis: {x_col.replace('_', ' ').title()}\n"
                if data[x_col].dtype == 'object':
                    unique_values = data[x_col].nunique()
                    summary += f"  - {unique_values} unique categories\n"
                else:
                    summary += f"  - Range: {data[x_col].min():.2f} to {data[x_col].max():.2f}\n"
            
            if y_col and y_col in data.columns:
                summary += f"Y-axis: {y_col.replace('_', ' ').title()}\n"
                if data[y_col].dtype in ['int64', 'float64']:
                    summary += f"  - Range: {data[y_col].min():.2f} to {data[y_col].max():.2f}\n"
                    summary += f"  - Average: {data[y_col].mean():.2f}\n"
            
            return summary
            
        except Exception as e:
            return f"Could not generate chart summary: {str(e)}"

# Global chart generator instance
chart_generator = ChartGenerator()

"""
Flask API backend for automobile data analysis with Mistral AI.
"""

import sys
import json
import base64
import io
import os
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.utils
from werkzeug.utils import secure_filename

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_processor import data_processor
from src.llm_client import mistral_client
from src.chart_generator import chart_generator
from config.settings import settings

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - Allow specific origins
# Temporary - allow all origins for testing
CORS(app, origins="*", methods=['GET', 'POST', 'OPTIONS'], allow_headers=['Content-Type'])

# Global variables
app_initialized = False

# Upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_app():
    """Initialize the application once."""
    global app_initialized
    
    if app_initialized:
        return True
    
    try:
        # Validate settings
        validation = settings.validate_settings()
        if not validation["valid"]:
            return False
        
        # Load dataset
        if not data_processor.load_dataset():
            return False
        
        app_initialized = True
        return True
    
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        "name": "Automobile Data Analysis API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "initialize": "/api/initialize",
            "dataset_info": "/api/dataset-info",
            "dataset": "/api/dataset",
            "dataset_search": "/api/dataset/search",
            "dataset_filter": "/api/dataset/filter",
            "chart_types": "/api/chart-types",
            "analyze": "/api/analyze",
            "download_data": "/api/download-data",
            "example_queries": "/api/example-queries"
        },
        "features": [
            "View dataset in tabular format",
            "Search and filter data",
            "Generate multiple chart types",
            "Smart data analysis with AI",
            "Export data and charts"
        ],
        "description": "Enhanced API for analyzing automobile data with comprehensive chart support"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint that also helps wake up sleeping services."""
    from datetime import datetime
    return jsonify({
        "status": "healthy",
        "message": "Automobile Data Analysis API is running",
        "timestamp": str(datetime.now()),
        "initialized": app_initialized
    })

@app.route('/api/wake', methods=['GET', 'POST'])
def wake_up():
    """Simple wake-up endpoint for deployment services that go to sleep."""
    from datetime import datetime
    try:
        # Force initialization if not already done
        if not app_initialized:
            initialize_app()
        
        return jsonify({
            "status": "awake",
            "message": "Backend is fully operational",
            "timestamp": str(datetime.now()),
            "initialized": app_initialized
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Wake-up failed: {str(e)}",
            "timestamp": str(datetime.now())
        }), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check environment variables."""
    import os
    return jsonify({
        "env_vars_present": {
            "MISTRAL_API_KEY": bool(os.getenv('MISTRAL_API_KEY')),
            "MISTRAL_API_URL": bool(os.getenv('MISTRAL_API_URL')),
            "MISTRAL_MODEL": bool(os.getenv('MISTRAL_MODEL'))
        },
        "settings_validation": settings.validate_settings(),
        "app_initialized": app_initialized
    })

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the application."""
    try:
        print("Initialize endpoint called")
        if initialize_app():
            print("App initialized successfully")
            return jsonify({
                "success": True,
                "message": "Application initialized successfully"
            })
        else:
            print("App initialization failed")
            return jsonify({
                "success": False,
                "error": "Failed to initialize application. Check API key and dataset."
            }), 500
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Initialization error: {str(e)}"
        }), 500

@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    """Get dataset information."""
    if not app_initialized:
        return jsonify({"error": "Application not initialized"}), 400
    
    try:
        info = data_processor.get_dataset_info()
        preview = data_processor.preview_data(10)
        column_stats = data_processor.get_column_statistics()
        
        return jsonify({
            "success": True,
            "info": info,
            "preview": data_processor.to_json_safe(preview),
            "column_statistics": column_stats
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to get dataset info: {str(e)}"
        }), 500

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    """Get paginated dataset for viewing."""
    if not app_initialized:
        return jsonify({"error": "Application not initialized"}), 400
    
    try:
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        dataset_result = data_processor.get_full_dataset(page, page_size)
        
        return jsonify({
            "success": True,
            **dataset_result
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to get dataset: {str(e)}"
        }), 500

@app.route('/api/dataset/search', methods=['POST'])
def search_dataset():
    """Search dataset for specific terms."""
    if not app_initialized:
        return jsonify({"error": "Application not initialized"}), 400
    
    try:
        data = request.get_json()
        search_term = data.get('search_term', '').strip()
        columns = data.get('columns', None)
        
        if not search_term:
            return jsonify({"error": "Search term is required"}), 400
        
        result_df, error = data_processor.search_data(search_term, columns)
        
        if error:
            return jsonify({"success": False, "error": error}), 400
        
        return jsonify({
            "success": True,
            "results": data_processor.to_json_safe(result_df),
            "count": len(result_df) if result_df is not None else 0
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Search failed: {str(e)}"
        }), 500

@app.route('/api/dataset/filter', methods=['POST'])
def filter_dataset():
    """Filter dataset based on criteria."""
    if not app_initialized:
        return jsonify({"error": "Application not initialized"}), 400
    
    try:
        data = request.get_json()
        filters = data.get('filters', {})
        
        result_df, error = data_processor.filter_data(filters)
        
        if error:
            return jsonify({"success": False, "error": error}), 400
        
        return jsonify({
            "success": True,
            "results": data_processor.to_json_safe(result_df),
            "count": len(result_df) if result_df is not None else 0
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Filter failed: {str(e)}"
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat_query():
    """Handle general chat queries about the automobile dataset."""
    if not app_initialized:
        return jsonify({"error": "Application not initialized"}), 400
    
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        print(f"💬 Chat query: {user_query}")
        
        if not user_query:
            return jsonify({"error": "Query is required"}), 400
        
        # Get columns for context
        columns = data_processor.get_columns()
        
        # Get a small data sample for context
        sample_data = data_processor.preview_data(3)
        
        # Generate conversational response
        chat_response = mistral_client.generate_chat_response(columns, user_query, sample_data)
        
        if "error" in chat_response:
            return jsonify({
                "success": False,
                "error": chat_response["error"]
            }), 400
        
        return jsonify({
            "success": True,
            "response": chat_response["response"],
            "query": user_query
        })
    
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Chat error: {str(e)}"
        }), 500

@app.route('/api/chart-types', methods=['GET'])
def get_chart_types():
    """Get all supported chart types."""
    try:
        chart_types = chart_generator.get_supported_chart_types()
        
        return jsonify({
            "success": True,
            "chart_types": chart_types
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to get chart types: {str(e)}"
        }), 500

@app.route('/api/test-query', methods=['POST'])
def test_query():
    """Test endpoint for debugging query processing."""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        print(f"🧪 Test query: {user_query}")
        
        # Test basic functionality step by step
        if not app_initialized:
            return jsonify({"error": "Application not initialized", "step": "initialization"})
        
        # Test data processor
        columns = data_processor.get_columns()
        if not columns:
            return jsonify({"error": "No columns found", "step": "data_loading"})
        
        return jsonify({
            "success": True,
            "query": user_query,
            "columns": columns,
            "dataset_shape": [len(data_processor.df), len(columns)] if data_processor.df is not None else None,
            "mistral_api_key_set": bool(settings.MISTRAL_API_KEY)
        })
        
    except Exception as e:
        print(f"❌ Test query error: {str(e)}")
        return jsonify({"error": str(e), "step": "unknown"}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_query():
    """Analyze user query and return data/chart based on the request."""
    if not app_initialized:
        return jsonify({"error": "Application not initialized"}), 400
    
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        chart_library = data.get('library', 'plotly')
        force_chart = data.get('force_chart', False)
        requested_chart_type = data.get('chart_type', None)  # Add this line
        
        print(f"🔍 Analyzing query: {user_query}")
        print(f"📊 Chart library: {chart_library}")
        print(f"🎯 Force chart: {force_chart}")
        print(f"📈 Requested chart type: {requested_chart_type}")  # Add this line
        
        if not user_query:
            return jsonify({"error": "Query is required"}), 400
        
        # Get columns
        columns = data_processor.get_columns()
        print(f"📋 Available columns: {columns}")
        
        # Check if query explicitly asks for a chart
        chart_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualization', 'bar', 'pie', 'line', 'scatter', 'histogram', 'box', 'heatmap', 'violin', 'area']
        needs_chart = force_chart or any(keyword in user_query.lower() for keyword in chart_keywords)
        
        print(f"📊 Needs chart: {needs_chart}")
        
        # Analyze with Mistral AI
        print("🤖 Calling Mistral AI...")
        analysis_result = mistral_client.analyze_query(columns, user_query)
        print(f"🤖 Mistral response: {analysis_result}")
        
        if "error" in analysis_result:
            print(f"❌ Mistral error: {analysis_result['error']}")
            
            # Check if it's an unrelated query error vs other errors
            error_message = analysis_result["error"]
            if ("unrelated to automobile" in error_message.lower() or 
                "not about automobiles" in error_message.lower() or
                "automobile data query" in error_message.lower()):
                # Return 200 for unrelated queries (not a server error)
                return jsonify({
                    "success": False,
                    "error": error_message,
                    "error_type": "unrelated_query"
                }), 200
            else:
                # Return 400 for other types of errors
                return jsonify({
                    "success": False,
                    "error": analysis_result["error"]
                }), 400
        
        # Override chart type if specifically requested
        if requested_chart_type and needs_chart:
            print(f"🎯 Overriding chart type to: {requested_chart_type}")
            analysis_result["chart"] = requested_chart_type
        
        # Process data
        print("📈 Processing data...")
        processed_data, error = data_processor.process_data_for_chart(analysis_result)
        
        if error:
            print(f"❌ Data processing error: {error}")
            return jsonify({
                "success": False,
                "error": f"Data processing error: {error}"
            }), 400
        
        response_data = {
            "success": True,
            "analysis": analysis_result,
            "data": data_processor.to_json_safe(processed_data),
            "data_count": len(processed_data),
            "columns": processed_data.columns.tolist(),
            "needs_chart": needs_chart
        }
        
        # Generate chart only if needed
        if needs_chart and analysis_result.get("chart"):
            print("🎨 Generating chart...")
            chart, chart_error = chart_generator.generate_chart(
                processed_data,
                analysis_result,
                library=chart_library
            )
            
            if chart_error:
                print(f"⚠️ Chart generation error: {chart_error}")
                response_data["chart_error"] = chart_error
            else:
                # Prepare chart data based on library
                if chart_library == "plotly":
                    chart_data = json.loads(plotly.utils.PlotlyJSONEncoder().encode(chart))
                    response_data["chart_data"] = chart_data
                elif chart_library == "matplotlib":
                    # Convert matplotlib figure to base64 image
                    img_buffer = io.BytesIO()
                    chart.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                    img_buffer.seek(0)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    response_data["chart_data"] = f"data:image/png;base64,{img_base64}"
                    plt.close(chart)  # Close the figure to free memory
                
                # Get summary
                summary = chart_generator.get_chart_summary(processed_data, analysis_result)
                response_data["chart_summary"] = summary
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Analysis error: {str(e)}"
        }), 500

@app.route('/api/download-data', methods=['POST'])
def download_data():
    """Download processed data as CSV."""
    if not app_initialized:
        return jsonify({"error": "Application not initialized"}), 400
    
    try:
        data = request.get_json()
        processed_data_records = data.get('data', [])
        
        if not processed_data_records:
            return jsonify({"error": "No data to download"}), 400
        
        # Convert back to DataFrame
        import pandas as pd
        df = pd.DataFrame(processed_data_records)
        
        # Create CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        return jsonify({
            "success": True,
            "csv_content": csv_content,
            "filename": "automobile_analysis.csv"
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Download error: {str(e)}"
        }), 500

@app.route('/api/example-queries', methods=['GET'])
def get_example_queries():
    """Get example queries for the user."""
    
    # Try to generate dynamic suggestions based on loaded dataset
    dynamic_suggestions = data_processor.generate_smart_suggestions()
    
    if dynamic_suggestions:
        examples = dynamic_suggestions
    else:
        # Fallback to defaults if no dataset loaded
        examples = [
            {
                "query": "show average price by brand",
                "description": "Compare average car prices across different brands"
            },
            {
                "query": "plot horsepower vs price",
                "description": "Scatter plot showing relationship between horsepower and price"
            },
            {
                "query": "display price distribution histogram",
                "description": "Show distribution of car prices"
            },
            {
                "query": "compare fuel efficiency by fuel type",
                "description": "Compare miles per gallon between gas and diesel cars"
            },
            {
                "query": "show engine size distribution",
                "description": "Display histogram of engine sizes"
            },
            {
                "query": "analyze price by body style",
                "description": "Compare prices across different body styles"
            }
        ]
    
    return jsonify({
        "success": True,
        "examples": examples
    })

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Handle CSV dataset file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Only CSV and XLSX files are allowed"}), 400

        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(filepath)

        # Try to load the dataset
        if not data_processor.load_dataset(filepath):
            return jsonify({
                "success": False,
                "error": "Failed to load dataset. Check CSV format and encoding. Try saving as UTF-8."
            }), 400

        # Get dataset info
        info = data_processor.get_dataset_info()
        preview = data_processor.preview_data(5)

        # Convert preview to JSON-safe format (handle NaN, None, etc.)
        preview_records = []
        if preview is not None:
            import pandas as pd
            import numpy as np
            preview_dict = preview.where(pd.notna(preview), None).to_dict('records')
            # Additional cleanup for any remaining NaN values
            preview_records = [
                {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}
                for row in preview_dict
            ]

        return jsonify({
            "success": True,
            "filename": file.filename,
            "info": info,
            "preview": preview_records
        }), 200

    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List all available datasets."""
    try:
        datasets = data_processor.list_datasets()
        return jsonify({
            "success": True,
            "datasets": datasets,
            "current_dataset_id": data_processor.current_dataset_id
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to list datasets: {str(e)}"
        }), 500

@app.route('/api/dataset/switch', methods=['POST'])
def switch_dataset():
    """Switch to a specific dataset."""
    try:
        data = request.get_json()
        dataset_id = data.get('id')
        
        if not dataset_id:
            return jsonify({"success": False, "error": "Dataset ID required"}), 400
            
        if data_processor.load_dataset_by_id(dataset_id):
            return jsonify({
                "success": True,
                "message": f"Switched to dataset {data_processor.current_dataset_name}",
                "info": data_processor.get_dataset_info()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to load dataset. ID may be invalid."
            }), 404
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Switch failed: {str(e)}"
        }), 500

@app.route('/api/reinitialize', methods=['POST'])
def reinitialize():
    """Reinitialize app with currently loaded dataset."""
    try:
        global app_initialized
        
        print("🔄 Reinitializing application...")
        
        # Validate settings
        validation = settings.validate_settings()
        if not validation["valid"]:
            return jsonify({
                "success": False,
                "error": "Configuration validation failed"
            }), 500
        
        # Check if dataset is loaded
        if data_processor.df is None:
            return jsonify({
                "success": False,
                "error": "No dataset loaded"
            }), 400
        
        app_initialized = True
        
        print("✅ Reinitialization successful")
        return jsonify({
            "success": True,
            "message": "Application reinitialized with new dataset"
        })
    
    except Exception as e:
        print(f"❌ Reinitialization error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Reinitialization failed: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚗 Starting Enhanced Automobile Data Analysis API...")
    print("📝 Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/initialize - Initialize application")
    print("  GET  /api/dataset-info - Get dataset information and statistics")
    print("  GET  /api/dataset - Get paginated dataset for viewing")
    print("  POST /api/dataset/search - Search dataset")
    print("  POST /api/dataset/filter - Filter dataset")
    print("  GET  /api/chart-types - Get all supported chart types")
    print("  POST /api/analyze - Analyze query (data + optional charts)")
    print("  POST /api/download-data - Download processed data")
    print("  GET  /api/example-queries - Get example queries")
    print("\n✨ New Features:")
    print("  📊 All chart types: bar, pie, line, scatter, histogram, box, violin, area, heatmap, etc.")
    print("  📋 Default tabular data display")
    print("  🔍 Dataset viewing and exploration")
    print("  🎯 Charts only when explicitly requested")
    print("  📱 Enhanced data filtering and search")
    
    # Auto-initialize the application
    print("🔧 Auto-initializing application...")
    if initialize_app():
        print("✅ Application initialized successfully!")
    else:
        print("❌ Failed to initialize application. Check API key and dataset.")
    
    # Use environment variable for port (deployment-friendly)
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(debug=debug, host='0.0.0.0', port=port)



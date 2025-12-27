#!/usr/bin/env python3
"""
Main CLI application for automobile data analysis with Mistral AI.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_processor import data_processor
from src.llm_client import mistral_client
from src.chart_generator import chart_generator
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutomobileAnalyzer:
    """Main application class for automobile data analysis."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.data_loaded = False
        self.setup_complete = False
        
    def setup(self) -> bool:
        """
        Setup the application by validating settings and loading data.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Validate settings
            validation = settings.validate_settings()
            if not validation["valid"]:
                logger.error("Configuration validation failed:")
                for error in validation["errors"]:
                    logger.error(f"  - {error}")
                return False
            
            # Load dataset
            if not data_processor.load_dataset():
                logger.error("Failed to load dataset")
                return False
                
            self.data_loaded = True
            self.setup_complete = True
            
            logger.info("Application setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            return False
    
    def analyze_query(self, user_query: str, chart_library: str = "matplotlib") -> bool:
        """
        Analyze user query and generate chart.
        
        Args:
            user_query: Natural language query from user
            chart_library: Chart library to use ("matplotlib" or "plotly")
            
        Returns:
            True if analysis successful, False otherwise
        """
        if not self.setup_complete:
            logger.error("Application not properly setup. Run setup() first.")
            return False
            
        try:
            print(f"\n🔍 Analyzing query: '{user_query}'")
            
            # Get available columns
            columns = data_processor.get_columns()
            
            # Analyze query with Mistral AI
            print("🤖 Processing with Mistral AI...")
            analysis_result = mistral_client.analyze_query(columns, user_query)
            
            if "error" in analysis_result:
                print(f"❌ Error: {analysis_result['error']}")
                return False
            
            print(f"📊 Chart specifications: {analysis_result}")
            
            # Process data based on analysis
            print("📈 Processing data...")
            processed_data, error = data_processor.process_data_for_chart(analysis_result)
            
            if error:
                print(f"❌ Data processing error: {error}")
                return False
            
            print(f"✅ Data processed: {len(processed_data)} rows")
            
            # Generate chart
            print(f"🎨 Generating {chart_library} chart...")
            chart, error = chart_generator.generate_chart(
                processed_data, 
                analysis_result, 
                library=chart_library
            )
            
            if error:
                print(f"❌ Chart generation error: {error}")
                return False
            
            # Display chart
            chart_generator.display_chart(chart, library=chart_library)
            
            # Print summary
            summary = chart_generator.get_chart_summary(processed_data, analysis_result)
            print(f"\n📋 Chart Summary:\n{summary}")
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            print(f"❌ Unexpected error: {str(e)}")
            return False
    
    def show_dataset_info(self):
        """Display information about the loaded dataset."""
        if not self.data_loaded:
            print("❌ Dataset not loaded")
            return
            
        info = data_processor.get_dataset_info()
        print("\n📊 Dataset Information:")
        print(f"  Rows: {info['rows']}")
        print(f"  Columns: {info['columns']}")
        print(f"  Column Names: {', '.join(info['column_names'])}")
        print(f"  Numeric Columns: {', '.join(info['numeric_columns'])}")
        print(f"  Categorical Columns: {', '.join(info['categorical_columns'])}")
        
        # Show preview
        preview = data_processor.preview_data()
        if preview is not None:
            print("\n📋 Dataset Preview:")
            print(preview.to_string(index=False))
    
    def interactive_mode(self):
        """Run the application in interactive mode."""
        print("🚗 Automobile Data Analysis with Mistral AI")
        print("=" * 50)
        
        if not self.setup():
            print("❌ Failed to setup application. Please check your configuration.")
            return
        
        self.show_dataset_info()
        
        print("\n💡 Example queries:")
        print("  - 'show average price by brand'")
        print("  - 'plot horsepower vs price'")
        print("  - 'display price distribution histogram'")
        print("  - 'compare fuel efficiency by fuel type'")
        print("\nType 'quit' to exit, 'info' to show dataset info")
        
        while True:
            try:
                query = input("\n🔍 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'info':
                    self.show_dataset_info()
                    continue
                elif not query:
                    continue
                
                # Ask for chart library preference
                while True:
                    lib_choice = input("📊 Chart library (matplotlib/plotly) [matplotlib]: ").strip().lower()
                    if not lib_choice:
                        lib_choice = "matplotlib"
                    
                    if lib_choice in ["matplotlib", "plotly"]:
                        break
                    else:
                        print("❌ Invalid choice. Please enter 'matplotlib' or 'plotly'")
                
                # Analyze query
                success = self.analyze_query(query, lib_choice)
                
                if success:
                    print("✅ Analysis completed successfully!")
                else:
                    print("❌ Analysis failed. Please try a different query.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")

def main():
    """Main entry point."""
    analyzer = AutomobileAnalyzer()
    
    if len(sys.argv) > 1:
        # Non-interactive mode with command line argument
        query = " ".join(sys.argv[1:])
        
        if not analyzer.setup():
            print("❌ Failed to setup application")
            sys.exit(1)
        
        success = analyzer.analyze_query(query)
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        analyzer.interactive_mode()

if __name__ == "__main__":
    main()

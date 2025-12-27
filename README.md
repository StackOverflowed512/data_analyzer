# Automobile Data Analysis with Mistral AI

## Overview

This project uses Mistral AI API to analyze automobile datasets through natural language queries and automatically generate appropriate visualizations. The application features a React frontend with a Flask API backend.

## Architecture

- **Frontend**: React with TypeScript, Vite, Tailwind CSS
- **Backend**: Flask API with Python
- **AI**: Mistral AI API for natural language processing
- **Charts**: Plotly.js and Matplotlib
- **Data**: Pandas for data processing

## Features

- Natural language query processing using Mistral AI API
- Automatic chart type selection (bar, scatter, histogram, line)
- Smart column analysis and aggregation
- Interactive React frontend with modern UI
- RESTful API backend
- Support for both Plotly and Matplotlib charts
- Data export functionality

## Setup

### 1. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_API_URL=https://api.mistral.ai/v1/chat/completions
MISTRAL_MODEL=mistral-7b-instruct
```

### 4. Run the Application

**Start the Flask API (Terminal 1):**

```bash
python api_server.py
```

**Start the React Frontend (Terminal 2):**

```bash
cd frontend
npm run dev
```

The application will be available at:

- Frontend: http://localhost:8080
- API: http://localhost:5000

## Usage Examples

- "Show average price by brand"
- "Plot horsepower vs fuel efficiency"
- "Display price distribution histogram"
- "Compare engine size across different fuel types"

## Project Structure

```
├── frontend/                 # React TypeScript frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── services/         # API service layer
│   │   └── types/           # TypeScript declarations
│   ├── package.json
│   └── tailwind.config.js
backend/
├── api_server.py            # Flask API server
├── app.py                   # CLI application
├── config/
│   └── settings.py          # Configuration settings
├── src/
│   ├── data_processor.py    # Data handling and processing
│   ├── llm_client.py        # Mistral AI API integration
│   └── chart_generator.py   # Chart generation logic
├── data/
│   └── automobile.csv       # Sample automobile dataset
├── requirements.txt         # Python dependencies
└── .env.example            # Environment variables template
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/initialize` - Initialize application
- `GET /api/dataset-info` - Get dataset information
- `POST /api/analyze` - Analyze query and generate chart
- `POST /api/download-data` - Download processed data
- `GET /api/example-queries` - Get example queries

## Development

### Backend Development

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run Flask in development mode
python api_server.py
```

### Frontend Development

```bash
cd frontend
npm run dev
```

### Building for Production

```bash
# Build frontend
cd frontend
npm run build

# The built files will be in frontend/dist/
```

## Environment Variables

- `MISTRAL_API_KEY`: Your Mistral AI API key (required)
- `MISTRAL_API_URL`: Mistral API endpoint (optional)
- `MISTRAL_MODEL`: Model to use (optional)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

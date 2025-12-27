import React, { useState } from "react";
import {
  BarChart3,
  Database,
  Send,
  CircleDot,
  TrendingUp,
  Fuel,
  Gauge,
  DollarSign,
  Calendar,
  BarChart2,
  Car,
} from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

interface EnhancedQueryInterfaceProps {
  onAnalyze: (
    query: string,
    library: "plotly" | "matplotlib",
    forceChart?: boolean
  ) => void;
  onChartRequest: (chartType: string, query: string) => void;
  onDatasetView: () => void;
  isAnalyzing: boolean;
}

const CHART_TYPES = [
  { id: "bar", name: "Bar Chart", description: "Compare categories" },
  { id: "pie", name: "Pie Chart", description: "Show proportions" },
  { id: "line", name: "Line Chart", description: "Trends over time" },
  { id: "scatter", name: "Scatter Plot", description: "Relationships" },
  { id: "histogram", name: "Histogram", description: "Data distribution" },
  { id: "box", name: "Box Plot", description: "Statistical summary" },
  { id: "violin", name: "Violin Plot", description: "Distribution shape" },
  { id: "area", name: "Area Chart", description: "Cumulative data" },
];

// Example queries with icons
const EXAMPLE_QUERIES = [
  {
    text: "Show average price by brand",
    icon: <DollarSign className="h-4 w-4 text-white" />,
  },
  {
    text: "Display fuel type distribution as pie chart",
    icon: <Fuel className="h-4 w-4 text-white" />,
  },
  {
    text: "Compare horsepower vs price relationship",
    icon: <Gauge className="h-4 w-4 text-white" />,
  },
  {
    text: "Show price distribution across all cars",
    icon: <CircleDot className="h-4 w-4 text-white" />,
  },
  {
    text: "Show average price by fuel type",
    icon: <TrendingUp className="h-4 w-4 text-white" />,
  },
  {
    text: "List top 10 most expensive cars",
    icon: <Calendar className="h-4 w-4 text-white" />,
  },
  {
    text: "Show average horsepower by body style",
    icon: <Car className="h-4 w-4 text-white" />,
  },
  {
    text: "Compare city vs highway MPG by brand",
    icon: <BarChart2 className="h-4 w-4 text-white" />,
  },
];

export const EnhancedQueryInterface: React.FC<EnhancedQueryInterfaceProps> = ({
  onAnalyze,
  onChartRequest,
  onDatasetView,
  isAnalyzing,
}) => {
  const [query, setQuery] = useState("");
  const [selectedChartType, setSelectedChartType] = useState<string | null>(
    null
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    console.log("🔧 Submit clicked:", {
      query,
      selectedChartType,
      isAnalyzing,
    });

    if (!query.trim()) {
      console.log("❌ No query entered");
      return;
    }

    if (isAnalyzing) {
      console.log("❌ Already analyzing");
      return;
    }

    // Detect sorting intent in query
    let sortBy = null;
    let sortOrder = "asc";
    const sortMatch =
      query.match(/sorted by ([\w\s_]+)(?: (asc|desc))?/i) ||
      query.match(/order by ([\w\s_]+)(?: (asc|desc))?/i);
    if (sortMatch) {
      sortBy = sortMatch[1].replace(/ /g, "_").toLowerCase();
      if (sortMatch[2]) sortOrder = sortMatch[2].toLowerCase();
      console.log(`🟢 Detected sorting: ${sortBy}, order: ${sortOrder}`);
    }

    if (selectedChartType) {
      console.log(
        "🎯 Using onChartRequest for specific chart type:",
        selectedChartType
      );
      if (sortBy) {
        onChartRequest(
          selectedChartType,
          query + ` [sort_by:${sortBy}, sort_order:${sortOrder}]`
        );
      } else {
        onChartRequest(selectedChartType, query);
      }
    } else {
      console.log("🤖 Using onAnalyze for automatic chart selection");
      if (sortBy) {
        onAnalyze(
          query + ` [sort_by:${sortBy}, sort_order:${sortOrder}]`,
          "plotly"
        );
      } else {
        onAnalyze(query, "plotly");
      }
    }

    setQuery("");
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-white">
          <BarChart3 className="h-5 w-5" />
          Ask Your Questions About Automobile Dataset
        </CardTitle>
        <Button
          onClick={onDatasetView}
          variant="outline"
          size="sm"
          className="w-fit"
        >
          <Database className="h-4 w-4 mr-2" />
          📊 View Dataset
        </Button>
        <Badge variant="default" className="w-fit">
          System Ready • All Features Operational
        </Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Chart Type Selection */}
        <div className="space-y-3">
          <p className="text-sm text-gray-400">
            📊 Optional: Select a chart type for specific visualization, or let
            the AI choose automatically:
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {CHART_TYPES.map((chart) => (
              <Button
                key={chart.id}
                variant={selectedChartType === chart.id ? "default" : "outline"}
                onClick={() =>
                  setSelectedChartType(
                    selectedChartType === chart.id ? null : chart.id
                  )
                }
                className="h-auto p-3 flex flex-col items-center text-center"
              >
                <span className="font-medium text-sm">{chart.name}</span>
                <span className="text-xs text-gray-500">
                  {chart.description}
                </span>
              </Button>
            ))}
          </div>
          {selectedChartType && (
            <Badge variant="secondary">
              Selected:{" "}
              {CHART_TYPES.find((c) => c.id === selectedChartType)?.name}
            </Badge>
          )}
        </div>

        {/* Query Input */}
        <div className="flex gap-2">
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask anything about the automobile data..."
            onKeyDown={handleKeyDown}
            disabled={isAnalyzing}
          />
          <Button
            onClick={() => {
              console.log("🔧 Button clicked!");
              handleSubmit();
            }}
            disabled={isAnalyzing || !query.trim()}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>

        {/* Example Queries */}
        <div className="text-sm">
          <p className="font-medium mb-2 text-white">💡Smart Suggestions:</p>
          <div className="flex flex-wrap gap-3">
            {EXAMPLE_QUERIES.map((q, idx) => (
              <button
                key={idx}
                onClick={() => setQuery(q.text)}
                className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg text-white hover:bg-cyan-700 transition"
              >
                {q.icon}
                <span>{q.text}</span>
              </button>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default EnhancedQueryInterface;


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
  PieChart,
  Mic,
  MicOff,
} from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import type { ExampleQuery } from "../services/api";

interface EnhancedQueryInterfaceProps {
  onAnalyze: (
    query: string,
    library: "plotly" | "matplotlib",
    forceChart?: boolean
  ) => void;
  onChartRequest: (chartType: string, query: string) => void;
  onDatasetView: () => void;
  isAnalyzing: boolean;
  examples: ExampleQuery[];
  isDatasetLoaded: boolean;
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

const ICONS = [
  DollarSign,
  Fuel,
  Gauge,
  CircleDot,
  TrendingUp,
  Calendar,
  Car,
  BarChart2,
  PieChart,
];

export const EnhancedQueryInterface: React.FC<EnhancedQueryInterfaceProps> = ({
  onAnalyze,
  onChartRequest,
  onDatasetView,
  isAnalyzing,
  examples,
  isDatasetLoaded,
}) => {
  const [query, setQuery] = useState("");
  const [selectedChartType, setSelectedChartType] = useState<string | null>(
    null
  );

  const [isListening, setIsListening] = useState(false);
  const recognitionRef = React.useRef<any>(null);

  const toggleListening = () => {
    if (!isDatasetLoaded) return;
    
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }

    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Voice input is not supported in this browser. Please use Chrome or Edge.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setQuery(transcript);
      setTimeout(() => {
          // Optional: Auto-submit if high confidence?
          // For now, let user verify
      }, 500);
    };

    recognition.onerror = (event: any) => {
      console.error("Speech recognition error", event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && isDatasetLoaded) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (!isDatasetLoaded) return;
    
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
    <Card className={`w-full ${!isDatasetLoaded ? 'opacity-80' : ''}`}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-white">
          <BarChart3 className="h-5 w-5" />
          Ask Your Questions About the Dataset
        </CardTitle>
        <div className="flex gap-2">
            <Button
            onClick={onDatasetView}
            variant="outline"
            size="sm"
            className="w-fit"
            disabled={!isDatasetLoaded}
            >
            <Database className="h-4 w-4 mr-2" />
            📊 View Dataset
            </Button>
            {!isDatasetLoaded && (
                <Badge variant="destructive" className="animate-pulse">
                    ⚠️ No Dataset Selected
                </Badge>
            )}
             {isDatasetLoaded && (
                <Badge variant="default" className="w-fit">
                System Ready • All Features Operational
                </Badge>
            )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Chart Type Selection */}
        <div className="space-y-3">
          <p className="text-sm text-gray-400">
            📊 Optional: Select a chart type for specific visualization, or let
            the AI choose automatically:
          </p>
          <div className={`grid grid-cols-2 md:grid-cols-4 gap-2 ${!isDatasetLoaded ? 'pointer-events-none opacity-50' : ''}`}>
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
                disabled={!isDatasetLoaded}
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
        <div className="flex gap-2 relative">
          {!isDatasetLoaded && (
             <div className="absolute inset-0 z-10 bg-white/5 backdrop-blur-[1px] rounded-lg flex items-center justify-center text-sm font-semibold text-white">
                 Please select or upload a dataset to start querying
             </div>
          )}
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={isListening ? "Listening..." : "Ask anything about the data..."}
            onKeyDown={handleKeyDown}
            disabled={isAnalyzing || !isDatasetLoaded}
            className={isListening ? "border-red-500 ring-1 ring-red-500" : ""}
          />
          <Button
            onClick={toggleListening}
            variant={isListening ? "destructive" : "secondary"}
            disabled={isAnalyzing || !isDatasetLoaded}
            title={isListening ? "Stop listening" : "Start voice input"}
            className="w-12"
          >
            {isListening ? (
              <MicOff className="h-4 w-4 animate-pulse" />
            ) : (
              <Mic className="h-4 w-4" />
            )}
          </Button>
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
            {examples.map((q, idx) => {
              const IconComponent = ICONS[idx % ICONS.length];
              return (
                <button
                  key={idx}
                  onClick={() => setQuery(q.query)}
                  className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg text-white hover:bg-cyan-700 transition"
                  title={q.description}
                >
                  <IconComponent className="h-4 w-4 text-white" />
                  <span>{q.query}</span>
                </button>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default EnhancedQueryInterface;

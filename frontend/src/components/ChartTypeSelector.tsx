import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  BarChart3,
  PieChart,
  LineChart,
  Scatter3D,
  Activity,
  TrendingUp,
  Layers,
  Grid3X3,
  Target,
  Zap,
  Info,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

interface ChartType {
  name: string;
  description: string;
  requires_x: boolean;
  requires_y: boolean;
  best_for: string;
}

interface ChartTypeSelectorProps {
  onSelect: (chartType: string) => void;
  selectedType?: string;
  isOpen: boolean;
  onClose: () => void;
}

const chartIcons: { [key: string]: React.ReactNode } = {
  bar: <BarChart3 className="h-6 w-6" />,
  pie: <PieChart className="h-6 w-6" />,
  line: <LineChart className="h-6 w-6" />,
  scatter: <Scatter3D className="h-6 w-6" />,
  histogram: <Activity className="h-6 w-6" />,
  box: <Target className="h-6 w-6" />,
  violin: <TrendingUp className="h-6 w-6" />,
  area: <Layers className="h-6 w-6" />,
  heatmap: <Grid3X3 className="h-6 w-6" />,
  sunburst: <Zap className="h-6 w-6" />,
  treemap: <Grid3X3 className="h-6 w-6" />,
  funnel: <TrendingUp className="h-6 w-6" />,
  density: <Activity className="h-6 w-6" />,
};

const ChartTypeSelector: React.FC<ChartTypeSelectorProps> = ({
  onSelect,
  selectedType,
  isOpen,
  onClose,
}) => {
  const [chartTypes, setChartTypes] = useState<{ [key: string]: ChartType }>(
    {}
  );
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadChartTypes();
    }
  }, [isOpen]);

  const loadChartTypes = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/api/chart-types`
      );
      const result = await response.json();
      if (result.success) {
        setChartTypes(result.chart_types);
      }
    } catch (error) {
      console.error("Failed to load chart types:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = (chartType: string) => {
    onSelect(chartType);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-white dark:bg-gray-900 rounded-lg shadow-xl w-full max-w-4xl max-h-[80vh] overflow-auto"
      >
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold">Select Chart Type</h2>
              <p className="text-gray-600 dark:text-gray-400">
                Choose the best visualization for your data
              </p>
            </div>
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>

        <div className="p-6">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(chartTypes).map(([key, chartType]) => (
                <motion.div
                  key={key}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Card
                    className={`cursor-pointer transition-all ${
                      selectedType === key
                        ? "ring-2 ring-blue-500 border-blue-500"
                        : "hover:shadow-md"
                    }`}
                    onClick={() => handleSelect(key)}
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                            {chartIcons[key] || (
                              <BarChart3 className="h-6 w-6" />
                            )}
                          </div>
                          <div>
                            <CardTitle className="capitalize text-lg">
                              {key.replace("_", " ")}
                            </CardTitle>
                          </div>
                        </div>
                        {selectedType === key && (
                          <Badge className="bg-blue-500">Selected</Badge>
                        )}
                      </div>
                    </CardHeader>
                    <CardContent>
                      <CardDescription className="mb-3">
                        {chartType.description}
                      </CardDescription>

                      <div className="space-y-2 text-sm">
                        <div className="flex items-center space-x-2">
                          <Badge
                            variant={
                              chartType.requires_x ? "default" : "secondary"
                            }
                          >
                            X-axis:{" "}
                            {chartType.requires_x ? "Required" : "Optional"}
                          </Badge>
                          <Badge
                            variant={
                              chartType.requires_y ? "default" : "secondary"
                            }
                          >
                            Y-axis:{" "}
                            {chartType.requires_y ? "Required" : "Optional"}
                          </Badge>
                        </div>

                        <div className="mt-3 p-2 bg-gray-50 dark:bg-gray-800 rounded text-xs">
                          <div className="flex items-start space-x-2">
                            <Info className="h-3 w-3 mt-0.5 text-blue-500 flex-shrink-0" />
                            <span className="text-gray-600 dark:text-gray-400">
                              <strong>Best for:</strong> {chartType.best_for}
                            </span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ChartTypeSelector;

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Table,
  Search,
  ChevronLeft,
  ChevronRight,
  Database,
} from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Badge } from "./ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

interface DatasetViewerProps {
  datasetInfo: any;
  onClose: () => void;
  isOpen?: boolean;
}

interface PaginationData {
  current_page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

interface DatasetData {
  data: any[];
  pagination: PaginationData;
}

const DatasetViewer: React.FC<DatasetViewerProps> = ({
  datasetInfo,
  onClose,
}) => {
  const [datasetData, setDatasetData] = useState<DatasetData | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [searchTerm, setSearchTerm] = useState("");
  const [filters, setFilters] = useState<any>({});
  const [activeTab, setActiveTab] = useState("data");

  useEffect(() => {
    loadDataset();
  }, [currentPage, pageSize]);

  const loadDataset = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `${
          import.meta.env.VITE_API_URL
        }/api/dataset?page=${currentPage}&page_size=${pageSize}`
      );
      const result = await response.json();
      if (result.success) {
        setDatasetData(result);
      }
    } catch (error) {
      console.error("Failed to load dataset:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchTerm.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/api/dataset/search`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ search_term: searchTerm }),
        }
      );
      const result = await response.json();
      if (result.success) {
        setDatasetData({
          data: result.results,
          pagination: {
            current_page: 1,
            page_size: result.results.length,
            total_rows: result.count,
            total_pages: 1,
            has_next: false,
            has_previous: false,
          },
        });
      }
    } catch (error) {
      console.error("Search failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage);
  };

  const renderTable = () => {
    if (!datasetData?.data || datasetData.data.length === 0) {
      return (
        <div className="text-center py-8 text-gray-500">No data available</div>
      );
    }

    const columns = Object.keys(datasetData.data[0]);

    return (
      <div className="overflow-x-auto">
        <table className="w-full border-collapse border border-gray-200 dark:border-gray-700">
          <thead>
            <tr className="bg-gray-50 dark:bg-gray-800">
              {columns.map((column) => (
                <th
                  key={column}
                  className="border border-gray-200 dark:border-gray-700 px-4 py-2 text-left text-sm font-medium text-gray-900 dark:text-gray-100"
                >
                  {column.replace("_", " ").toUpperCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {datasetData.data.map((row, index) => (
              <tr
                key={index}
                className={`${
                  index % 2 === 0
                    ? "bg-white dark:bg-gray-900"
                    : "bg-gray-50 dark:bg-gray-800"
                } hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors`}
              >
                {columns.map((column) => (
                  <td
                    key={column}
                    className="border border-gray-200 dark:border-gray-700 px-4 py-2 text-sm text-gray-900 dark:text-gray-100"
                  >
                    {row[column] !== null && row[column] !== undefined ? (
                      String(row[column])
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const renderPagination = () => {
    if (!datasetData?.pagination) return null;

    const { current_page, total_pages, has_previous, has_next } =
      datasetData.pagination;

    return (
      <div className="flex items-center justify-between mt-4">
        <div className="text-sm text-gray-600 dark:text-gray-400">
          Showing page {current_page} of {total_pages}(
          {datasetData.pagination.total_rows} total rows)
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(current_page - 1)}
            disabled={!has_previous}
            className="border-blue-300 text-blue-600 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(current_page + 1)}
            disabled={!has_next}
            className="border-blue-300 text-blue-600 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  };

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
        className="bg-white dark:bg-gray-900 rounded-lg shadow-xl w-full max-w-7xl h-[90vh] flex flex-col"
      >
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700 bg-blue-50 dark:bg-blue-900/10">
          <div className="flex items-center space-x-3">
            <Database className="h-6 w-6 text-blue-600" />
            <h2 className="text-2xl font-bold text-blue-700 dark:text-blue-300">
              Dataset Viewer
            </h2>
          </div>
          <Button
            variant="outline"
            onClick={onClose}
            className="border-blue-300 text-blue-600 hover:bg-blue-50 hover:text-blue-700"
          >
            Close
          </Button>
        </div>

        <div className="flex-1 overflow-hidden">
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="h-full flex flex-col"
          >
            <TabsList className="grid w-full grid-cols-1 mx-6 mt-4 bg-blue-50 dark:bg-blue-900/20">
              <TabsTrigger
                value="data"
                className="flex items-center space-x-2 data-[state=active]:bg-blue-600 data-[state=active]:text-white text-blue-600"
              >
                <Table className="h-4 w-4" />
                <span>Data</span>
              </TabsTrigger>
            </TabsList>

            <div className="flex-1 overflow-hidden px-6 pb-6">
              <TabsContent value="data" className="h-full flex flex-col mt-4">
                <div className="flex items-center space-x-4 mb-4">
                  <div className="flex items-center space-x-2">
                    <Input
                      placeholder="Search dataset..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      onKeyPress={(e) => e.key === "Enter" && handleSearch()}
                      className="w-64 border-blue-300 focus:border-blue-500 focus:ring-blue-500"
                    />
                    <Button
                      onClick={handleSearch}
                      size="sm"
                      className="bg-blue-600 hover:bg-blue-700 text-white"
                    >
                      <Search className="h-4 w-4" />
                    </Button>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={loadDataset}
                    className="border-blue-300 text-blue-600 hover:bg-blue-50 hover:text-blue-700"
                  >
                    Reset
                  </Button>
                </div>

                <div className="flex-1 overflow-auto border border-gray-200 dark:border-gray-700 rounded-lg">
                  {loading ? (
                    <div className="flex items-center justify-center h-64">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    </div>
                  ) : (
                    renderTable()
                  )}
                </div>

                {renderPagination()}
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default DatasetViewer;

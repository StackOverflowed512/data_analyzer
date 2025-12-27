import React from "react";
import { motion } from "framer-motion";
import { Database, Eye, BarChart3 } from "lucide-react";

interface DatasetPreviewProps {
  data: any[];
}

const DatasetPreview: React.FC<DatasetPreviewProps> = ({ data }) => {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return null;
  }

  const columns = data[0] ? Object.keys(data[0]) : [];

  return (
    <motion.div 
      className="glass rounded-2xl shadow-2xl p-6 border border-white/20 backdrop-blur-xl"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
    >
      <motion.div 
        className="flex items-center mb-6"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="relative">
          <Eye className="h-7 w-7 text-white mr-3" />
          <div className="absolute inset-0 h-7 w-7 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full blur-lg opacity-50 animate-pulse"></div>
        </div>
        <h3 className="text-xl font-bold text-white">
          Dataset Preview
        </h3>
      </motion.div>

      <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead className="bg-white/10">
              <tr>
                {columns.map((column, index) => (
                  <motion.th
                    key={column}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 + index * 0.05 }}
                    className="text-left py-4 px-4 font-bold text-white border-b border-white/10"
                  >
                    <div className="flex items-center space-x-2">
                      <BarChart3 className="h-4 w-4 text-white/60" />
                      <span>{column}</span>
                    </div>
                  </motion.th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, rowIndex) => (
                <motion.tr 
                  key={rowIndex} 
                  className="border-b border-white/5 hover:bg-white/5 transition-colors duration-200"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 + rowIndex * 0.05 }}
                >
                  {columns.map((column, colIndex) => (
                    <td key={column} className="py-3 px-4 text-white/80 font-medium">
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.8 + (rowIndex * columns.length + colIndex) * 0.02 }}
                      >
                        {typeof row[column] === "number"
                          ? row[column].toFixed(2)
                          : String(row[column])}
                      </motion.span>
                    </td>
                  ))}
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <motion.div 
        className="mt-4 flex items-center justify-center space-x-2 text-white/60 text-sm bg-white/5 backdrop-blur-sm rounded-xl p-3 border border-white/10"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1 }}
      >
        <Database className="h-4 w-4" />
        <span>Showing first {data.length} rows of your dataset</span>
      </motion.div>
    </motion.div>
  );
};

export default DatasetPreview;
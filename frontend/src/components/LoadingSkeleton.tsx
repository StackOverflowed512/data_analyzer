import React from "react";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

const LoadingSkeleton: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6 p-6"
    >
      <div className="flex items-center space-x-2 text-blue-600">
        <Loader2 className="h-5 w-5 animate-spin" />
        <span className="text-sm font-medium">Analyzing your query...</span>
      </div>

      <div className="space-y-4">
        <div className="h-4 bg-gray-200 rounded animate-pulse"></div>
        <div className="h-4 bg-gray-200 rounded animate-pulse w-3/4"></div>
        <div className="h-32 bg-gray-200 rounded animate-pulse"></div>
        <div className="h-4 bg-gray-200 rounded animate-pulse w-1/2"></div>
      </div>
    </motion.div>
  );
};

export default LoadingSkeleton;

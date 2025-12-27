import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Clock, X, Plus, Minus } from "lucide-react";
import type { QueryHistoryItem } from "../hooks/useQueryHistory";

interface QueryHistoryProps {
  history: QueryHistoryItem[];
  activeQueryId: string | null;
  addQuery: (query: string, result: any) => void;
  onSelectQuery: (id: string) => void;
  onDeleteQuery: (id: string) => void;
  onClearHistory: () => void;
}

const QueryHistory: React.FC<QueryHistoryProps> = ({
  history,
  activeQueryId,
  onSelectQuery,
  onDeleteQuery,
  onClearHistory,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="w-full px-4 py-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-white text-lg font-semibold flex items-center space-x-2">
          <motion.button
            onClick={() => setIsOpen((prev) => !prev)}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className="p-2 rounded-xl bg-white/10 backdrop-blur-sm text-white hover:bg-white/20 transition-all duration-300 border border-white/20"
            title={isOpen ? "Hide history" : "Show history"}
          >
            {isOpen ? (
              <Minus className="h-5 w-5" />
            ) : (
              <Plus className="h-5 w-5" />
            )}
          </motion.button>
          <span>Query History</span>
        </h2>

        <motion.button
          onClick={() => {
            if (window.confirm("Clear all query history?")) {
              onClearHistory();
            }
          }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="p-2 rounded-xl bg-white/10 backdrop-blur-sm text-white hover:bg-red-500/20 transition-all duration-300 border border-white/20"
          title="Clear all history"
        >
          <X className="h-5 w-5" />
        </motion.button>
      </div>

      {/* Collapsible Content */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            {history.length === 0 ? (
              <motion.div
                className="w-full flex flex-col items-center justify-center py-12"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <div className="w-16 h-16 bg-gradient-to-r from-primary-400 to-accent-400 rounded-full flex items-center justify-center mx-auto mb-4 opacity-50">
                  <Clock className="h-8 w-8 text-white" />
                </div>
                <p className="text-white/60 text-lg">No queries yet</p>
                <p className="text-white/40 text-sm mt-2">
                  Your analysis history will appear here
                </p>
              </motion.div>
            ) : (
              history.map((item, index) => (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: 20, x: 20 }}
                  animate={{ opacity: 1, y: 0, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`w-full rounded-xl mt-4 border transition-all duration-300 ${
                    activeQueryId === item.id
                      ? "bg-gradient-to-r from-primary-500/30 to-accent-500/30 border-primary-400/50 shadow-lg shadow-primary-500/20"
                      : "bg-white/5 backdrop-blur-sm border-white/20 hover:border-white/40 hover:bg-white/10"
                  }`}
                >
                  <div className="flex items-start space-x-3 p-4">
                    {/* Status dot */}
                    <div
                      className={`w-3 h-3 rounded-full mt-2 ${
                        activeQueryId === item.id
                          ? "bg-gradient-to-r from-primary-400 to-accent-400 animate-pulse"
                          : "bg-white/30"
                      }`}
                    />
                    {/* Query content */}
                    <div
                      className="flex-1 cursor-pointer"
                      onClick={() => onSelectQuery(item.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-white font-medium text-sm mb-1">
                            {item.query}
                          </p>
                          <p className="text-white/60 text-xs">
                            {item.timestamp.toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    </div>
                    {/* Individual delete button */}
                    <motion.button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteQuery(item.id);
                      }}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      className="p-1 rounded-full hover:bg-red-500/20 text-red-400 hover:text-red-300 transition-colors ml-2"
                      title="Delete query"
                    >
                      <p>Delete</p>
                    </motion.button>
                  </div>
                </motion.div>
              ))
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default QueryHistory;

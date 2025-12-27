import { useState } from "react";
import type { AnalysisResult } from "../services/api";

export interface QueryHistoryItem {
  id: string;
  query: string;
  result: AnalysisResult;
  timestamp: Date;
}

export const useQueryHistory = () => {
  // Load history from localStorage on mount
  const [history, setHistory] = useState<QueryHistoryItem[]>(() => {
    const stored = localStorage.getItem("queryHistory");
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        // Convert timestamp back to Date
        return parsed.map((item: any) => ({
          ...item,
          timestamp: new Date(item.timestamp),
        }));
      } catch {
        return [];
      }
    }
    return [];
  });

  const [activeQueryId, setActiveQueryId] = useState<string | null>(null);

  const addQuery = (query: string, result: AnalysisResult) => {
    const id = Date.now().toString();
    const newItem: QueryHistoryItem = {
      id,
      query,
      result,
      timestamp: new Date(),
    };
    console.log("[addQuery] Adding query to history:", newItem);
    setHistory((prev) => {
      const updated = [newItem, ...prev.slice(0, 9)];
      console.log("[addQuery] Updated history:", updated);
      localStorage.setItem("queryHistory", JSON.stringify(updated));
      return updated;
    });
    setActiveQueryId(id);
  };

  const selectQuery = (id: string) => {
    setActiveQueryId(id);
  };

  const deleteQuery = (id: string) => {
    console.log("Deleting query", id);
    setActiveQueryId((prevActive) => (prevActive === id ? null : prevActive));
    setHistory((prev) => {
      const updated = prev.filter((item) => item.id !== id);
      localStorage.setItem("queryHistory", JSON.stringify(updated));
      return updated;
    });
  };

  const clearHistory = () => {
    setHistory([]);
    setActiveQueryId(null);
    localStorage.removeItem("queryHistory");
  };

  const activeQuery = history.find((item) => item.id === activeQueryId);

  return {
    history,
    activeQuery,
    activeQueryId,
    addQuery,
    selectQuery,
    deleteQuery,
    clearHistory, // <-- export this
  };
};

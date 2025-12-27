import { useState, useEffect } from "react";
import { apiService } from "../services/api";

const ConnectionTest: React.FC = () => {
  const [connectionStatus, setConnectionStatus] = useState<
    "testing" | "connected" | "error"
  >("testing");
  const [errorMessage, setErrorMessage] = useState<string>("");

  useEffect(() => {
    testConnection();
  }, []);

  const testConnection = async () => {
    try {
      setConnectionStatus("testing");
      const response = await apiService.healthCheck();
      if (response.status === "healthy") {
        setConnectionStatus("connected");
      } else {
        setConnectionStatus("error");
        setErrorMessage("Backend responded but status is not healthy");
      }
    } catch (error) {
      setConnectionStatus("error");
      setErrorMessage(error instanceof Error ? error.message : "Unknown error");
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case "testing":
        return "text-yellow-500";
      case "connected":
        return "text-green-500";
      case "error":
        return "text-red-500";
      default:
        return "text-gray-500";
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case "testing":
        return "Testing connection...";
      case "connected":
        return "Connected to backend";
      case "error":
        return "Connection failed";
      default:
        return "Unknown status";
    }
  };

  return null; // Removed overlay to prevent navbar overlap
};

export default ConnectionTest;

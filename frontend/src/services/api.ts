import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

// Configure axios with better error handling and timeout
const axiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30 seconds timeout
    headers: {
        "Content-Type": "application/json",
    },
});

// Add response interceptor for better error handling
axiosInstance.interceptors.response.use(
    (response) => response,
    (error) => {
        console.error("API Error:", error);
        if (error.code === "ECONNREFUSED") {
            console.error("Backend server is not running or not accessible");
        }
        return Promise.reject(error);
    }
);

export interface DatasetInfo {
    id?: number;
    name?: string;
    rows: number;
    columns: number;
    column_names: string[];
    numeric_columns: string[];
    categorical_columns: string[];
    missing_values: Record<string, number>;
}

export interface ChartAnalysis {
    chart: string;
    x: string;
    y?: string;
    agg?: string;
}

export interface AnalysisResult {
    success: boolean;
    analysis: ChartAnalysis;
    chart_data?: any;
    data: any[];
    chart_summary?: string;
    summary?: string;
    data_count: number;
    columns: string[];
    needs_chart: boolean;
    chart_error?: string;
    error?: string;
}

export interface ExampleQuery {
    query: string;
    description: string;
}

export interface DatasetItem {
    id: number;
    filename: string;
    upload_date: string;
    row_count: number;
    column_count: number;
}

export interface DatasetListResult {
    success: boolean;
    datasets: DatasetItem[];
    current_dataset_id: number | null;
    error?: string;
}

class ApiService {
    private axiosInstance;

    constructor() {
        this.axiosInstance = axiosInstance;
    }

    async healthCheck(): Promise<{ status: string; message: string }> {
        // Retry logic for deployment issues
        for (let attempt = 1; attempt <= 2; attempt++) {
            try {
                console.log(`Health check attempt ${attempt}/2`);
                const response = await this.axiosInstance.get("/api/health");
                console.log("Health check response:", response.data);
                return response.data;
            } catch (error: any) {
                console.error(`Health check attempt ${attempt} failed:`, error);

                if (attempt === 2) {
                    // Final attempt failed
                    if (
                        error.response?.status === 503 ||
                        error.response?.status === 502
                    ) {
                        throw new Error(
                            "Backend is starting up. Please wait a moment."
                        );
                    } else if (
                        error.code === "NETWORK_ERROR" ||
                        error.code === "ECONNREFUSED"
                    ) {
                        throw new Error(
                            "Cannot connect to backend. Please check your connection."
                        );
                    } else {
                        throw new Error("Backend health check failed");
                    }
                }

                // Wait before retry
                await new Promise((resolve) => setTimeout(resolve, 2000));
            }
        }

        throw new Error("Health check failed after 2 attempts");
    }

    async initialize(): Promise<{
        success: boolean;
        message?: string;
        error?: string;
    }> {
        // Retry logic for deployment issues
        for (let attempt = 1; attempt <= 3; attempt++) {
            try {
                console.log(`Initialization attempt ${attempt}/3`);
                const response = await this.axiosInstance.post(
                    "/api/initialize"
                );
                console.log("Initialize response:", response.data);
                return response.data;
            } catch (error: any) {
                console.error(
                    `Initialization attempt ${attempt} failed:`,
                    error
                );

                if (attempt === 3) {
                    // Final attempt failed
                    if (
                        error.response?.status === 503 ||
                        error.response?.status === 502
                    ) {
                        throw new Error(
                            "Backend is starting up. Please refresh the page in a moment."
                        );
                    } else if (
                        error.code === "NETWORK_ERROR" ||
                        error.code === "ECONNREFUSED"
                    ) {
                        throw new Error(
                            "Cannot connect to backend. Please check your connection."
                        );
                    } else {
                        throw new Error(
                            error.response?.data?.error ||
                                "Initialization failed"
                        );
                    }
                }

                // Wait before retry (exponential backoff)
                await new Promise((resolve) =>
                    setTimeout(resolve, attempt * 1000)
                );
            }
        }

        // This should never be reached due to the throw in the loop
        throw new Error("Initialization failed after 3 attempts");
    }

    async getDatasetInfo(): Promise<{
        success: boolean;
        info?: DatasetInfo;
        preview?: any[];
        error?: string;
        code?: string;
    }> {
        try {
            const response = await this.axiosInstance.get("/api/dataset-info");
            return response.data;
        } catch (error: any) {
            if (error.response && error.response.data && error.response.data.code === "NO_DATASET") {
                return error.response.data;
            }
            throw error;
        }
    }

    async analyzeQuery(
        query: string,
        library: "plotly" | "matplotlib" = "plotly",
        forceChart?: boolean,
        chartType?: string
    ): Promise<AnalysisResult> {
        const requestBody: any = {
            query,
            library,
        };

        if (forceChart !== undefined) {
            requestBody.force_chart = forceChart;
        }

        if (chartType) {
            requestBody.chart_type = chartType;
        }

        // Debug logging
        console.log("Sending request to backend:", {
            url: "/api/analyze",
            baseURL: this.axiosInstance.defaults.baseURL,
            fullURL: `${this.axiosInstance.defaults.baseURL}/api/analyze`,
            body: requestBody,
        });

        try {
            const response = await this.axiosInstance.post(
                "/api/analyze",
                requestBody
            );

            console.log("Backend response:", response.data);
            return response.data;
        } catch (error: any) {
            console.error("API Request failed:", {
                message: error.message,
                status: error.response?.status,
                statusText: error.response?.statusText,
                data: error.response?.data,
                url: error.config?.url,
                requestBody: requestBody,
            });

            // Handle specific error cases
            if (
                error.response?.status === 503 ||
                error.response?.status === 502
            ) {
                // Backend is starting up (common on Render free tier)
                throw new Error(
                    "Backend is starting up. Please try again in a moment."
                );
            } else if (error.response?.status >= 500) {
                // Server error
                throw new Error("Server error. Please try again later.");
            } else if (
                error.code === "NETWORK_ERROR" ||
                error.code === "ECONNREFUSED"
            ) {
                // Network connection issue
                throw new Error(
                    "Cannot connect to backend. Please check your connection."
                );
            } else if (error.response?.data?.error) {
                // Backend returned an error message
                throw new Error(error.response.data.error);
            } else {
                // Generic error
                throw error;
            }
        }
    }

    async downloadData(data: any[]): Promise<{
        success: boolean;
        csv_content: string;
        filename: string;
        error?: string;
    }> {
        const response = await this.axiosInstance.post("/api/download-data", {
            data,
        });
        return response.data;
    }

    async getExampleQueries(): Promise<{
        success: boolean;
        examples: ExampleQuery[];
    }> {
        const response = await this.axiosInstance.get("/api/example-queries");
        return response.data;
    }

    async uploadDataset(file: File): Promise<{
        success: boolean;
        filename: string;
        info?: DatasetInfo;
        preview?: any[];
        error?: string;
    }> {
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(
                `${this.axiosInstance.defaults.baseURL}/api/upload-dataset`,
                {
                    method: "POST",
                    body: formData,
                }
            );

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(
                    errorData.error || `Upload failed: ${response.statusText}`
                );
            }

            const result = await response.json();
            return result;
        } catch (error: any) {
            console.error("Dataset upload failed:", error);
            throw new Error(error.message || "Failed to upload dataset");
        }
    }

    async reinitializeWithNewDataset(): Promise<{
        success: boolean;
        message?: string;
        error?: string;
    }> {
        const response = await this.axiosInstance.post("/api/reinitialize");
        return response.data;
    }

    async listDatasets(): Promise<DatasetListResult> {
        const response = await this.axiosInstance.get("/api/datasets");
        return response.data;
    }

    async switchDataset(id: number): Promise<{
        success: boolean;
        message?: string;
        info?: DatasetInfo;
        error?: string;
    }> {
        const response = await this.axiosInstance.post("/api/dataset/switch", { id });
        return response.data;
    }
}

export const apiService = new ApiService();

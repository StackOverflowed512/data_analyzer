import React, { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import {
    AlertCircle,
    CheckCircle,
    Loader2,
    Sparkles,
    Database,
    MessageCircle,
    BarChart3,
    Upload,
} from "lucide-react";
import { Toaster, toast } from "react-hot-toast";
import { motion, AnimatePresence } from "framer-motion";
import {
    apiService,
    type DatasetInfo,
    type AnalysisResult,
    type ExampleQuery,
} from "./services/api";
import Navbar from "./components/Navbar";
import LoadingSkeleton from "./components/LoadingSkeleton";
import QueryHistory from "./components/QueryHistory";
import DatasetPreview from "./components/DatasetPreview";
import ExampleQueries from "./components/ExampleQueries";
import DatasetViewer from "./components/DatasetViewer";
import EnhancedQueryInterface from "./components/EnhancedQueryInterface";
import FileUploader from "./components/FileUploader";
import { useQueryHistory } from "./hooks/useQueryHistory";
import { useTheme } from "./hooks/useTheme";
import ErrorBoundary from "./components/ErrorBoundary";
import ConnectionTest from "./components/ConnectionTest";

interface AppState {
    isInitialized: boolean;
    isLoading: boolean;
    error: string | null;
    datasetInfo: DatasetInfo | null;
    datasetPreview: any[];
    analysisResult: AnalysisResult | null;
    isAnalyzing: boolean;
    exampleQueries: ExampleQuery[];
    showDatasetViewer: boolean;
    currentDatasetName: string;
    showFileUploader: boolean;
}

function App() {
    useTheme();
    const {
        history,
        activeQuery,
        addQuery,
        selectQuery,
        deleteQuery,
        clearHistory,
    } = useQueryHistory();
    const [state, setState] = useState<AppState>({
        isInitialized: false,
        isLoading: false,
        error: null,
        datasetInfo: null,
        datasetPreview: [],
        analysisResult: null,
        isAnalyzing: false,
        exampleQueries: [],
        showDatasetViewer: false,
        currentDatasetName: "Automobile Dataset",
        showFileUploader: false,
    });

    useEffect(() => {
        initializeApp();
    }, []);

    const initializeApp = async (retryCount = 0) => {
        const maxRetries = 3;
        setState((prev) => ({ ...prev, isLoading: true, error: null }));

        try {
            console.log(
                `🔧 Initialization attempt ${retryCount + 1}/${maxRetries + 1}`
            );

            // Wake up the backend (for services like Render that go to sleep)
            console.log("⏰ Waking up backend...");
            try {
                try {
                    const wakeResponse = await fetch(
                        `${import.meta.env.VITE_API_URL}/api/wake`,
                        {
                            method: "GET",
                            headers: { "Content-Type": "application/json" },
                        }
                    );
                    if (wakeResponse.ok) {
                        console.log("✅ Backend woken up via wake endpoint");
                    }
                } catch (wakeError) {
                    console.log(
                        "⚠️ Wake endpoint failed, trying health check..."
                    );
                }

                await apiService.healthCheck();
                console.log("✅ Backend is awake");
            } catch (healthError) {
                console.log("⚠️ Health check failed, but continuing...");
            }

            // Add delay to let backend fully wake up
            if (retryCount > 0) {
                console.log(
                    `⏳ Waiting ${retryCount * 2} seconds before retry...`
                );
                await new Promise((resolve) =>
                    setTimeout(resolve, retryCount * 2000)
                );
            }

            // Initialize app with timeout
            console.log("🚀 Initializing app...");
            const initResult = (await Promise.race([
                apiService.initialize(),
                new Promise((_, reject) =>
                    setTimeout(
                        () => reject(new Error("Initialization timeout")),
                        15000
                    )
                ),
            ])) as any;

            if (!initResult.success) {
                throw new Error(initResult.error || "Failed to initialize");
            }
            console.log("✅ App initialized successfully");

            // Get dataset info with timeout
            console.log("📊 Loading dataset info...");
            const datasetResult = (await Promise.race([
                apiService.getDatasetInfo(),
                new Promise((_, reject) =>
                    setTimeout(
                        () => reject(new Error("Dataset loading timeout")),
                        10000
                    )
                ),
            ])) as any;

            if (!datasetResult.success) {
                throw new Error(
                    datasetResult.error || "Failed to load dataset"
                );
            }
            console.log("✅ Dataset loaded successfully");

            // Get example queries (optional, don't fail if this fails)
            console.log("💡 Loading example queries...");
            let exampleQueries = [];
            try {
                const examplesResult = await apiService.getExampleQueries();
                exampleQueries = examplesResult.success
                    ? examplesResult.examples
                    : [];
            } catch (err) {
                console.log(
                    "⚠️ Failed to load example queries, using defaults"
                );
            }

            setState((prev) => ({
                ...prev,
                isInitialized: true,
                isLoading: false,
                datasetInfo: datasetResult.info,
                datasetPreview: datasetResult.preview,
                exampleQueries,
            }));

            toast.success("🚀 Ready to analyze your data!", {
                style: {
                    background: "rgba(34, 197, 94, 0.9)",
                    color: "white",
                    backdropFilter: "blur(10px)",
                },
            });
        } catch (error) {
            console.error(
                `❌ Initialization attempt ${retryCount + 1} failed:`,
                error
            );

            if (retryCount < maxRetries) {
                const nextRetry = retryCount + 1;
                console.log(
                    `🔄 Retrying initialization (${nextRetry}/${
                        maxRetries + 1
                    })...`
                );

                setState((prev) => ({
                    ...prev,
                    isLoading: true,
                    error: `Retry ${nextRetry}/${
                        maxRetries + 1
                    }... Backend may be starting up.`,
                }));

                setTimeout(() => {
                    initializeApp(nextRetry);
                }, Math.pow(2, retryCount) * 1000);
            } else {
                const errorMessage =
                    error instanceof Error
                        ? error.message
                        : "Initialization failed";

                setState((prev) => ({
                    ...prev,
                    isLoading: false,
                    error: `${errorMessage}. Please check if the backend is running.`,
                }));

                toast.error(
                    "❌ Failed to connect to backend after multiple attempts",
                    {
                        style: {
                            background: "rgba(239, 68, 68, 0.9)",
                            color: "white",
                            backdropFilter: "blur(10px)",
                        },
                    }
                );
            }
        }
    };

    const handleAnalyzeQuery = async (
        query: string,
        library: "plotly" | "matplotlib",
        forceChart: boolean = false
    ) => {
        setState((prev) => ({
            ...prev,
            isAnalyzing: true,
            error: null,
            analysisResult: null,
        }));

        try {
            const result = await apiService.analyzeQuery(
                query,
                library,
                forceChart
            );

            setState((prev) => ({
                ...prev,
                isAnalyzing: false,
                analysisResult: result,
            }));

            if (result.success) {
                console.log(
                    "[App] Query result is success, calling addQuery:",
                    {
                        query,
                        result,
                    }
                );
                addQuery(query, result);
                if (result.needs_chart && result.chart_data) {
                    toast.success("✨ Chart created successfully!", {
                        style: {
                            background: "rgba(34, 197, 94, 0.9)",
                            color: "white",
                            backdropFilter: "blur(10px)",
                        },
                    });
                } else {
                    toast.success("📊 Data retrieved successfully!", {
                        style: {
                            background: "rgba(59, 130, 246, 0.9)",
                            color: "white",
                            backdropFilter: "blur(10px)",
                        },
                    });
                }
            } else {
                const errorMessage = result.error || "Analysis failed";
                if (
                    errorMessage
                        .toLowerCase()
                        .includes("unrelated to automobile") ||
                    errorMessage
                        .toLowerCase()
                        .includes("not about automobiles") ||
                    errorMessage.toLowerCase().includes("automobile data query")
                ) {
                    toast.error(
                        "🚗 Please ask questions related to automobile data (cars, vehicles, specifications, etc.)",
                        {
                            style: {
                                background: "rgba(245, 158, 11, 0.9)",
                                color: "white",
                                backdropFilter: "blur(10px)",
                            },
                            duration: 4000,
                        }
                    );
                } else {
                    toast.error(errorMessage, {
                        style: {
                            background: "rgba(239, 68, 68, 0.9)",
                            color: "white",
                            backdropFilter: "blur(10px)",
                        },
                    });
                }
            }
        } catch (error) {
            setState((prev) => ({
                ...prev,
                isAnalyzing: false,
            }));

            const errorMessage =
                error instanceof Error
                    ? error.message
                    : "An unknown error occurred";

            console.error("🔥 Query analysis failed:", error);

            toast.error(`❌ Query Failed: ${errorMessage}`, {
                style: {
                    background: "rgba(239, 68, 68, 0.9)",
                    color: "white",
                    backdropFilter: "blur(10px)",
                },
                duration: 5000,
            });
        }
    };

    const handleChartRequest = async (chartType: string, query: string) => {
        setState((prev) => ({
            ...prev,
            isAnalyzing: true,
            error: null,
            analysisResult: null,
        }));

        try {
            const result = await apiService.analyzeQuery(
                query,
                "plotly",
                true,
                chartType
            );

            setState((prev) => ({
                ...prev,
                isAnalyzing: false,
                analysisResult: result,
            }));

            if (result.success) {
                console.log(
                    "[App] ChartRequest result is success, calling addQuery:",
                    {
                        query,
                        result,
                    }
                );
                addQuery(query, result);
                toast.success(`✨ ${chartType} chart created successfully!`, {
                    style: {
                        background: "rgba(34, 197, 94, 0.9)",
                        color: "white",
                        backdropFilter: "blur(10px)",
                    },
                });
            } else {
                const errorMessage = result.error || "Chart creation failed";
                if (
                    errorMessage
                        .toLowerCase()
                        .includes("unrelated to automobile") ||
                    errorMessage
                        .toLowerCase()
                        .includes("not about automobiles") ||
                    errorMessage.toLowerCase().includes("automobile data query")
                ) {
                    toast.error(
                        "🚗 Please ask questions related to automobile data for chart creation",
                        {
                            style: {
                                background: "rgba(245, 158, 11, 0.9)",
                                color: "white",
                                backdropFilter: "blur(10px)",
                            },
                            duration: 4000,
                        }
                    );
                } else {
                    toast.error(errorMessage, {
                        style: {
                            background: "rgba(239, 68, 68, 0.9)",
                            color: "white",
                            backdropFilter: "blur(10px)",
                        },
                    });
                }
            }
        } catch (error) {
            setState((prev) => ({
                ...prev,
                isAnalyzing: false,
            }));

            const errorMessage =
                error instanceof Error ? error.message : "An error occurred";

            console.error("🔥 Query processing error:", errorMessage);

            toast.error(`❌ Query failed: ${errorMessage}`, {
                style: {
                    background: "rgba(239, 68, 68, 0.9)",
                    color: "white",
                    backdropFilter: "blur(10px)",
                },
            });
        }
    };

    const handleOpenDatasetViewer = () => {
        setState((prev) => ({ ...prev, showDatasetViewer: true }));
    };

    const handleCloseDatasetViewer = () => {
        setState((prev) => ({ ...prev, showDatasetViewer: false }));
    };

    const handleDatasetUpload = async (filename: string, info?: DatasetInfo, preview?: any[]) => {
        console.log("📤 Dataset upload successful:", filename);

        // Fetch refreshed example queries based on the new dataset
        let newExampleQueries = state.exampleQueries;
        try {
            console.log("💡 Refreshing example queries for new dataset...");
            const examplesResult = await apiService.getExampleQueries();
            if (examplesResult.success) {
                newExampleQueries = examplesResult.examples;
            }
        } catch (error) {
            console.warn("⚠️ Failed to refresh example queries:", error);
            // Keep existing or default queries if refresh fails
        }

        setState((prev) => ({
            ...prev,
            currentDatasetName: filename,
            showFileUploader: false,
            analysisResult: null,
            isInitialized: true,
            // Update dataset info and preview if provided
            datasetInfo: info || prev.datasetInfo,
            datasetPreview: preview || prev.datasetPreview,
            exampleQueries: newExampleQueries,
        }));

        toast.success(`🚀 Ready to analyze "${filename}"!`, {
            style: {
                background: "rgba(34, 197, 94, 0.9)",
                color: "white",
                backdropFilter: "blur(10px)",
            },
        });
    };

    const handleUploadError = (error: string) => {
        console.error("❌ Upload error:", error);
        toast.error(error, {
            style: {
                background: "rgba(239, 68, 68, 0.9)",
                color: "white",
                backdropFilter: "blur(10px)",
            },
        });
    };

    const displayResult = activeQuery?.result || state.analysisResult;

    if (state.isLoading) {
        return (
            <div className="min-h-screen animated-bg flex items-center justify-center">
                <div className="glass rounded-2xl p-8 text-center border border-white/20 backdrop-blur-xl">
                    <Loader2 className="h-12 w-12 animate-spin text-white mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-white mb-2">
                        Initializing Application
                    </h2>
                    <p className="text-white/70">Loading your dataset...</p>
                </div>
            </div>
        );
    }

    if (state.error) {
        return (
            <div className="min-h-screen animated-bg flex items-center justify-center">
                <div className="glass rounded-2xl p-8 text-center border border-red-500/30 backdrop-blur-xl">
                    <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-white mb-2">
                        Initialization Error
                    </h2>
                    <p className="text-white/80 mb-6">{state.error}</p>
                    <button
                        onClick={() => initializeApp()}
                        className="px-6 py-3 bg-gradient-to-r from-primary-500 to-secondary-500 text-white rounded-xl font-medium hover:from-primary-600 hover:to-secondary-600 transition-all duration-300 transform hover:scale-105"
                    >
                        Retry Initialization
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen animated-bg">
            <ConnectionTest />
            <Toaster
                position="top-right"
                toastOptions={{
                    duration: 4000,
                    style: {
                        borderRadius: "12px",
                        border: "1px solid rgba(255, 255, 255, 0.2)",
                    },
                }}
            />

            <Navbar
                onClearHistory={clearHistory}
                historyCount={history.length}
                onToggleFileUploader={() =>
                    setState((prev) => ({
                        ...prev,
                        showFileUploader: !prev.showFileUploader,
                    }))
                }
            />

            {state.showDatasetViewer && state.datasetInfo && (
                <DatasetViewer
                    isOpen={state.showDatasetViewer}
                    datasetInfo={state.datasetInfo}
                    onClose={handleCloseDatasetViewer}
                />
            )}

            <div className="mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="lg:col-span-1 space-y-8"
                    >
                        {/* File Uploader Section */}
                        {state.showFileUploader && (
                            <motion.div
                                initial={{ opacity: 0, y: -20 }}
                                animate={{ opacity: 1, y: 0 }}
                            >
                                <FileUploader
                                    onUploadSuccess={handleDatasetUpload}
                                    onUploadError={handleUploadError}
                                    isLoading={state.isLoading}
                                />
                            </motion.div>
                        )}

                        {/* Enhanced Query Interface */}
                        <EnhancedQueryInterface
                            onAnalyze={handleAnalyzeQuery}
                            onChartRequest={handleChartRequest}
                            onDatasetView={handleOpenDatasetViewer}
                            isAnalyzing={state.isAnalyzing}
                            examples={state.exampleQueries}
                        />

                        <div className="w-full px-4 py-6">
                            <QueryHistory
                                history={history}
                                activeQueryId={activeQuery?.id || null}
                                addQuery={addQuery}
                                onSelectQuery={selectQuery}
                                onDeleteQuery={deleteQuery}
                                onClearHistory={clearHistory}
                            />
                        </div>

                        {/* Analysis Results */}
                        <AnimatePresence mode="wait">
                            {state.isAnalyzing ? (
                                <LoadingSkeleton key="loading" />
                            ) : displayResult ? (
                                <motion.div
                                    key="results"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    className="space-y-8"
                                >
                                    {/* Success Message */}
                                    {displayResult.success && (
                                        <motion.div
                                            initial={{
                                                opacity: 0,
                                                scale: 0.95,
                                            }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            className="glass rounded-2xl p-6 border border-success-500/30 backdrop-blur-xl shadow-lg"
                                        >
                                            <div className="flex items-center space-x-3">
                                                <div className="w-12 h-12 bg-gradient-to-r from-success-500 to-success-600 rounded-full flex items-center justify-center">
                                                    <CheckCircle className="h-6 w-6 text-white" />
                                                </div>
                                                <div>
                                                    <p className="text-white font-bold text-lg">
                                                        Analysis Completed
                                                        Successfully! ✨
                                                    </p>
                                                    <p className="text-success-200 text-sm">
                                                        Your data visualization
                                                        is ready to explore
                                                    </p>
                                                </div>
                                            </div>
                                        </motion.div>
                                    )}

                                    {/* Chart Display */}
                                    {displayResult.success &&
                                        displayResult.chart_data && (
                                            <motion.div
                                                initial={{ opacity: 0, y: 20 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                transition={{ delay: 0.2 }}
                                                className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 border"
                                            >
                                                <div className="flex items-center gap-2 mb-4">
                                                    <BarChart3 className="h-5 w-5 text-blue-600" />
                                                    <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                                                        Chart Result
                                                    </h3>
                                                </div>

                                                {displayResult.chart_data && (
                                                    <div className="w-full flex justify-center">
                                                        {typeof displayResult.chart_data ===
                                                            "string" &&
                                                        displayResult.chart_data.startsWith(
                                                            "data:image/"
                                                        ) ? (
                                                            <img
                                                                src={
                                                                    displayResult.chart_data
                                                                }
                                                                alt="Generated Chart"
                                                                className="max-w-full h-auto rounded-lg shadow-lg"
                                                                style={{
                                                                    maxHeight:
                                                                        "600px",
                                                                }}
                                                            />
                                                        ) : typeof displayResult.chart_data ===
                                                              "object" &&
                                                          displayResult
                                                              .chart_data
                                                              .data ? (
                                                            <div
                                                                className="w-full"
                                                                style={{
                                                                    height: "500px",
                                                                }}
                                                            >
                                                                <Plot
                                                                    data={
                                                                        displayResult
                                                                            .chart_data
                                                                            .data
                                                                    }
                                                                    layout={{
                                                                        ...displayResult
                                                                            .chart_data
                                                                            .layout,
                                                                        autosize:
                                                                            true,
                                                                        responsive:
                                                                            true,
                                                                    }}
                                                                    config={{
                                                                        responsive:
                                                                            true,
                                                                        displayModeBar:
                                                                            true,
                                                                        displaylogo:
                                                                            false,
                                                                    }}
                                                                    style={{
                                                                        width: "100%",
                                                                        height: "100%",
                                                                    }}
                                                                />
                                                            </div>
                                                        ) : (
                                                            <div className="w-full">
                                                                <p className="text-gray-600">
                                                                    Chart
                                                                    generated
                                                                    successfully!
                                                                </p>
                                                                <details className="mt-2">
                                                                    <summary className="cursor-pointer text-blue-600">
                                                                        View
                                                                        technical
                                                                        details
                                                                    </summary>
                                                                    <pre className="mt-2 p-2 bg-gray-100 rounded text-xs overflow-auto max-h-40">
                                                                        {JSON.stringify(
                                                                            displayResult.chart_data,
                                                                            null,
                                                                            2
                                                                        )}
                                                                    </pre>
                                                                </details>
                                                            </div>
                                                        )}
                                                    </div>
                                                )}
                                                {displayResult.summary && (
                                                    <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                                                        <p className="text-gray-700 dark:text-gray-300">
                                                            {
                                                                displayResult.summary
                                                            }
                                                        </p>
                                                    </div>
                                                )}
                                            </motion.div>
                                        )}

                                    {/* Data Table (when no chart) */}
                                    {displayResult.success &&
                                        displayResult.data &&
                                        !displayResult.chart_data && (
                                            <motion.div
                                                initial={{ opacity: 0, y: 20 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                transition={{ delay: 0.2 }}
                                                className="glass rounded-2xl shadow-2xl p-6 border border-white/20 backdrop-blur-xl"
                                            >
                                                <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                                                    <Database className="h-6 w-6 text-white mr-3" />
                                                    Data Results
                                                </h3>
                                                <div className="overflow-x-auto">
                                                    <table className="w-full text-white">
                                                        <thead>
                                                            <tr className="border-b border-white/20">
                                                                {Object.keys(
                                                                    displayResult
                                                                        .data[0] ||
                                                                        {}
                                                                ).map((key) => (
                                                                    <th
                                                                        key={
                                                                            key
                                                                        }
                                                                        className="text-left p-2 text-white/80"
                                                                    >
                                                                        {key}
                                                                    </th>
                                                                ))}
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {displayResult.data
                                                                .slice(0, 10)
                                                                .map(
                                                                    (
                                                                        row: any,
                                                                        index: number
                                                                    ) => (
                                                                        <tr
                                                                            key={
                                                                                index
                                                                            }
                                                                            className="border-b border-white/10"
                                                                        >
                                                                            {Object.values(
                                                                                row
                                                                            ).map(
                                                                                (
                                                                                    value: any,
                                                                                    cellIndex: number
                                                                                ) => (
                                                                                    <td
                                                                                        key={
                                                                                            cellIndex
                                                                                        }
                                                                                        className="p-2 text-white/90"
                                                                                    >
                                                                                        {typeof value ===
                                                                                        "number"
                                                                                            ? value.toLocaleString()
                                                                                            : String(
                                                                                                  value
                                                                                              )}
                                                                                    </td>
                                                                                )
                                                                            )}
                                                                        </tr>
                                                                    )
                                                                )}
                                                        </tbody>
                                                    </table>
                                                    {displayResult.data.length >
                                                        10 && (
                                                        <p className="text-white/60 text-sm mt-2">
                                                            Showing first 10 of{" "}
                                                            {
                                                                displayResult
                                                                    .data.length
                                                            }{" "}
                                                            rows
                                                        </p>
                                                    )}
                                                </div>
                                            </motion.div>
                                        )}
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="welcome"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="glass rounded-2xl p-12 text-center border border-white/20 backdrop-blur-xl"
                                >
                                    <BarChart3 className="h-16 w-16 text-white mx-auto mb-6 opacity-50" />
                                    <h2 className="text-3xl font-bold text-white mb-4">
                                        Welcome to Data Analysis
                                    </h2>
                                    <p className="text-white/70 text-lg max-w-2xl mx-auto">
                                        Ask questions about your dataset in
                                        natural language, and I'll create
                                        beautiful visualizations for you. Use
                                        Chart Mode to create specific
                                        visualizations.
                                    </p>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                </motion.div>
            </div>
        </div>
    );
}

export default function AppWithErrorBoundary() {
    return (
        <ErrorBoundary>
            <App />
        </ErrorBoundary>
    );
}

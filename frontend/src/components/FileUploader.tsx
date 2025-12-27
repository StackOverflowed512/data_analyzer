import React, { useState, useRef } from "react";
import { Upload, AlertCircle, CheckCircle, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import toast from "react-hot-toast";

interface FileUploaderProps {
    onUploadSuccess: (filename: string) => void;
    onUploadError: (error: string) => void;
    isLoading?: boolean;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
    onUploadSuccess,
    onUploadError,
    isLoading = false,
}) => {
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragEnter = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    };

    const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.currentTarget.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    };

    const handleFileSelect = async (file: File) => {
        // Validate file type
        if (!file.name.endsWith(".csv")) {
            onUploadError("Please select a CSV file (.csv)");
            toast.error("❌ Only CSV files are supported");
            return;
        }

        // Validate file size (max 50MB)
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            onUploadError("File size exceeds 50MB limit");
            toast.error("❌ File is too large (max 50MB)");
            return;
        }

        setIsUploading(true);

        try {
            const formData = new FormData();
            formData.append("file", file);

            console.log(`📤 Uploading file: ${file.name}`);

            const response = await fetch(
                `${import.meta.env.VITE_API_URL}/api/upload-dataset`,
                {
                    method: "POST",
                    body: formData,
                }
            );

            // Check response status first
            if (!response.ok) {
                let errorData;
                try {
                    errorData = await response.json();
                } catch (parseError) {
                    errorData = {
                        error: `HTTP ${response.status}: ${response.statusText}`,
                    };
                }
                throw new Error(
                    errorData.error ||
                        `Upload failed with status ${response.status}`
                );
            }

            // Parse response text first to debug JSON issues
            const responseText = await response.text();
            console.log("📨 Raw response:", responseText.substring(0, 500));

            let result;
            try {
                result = JSON.parse(responseText);
            } catch (parseError) {
                console.error("❌ JSON parse error:", parseError);
                console.error(
                    "Response text:",
                    responseText.substring(0, 1000)
                );
                throw new Error(
                    "Server returned invalid response. Check file encoding and format."
                );
            }

            if (result.success) {
                console.log("✅ File uploaded successfully:", result);
                toast.success(
                    `✨ Dataset "${file.name}" uploaded successfully!`,
                    {
                        style: {
                            background: "rgba(34, 197, 94, 0.9)",
                            color: "white",
                            backdropFilter: "blur(10px)",
                        },
                    }
                );
                onUploadSuccess(file.name);
            } else {
                throw new Error(result.error || "Upload failed");
            }
        } catch (error) {
            const errorMessage =
                error instanceof Error ? error.message : "Upload failed";
            console.error("❌ Upload error:", errorMessage);
            onUploadError(errorMessage);

            let detailedMessage = `❌ Upload failed: ${errorMessage}`;

            // Provide specific guidance based on error type
            if (
                errorMessage.includes("CSV format") ||
                errorMessage.includes("encoding") ||
                errorMessage.includes("JSON")
            ) {
                detailedMessage +=
                    "\n\nTroubleshooting:\n• Save your CSV as UTF-8 encoding\n• Check that all characters are valid\n• Try opening in Excel and re-saving\n• Ensure no special characters in headers\n• Remove any formula cells (should be values only)";
            }

            toast.error(detailedMessage, {
                style: {
                    background: "rgba(239, 68, 68, 0.9)",
                    color: "white",
                    backdropFilter: "blur(10px)",
                },
                duration: 6000,
            });
        } finally {
            setIsUploading(false);
            // Reset file input
            if (fileInputRef.current) {
                fileInputRef.current.value = "";
            }
        }
    };

    return (
        <Card className="w-full border-dashed">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-white">
                    <Upload className="h-5 w-5" />
                    Upload CSV Dataset
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div
                    onDragEnter={handleDragEnter}
                    onDragLeave={handleDragLeave}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
                        isDragging
                            ? "border-blue-500 bg-blue-500/10"
                            : "border-gray-400 hover:border-blue-500"
                    }`}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv"
                        onChange={handleFileInputChange}
                        className="hidden"
                        disabled={isUploading || isLoading}
                    />

                    {isUploading ? (
                        <div className="space-y-3">
                            <Loader2 className="h-8 w-8 animate-spin text-blue-500 mx-auto" />
                            <p className="text-gray-600">
                                Uploading your dataset...
                            </p>
                        </div>
                    ) : (
                        <div
                            onClick={() => fileInputRef.current?.click()}
                            className="space-y-3"
                        >
                            <Upload className="h-8 w-8 text-gray-400 mx-auto" />
                            <div>
                                <p className="text-gray-700 font-medium">
                                    Drag and drop your CSV file here
                                </p>
                                <p className="text-gray-500 text-sm mt-1">
                                    or click to browse
                                </p>
                            </div>
                            <p className="text-gray-400 text-xs">
                                CSV files up to 50MB are supported
                            </p>
                        </div>
                    )}
                </div>

                <div className="mt-4 space-y-2 text-sm text-gray-600">
                    <p className="font-medium text-gray-700">Requirements:</p>
                    <ul className="list-disc list-inside space-y-1 text-gray-600">
                        <li>File must be in CSV format (.csv)</li>
                        <li>Maximum file size: 50MB</li>
                        <li>First row should contain column headers</li>
                        <li>Supports text and numeric data</li>
                    </ul>
                </div>
            </CardContent>
        </Card>
    );
};

export default FileUploader;

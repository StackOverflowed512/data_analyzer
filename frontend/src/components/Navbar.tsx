import React from "react";
import { BarChart3, Trash2, Upload } from "lucide-react";
import { Button } from "./ui/button";

import DatasetSelector from "./DatasetSelector";

interface NavbarProps {
    onClearHistory: () => void;
    historyCount: number;
    onToggleFileUploader?: () => void;
    currentDatasetName: string;
    currentDatasetId?: number;
    onDatasetSwitch: (name: string, info: any) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
    onClearHistory,
    historyCount,
    onToggleFileUploader,
    currentDatasetName = "Dataset",
    currentDatasetId,
    onDatasetSwitch,
}) => {
    return (
        <nav className="glass border-b border-white/20 backdrop-blur-xl sticky top-0 z-40">
            <div className="mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    {/* Left section */}
                    <div className="flex items-center space-x-3">
                        <BarChart3 className="h-8 w-8 text-white" />
                        <h1 className="text-2xl font-bold text-white">
                            InsightGPT
                        </h1>
                    </div>

                    {/* Center section - Dataset Selector */}
                     <div className="flex-1 flex justify-center px-4">
                        <DatasetSelector 
                            currentDatasetName={currentDatasetName}
                            currentDatasetId={currentDatasetId}
                            onDatasetSwitch={onDatasetSwitch}
                        />
                    </div>

                    {/* Right section */}
                    <div className="flex items-center space-x-4">
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={onToggleFileUploader}
                            className="border-blue-400 text-blue-400 hover:bg-blue-500/10"
                        >
                            <Upload className="h-4 w-4 mr-2" />
                            Upload Dataset
                        </Button>

                        <Button
                            variant="outline"
                            size="sm"
                            onClick={onClearHistory}
                            className="border-red-400 text-red-400 hover:bg-red-500/10"
                            disabled={historyCount === 0}
                        >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Clear History ({historyCount})
                        </Button>
                    </div>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;

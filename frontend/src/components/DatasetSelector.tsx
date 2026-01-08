import React, { useEffect, useState } from 'react';
import { apiService, type DatasetItem } from '../services/api';
import { Database, Calendar, FileText, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import toast from 'react-hot-toast';

interface DatasetSelectorProps {
    currentDatasetName: string;
    currentDatasetId?: number;
    onDatasetSwitch: (name: string, info: any) => void;
}

const DatasetSelector: React.FC<DatasetSelectorProps> = ({ currentDatasetName, currentDatasetId, onDatasetSwitch }) => {
    const [datasets, setDatasets] = useState<DatasetItem[]>([]);
    const [isOpen, setIsOpen] = useState(false);
    const [loading, setLoading] = useState(false);
    const [switching, setSwitching] = useState<number | null>(null);

    useEffect(() => {
        if (isOpen) {
            loadDatasets();
        }
    }, [isOpen]);

    const loadDatasets = async () => {
        setLoading(true);
        try {
            const result = await apiService.listDatasets();
            if (result.success) {
                setDatasets(result.datasets);
            }
        } catch (error) {
            console.error('Failed to load datasets:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSwitch = async (id: number, filename: string) => {
        if (filename === currentDatasetName) return;
        
        setSwitching(id);
        const toastId = toast.loading(`Switching to ${filename}...`);
        
        try {
            const result = await apiService.switchDataset(id);
            if (result.success) {
                toast.success(`Switched to ${filename}`, { id: toastId });
                onDatasetSwitch(filename, result.info);
                setIsOpen(false);
            } else {
                toast.error(`Failed to switch: ${result.error}`, { id: toastId });
            }
        } catch (error) {
            toast.error('Failed to switch dataset', { id: toastId });
        } finally {
            setSwitching(null);
        }
    };

    return (
        <div className="relative z-50">
            <Button
                variant="outline"
                className="flex items-center space-x-2 bg-white/10 border-white/20 text-white hover:bg-white/20 backdrop-blur-sm"
                onClick={() => setIsOpen(!isOpen)}
            >
                <Database className="h-4 w-4" />
                <span>{currentDatasetName}</span>
            </Button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                        className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-900 rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700 overflow-hidden"
                    >
                        <div className="p-3 border-b border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
                            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 flex items-center">
                                <Database className="h-4 w-4 mr-2 text-primary-500" />
                                Available Datasets
                            </h3>
                        </div>

                        <div className="max-h-64 overflow-y-auto p-2 space-y-1">
                            {loading ? (
                                <div className="p-4 text-center text-sm text-gray-500">Loading datasets...</div>
                            ) : datasets.length === 0 ? (
                                <div className="p-4 text-center text-sm text-gray-500">No stored datasets found.</div>
                            ) : (
                                datasets.map((ds) => (
                                    <button
                                        key={ds.id}
                                        onClick={() => handleSwitch(ds.id, ds.filename)}
                                        disabled={switching !== null}
                                        className={`w-full text-left p-2 rounded-lg text-sm transition-colors flex items-center justify-between group
                                            ${(currentDatasetId === ds.id || (!currentDatasetId && ds.filename === currentDatasetName))
                                                ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300' 
                                                : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'}
                                        `}
                                    >
                                        <div className="flex-1 min-w-0">
                                            <div className="font-medium truncate flex items-center">
                                                <FileText className="h-3 w-3 mr-1.5 opacity-70" />
                                                {ds.filename}
                                            </div>
                                            <div className="text-xs text-gray-500 flex items-center mt-0.5">
                                                <Calendar className="h-3 w-3 mr-1" />
                                                {new Date(ds.upload_date).toLocaleDateString()}
                                                <span className="mx-1">•</span>
                                                {ds.row_count} rows
                                            </div>
                                        </div>
                                        {(currentDatasetId === ds.id || (!currentDatasetId && ds.filename === currentDatasetName)) && (
                                            <Check className="h-4 w-4 text-primary-600 dark:text-primary-400" />
                                        )}
                                        {switching === ds.id && (
                                            <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-500 border-t-transparent" />
                                        )}
                                    </button>
                                ))
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
            
            {/* Backdrop to close on click outside */}
            {isOpen && (
                <div 
                    className="fixed inset-0 z-[-1]" 
                    onClick={() => setIsOpen(false)}
                />
            )}
        </div>
    );
};

export default DatasetSelector;

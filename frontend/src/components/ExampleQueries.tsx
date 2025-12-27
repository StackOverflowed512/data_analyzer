import React from "react";
import { motion } from "framer-motion";
import type { ExampleQuery } from "../services/api";
import { Lightbulb, MessageSquare, Sparkles, Zap } from "lucide-react";

interface ExampleQueriesProps {
  examples: ExampleQuery[];
  onSelectExample: (query: string) => void;
}

const ExampleQueries: React.FC<ExampleQueriesProps> = ({
  examples,
  onSelectExample,
}) => {
  const queryIcons = ["🚗", "📊", "⚡", "🔍", "📈", "💡", "🎯", "🌟"];
  const gradients = [
    "white",
    "white",
    "white",
    "white",
    "white",
    "black",
    "from-pink-500 to-rose-600",
    "from-cyan-500 to-blue-600",
  ];

  return (
    <motion.div
      className="glass rounded-2xl shadow-2xl p-6 border border-white/20 backdrop-blur-xl"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
    >
      <motion.div
        className="flex items-center mb-6"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
      >
        <div className="relative">
          <Lightbulb className="h-7 w-7 text-white mr-3" />
          <div className="absolute inset-0 h-7 w-7 bg-gradient-to-r from-yellow-400 to-orange-400 rounded-full blur-lg opacity-50 animate-pulse"></div>
        </div>
        <h3 className="text-xl font-bold text-white">Smart Suggestions</h3>
        <Sparkles className="h-5 w-5 text-yellow-300 ml-2 animate-bounce" />
      </motion.div>

      <div className=" md:grid grid-cols-3 gap-4">
        {examples.map((example, index) => (
          <motion.button
            key={index}
            onClick={() => onSelectExample(example.query)}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 + index * 0.1 }}
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            className={` w-full text-left p-4 mt-4 h-36 rounded-xl bg-gradient-to-r ${
              gradients[index % gradients.length]
            } bg-opacity-20 backdrop-blur-sm border border-white/20 hover:border-white/40 transition-all duration-300 group shadow-lg`}
          >
            <div className="flex items-center space-x-3">
              <motion.div
                className="flex-shrink-0 w-10 h-10 bg-white/10 backdrop-blur-sm rounded-xl flex items-center justify-center border border-white/20 group-hover:scale-110 transition-transform duration-300"
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
              >
                <span className="text-xl">
                  {queryIcons[index % queryIcons.length]}
                </span>
              </motion.div>
              <div className="flex-1 min-w-0">
                <p className="text-white font-semibold group-hover:text-white transition-colors duration-300 mb-1">
                  {example.query}
                </p>
                <p className="text-white/70 text-sm group-hover:text-white/90 transition-colors duration-300">
                  {example.description}
                </p>
              </div>
              <motion.div
                className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                whileHover={{ x: 5 }}
              >
                <Zap className="h-5 w-5 text-yellow-300" />
              </motion.div>
            </div>
          </motion.button>
        ))}
      </div>

      <motion.div
        className="mt-6 p-4 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <div className="flex items-center space-x-2 text-white/80 text-sm">
          <MessageSquare className="h-4 w-4" />
          <span>Click any suggestion to get started instantly!</span>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ExampleQueries;

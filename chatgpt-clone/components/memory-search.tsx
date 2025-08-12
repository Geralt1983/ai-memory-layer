"use client"

import { useState, useEffect } from 'react';
import { Search, Brain, Clock, Hash } from 'lucide-react';
import { getMemoryClient, type Memory, type MemoryStats } from '@/lib/memory-client';

interface MemorySearchProps {
  onMemorySelect?: (memory: Memory) => void;
}

export default function MemorySearch({ onMemorySelect }: MemorySearchProps) {
  const [query, setQuery] = useState('');
  const [memories, setMemories] = useState<Memory[]>([]);
  const [stats, setStats] = useState<MemoryStats>({ available: false });
  const [isSearching, setIsSearching] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const memoryClient = getMemoryClient();

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    const statsData = await memoryClient.getStats();
    setStats(statsData);
  };

  const searchMemories = async () => {
    if (!query.trim() || !stats.available) return;

    setIsSearching(true);
    try {
      const result = await memoryClient.searchMemories({ query, limit: 10 });
      setMemories(result.memories);
      setIsExpanded(true);
    } catch (error) {
      console.error('Memory search failed:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      searchMemories();
    } else if (e.key === 'Escape') {
      setIsExpanded(false);
    }
  };

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return '';
    return new Date(timestamp).toLocaleDateString();
  };

  if (!stats.available) {
    return (
      <div className="p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
        <div className="flex items-center space-x-2 text-amber-600 dark:text-amber-400">
          <Brain className="w-4 h-4" />
          <span className="text-sm">Memory system not available</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Stats Header */}
      <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
        <div className="flex items-center space-x-2">
          <Brain className="w-4 h-4" />
          <span>Memory Search</span>
        </div>
        {stats.total_memories && (
          <span>{stats.total_memories} memories</span>
        )}
      </div>

      {/* Search Input */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          placeholder="Search memories..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          className="w-full pl-10 pr-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
        />
        {isSearching && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            <div className="animate-spin w-4 h-4 border-2 border-gray-300 border-t-blue-500 rounded-full"></div>
          </div>
        )}
      </div>

      {/* Search Results */}
      {isExpanded && memories.length > 0 && (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          <div className="text-xs text-gray-500 dark:text-gray-400 px-1">
            Found {memories.length} memories
          </div>
          {memories.map((memory, index) => (
            <div
              key={index}
              onClick={() => onMemorySelect?.(memory)}
              className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="text-sm text-gray-900 dark:text-gray-100 line-clamp-3">
                {memory.content}
              </div>
              <div className="flex items-center justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
                <div className="flex items-center space-x-3">
                  {memory.relevance_score && (
                    <div className="flex items-center space-x-1">
                      <Hash className="w-3 h-3" />
                      <span>{Math.round(memory.relevance_score * 100)}%</span>
                    </div>
                  )}
                  {memory.type && (
                    <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded">
                      {memory.type}
                    </span>
                  )}
                </div>
                {memory.timestamp && (
                  <div className="flex items-center space-x-1">
                    <Clock className="w-3 h-3" />
                    <span>{formatTimestamp(memory.timestamp)}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {isExpanded && memories.length === 0 && query.trim() && (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No memories found for "{query}"</p>
        </div>
      )}
    </div>
  );
}
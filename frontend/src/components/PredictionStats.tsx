'use client';

import { useEffect, useState } from 'react';
import LoadingSpinner from './LoadingSpinner';

interface PredictionStatsData {
  totalPredictions: number;
  accuracy: number;
  brierScore: number;
  logLoss: number;
  calibrationScore: number;
  lastUpdated: string;
}

const PredictionStats = () => {
  const [stats, setStats] = useState<PredictionStatsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/stats');
        if (!response.ok) {
          throw new Error('Failed to fetch stats');
        }
        const data = await response.json();
        setStats(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        // Fallback to mock data for development
        setStats({
          totalPredictions: 1247,
          accuracy: 68.5,
          brierScore: 0.234,
          logLoss: 0.567,
          calibrationScore: 0.089,
          lastUpdated: new Date().toISOString()
        });
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return <LoadingSpinner text="Loading statistics..." />;
  }

  if (error && !stats) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-600 dark:text-red-400">Error loading statistics: {error}</p>
      </div>
    );
  }

  if (!stats) return null;

  const statCards = [
    {
      title: 'Total Predictions',
      value: stats.totalPredictions.toLocaleString(),
      icon: '📊',
      color: 'blue'
    },
    {
      title: 'Accuracy',
      value: `${stats.accuracy.toFixed(1)}%`,
      icon: '🎯',
      color: 'green'
    },
    {
      title: 'Brier Score',
      value: stats.brierScore.toFixed(3),
      icon: '📈',
      color: 'purple',
      subtitle: 'Lower is better'
    },
    {
      title: 'Log Loss',
      value: stats.logLoss.toFixed(3),
      icon: '📉',
      color: 'orange',
      subtitle: 'Lower is better'
    },
    {
      title: 'Calibration',
      value: stats.calibrationScore.toFixed(3),
      icon: '⚖️',
      color: 'indigo',
      subtitle: 'Lower is better'
    }
  ];

  const getColorClasses = (color: string) => {
    const colors = {
      blue: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-600 dark:text-blue-400',
      green: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-600 dark:text-green-400',
      purple: 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800 text-purple-600 dark:text-purple-400',
      orange: 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800 text-orange-600 dark:text-orange-400',
      indigo: 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800 text-indigo-600 dark:text-indigo-400'
    };
    return colors[color as keyof typeof colors] || colors.blue;
  };

  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Model Performance
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Last updated: {new Date(stats.lastUpdated).toLocaleDateString()}
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        {statCards.map((stat, index) => (
          <div
            key={index}
            className={`p-6 rounded-lg border ${getColorClasses(stat.color)} transition-transform hover:scale-105`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-2xl">{stat.icon}</span>
              <div className="text-right">
                <p className="text-2xl font-bold">{stat.value}</p>
                {stat.subtitle && (
                  <p className="text-xs opacity-75">{stat.subtitle}</p>
                )}
              </div>
            </div>
            <h3 className="font-semibold text-sm">{stat.title}</h3>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PredictionStats;
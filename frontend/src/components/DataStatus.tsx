'use client';

import React, { useState, useEffect } from 'react';

interface DataSourceInfo {
  source_type: string;
  artifacts_dir: string;
  matches_file: string;
  allowed_sources: string[];
  validation_enabled: boolean;
}

interface ArtifactsStatus {
  valid: boolean;
  errors: string[];
}

interface DataHealth {
  database: {
    status: string;
    error?: string;
  };
  artifacts: {
    status: string;
    errors: string[];
  };
  data_source: DataSourceInfo;
  overall_health: string;
}

interface DataStatusProps {
  className?: string;
}

const DataStatus: React.FC<DataStatusProps> = ({ className = '' }) => {
  const [dataHealth, setDataHealth] = useState<DataHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDataHealth = async () => {
    try {
      setLoading(true);
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
      const response = await fetch(`${apiUrl}/data/health`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setDataHealth(data.health);
      setError(null);
    } catch (err) {
      console.error('Error fetching data health:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch data health');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDataHealth();
    // Refresh every 30 seconds
    const interval = setInterval(fetchDataHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'unhealthy':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-yellow-600 bg-yellow-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return '✅';
      case 'unhealthy':
        return '❌';
      default:
        return '⚠️';
    }
  };

  if (loading) {
    return (
      <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-2">
            <div className="h-3 bg-gray-200 rounded w-3/4"></div>
            <div className="h-3 bg-gray-200 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
        <div className="text-red-600">
          <h3 className="text-lg font-semibold mb-2">Data Status Error</h3>
          <p className="text-sm">{error}</p>
          <button 
            onClick={fetchDataHealth}
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!dataHealth) {
    return null;
  }

  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Data Source Status</h3>
        <button 
          onClick={fetchDataHealth}
          className="text-sm text-blue-600 hover:text-blue-800"
          title="Refresh status"
        >
          🔄
        </button>
      </div>

      {/* Overall Health */}
      <div className="mb-4">
        <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
          getStatusColor(dataHealth.overall_health)
        }`}>
          <span className="mr-2">{getStatusIcon(dataHealth.overall_health)}</span>
          Overall Status: {dataHealth.overall_health.toUpperCase()}
        </div>
      </div>

      {/* Database Status */}
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Database</span>
          <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${
            getStatusColor(dataHealth.database.status)
          }`}>
            {getStatusIcon(dataHealth.database.status)} {dataHealth.database.status}
          </span>
        </div>
        {dataHealth.database.error && (
          <p className="text-xs text-red-600 mt-1">{dataHealth.database.error}</p>
        )}
      </div>

      {/* Artifacts Status */}
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Data Files</span>
          <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${
            getStatusColor(dataHealth.artifacts.status)
          }`}>
            {getStatusIcon(dataHealth.artifacts.status)} {dataHealth.artifacts.status}
          </span>
        </div>
        {dataHealth.artifacts.errors.length > 0 && (
          <div className="mt-1">
            {dataHealth.artifacts.errors.map((error, index) => (
              <p key={index} className="text-xs text-red-600">{error}</p>
            ))}
          </div>
        )}
      </div>

      {/* Data Source Info */}
      <div className="border-t pt-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Data Source Configuration</h4>
        <div className="space-y-1 text-xs text-gray-600">
          <div className="flex justify-between">
            <span>Source Type:</span>
            <span className="font-mono bg-gray-100 px-1 rounded">
              {dataHealth.data_source.source_type}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Validation:</span>
            <span className={`font-mono px-1 rounded ${
              dataHealth.data_source.validation_enabled 
                ? 'bg-green-100 text-green-800' 
                : 'bg-red-100 text-red-800'
            }`}>
              {dataHealth.data_source.validation_enabled ? 'Enabled' : 'Disabled'}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Matches File:</span>
            <span className="font-mono bg-gray-100 px-1 rounded">
              {dataHealth.data_source.matches_file}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataStatus;
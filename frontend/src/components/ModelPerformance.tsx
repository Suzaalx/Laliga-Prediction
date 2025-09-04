'use client';

import { useEffect, useState } from 'react';
import LoadingSpinner from './LoadingSpinner';

interface PerformanceData {
  calibrationCurve: {
    predicted: number[];
    actual: number[];
  };
  monthlyAccuracy: {
    month: string;
    accuracy: number;
    predictions: number;
  }[];
  confusionMatrix: {
    homeWin: { predicted: number; actual: number };
    draw: { predicted: number; actual: number };
    awayWin: { predicted: number; actual: number };
  };
  recentPerformance: {
    date: string;
    accuracy: number;
    brierScore: number;
  }[];
}

const ModelPerformance = () => {
  const [data, setData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'calibration' | 'accuracy' | 'confusion' | 'recent'>('calibration');

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        const response = await fetch('/api/model/performance');
        if (!response.ok) {
          throw new Error('Failed to fetch performance data');
        }
        const performanceData = await response.json();
        setData(performanceData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        // Fallback to mock data for development
        setData({
          calibrationCurve: {
            predicted: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            actual: [0.08, 0.18, 0.32, 0.41, 0.52, 0.58, 0.72, 0.79, 0.91]
          },
          monthlyAccuracy: [
            { month: 'Jan', accuracy: 65.2, predictions: 45 },
            { month: 'Feb', accuracy: 68.1, predictions: 42 },
            { month: 'Mar', accuracy: 71.3, predictions: 48 },
            { month: 'Apr', accuracy: 69.8, predictions: 46 },
            { month: 'May', accuracy: 72.5, predictions: 44 },
            { month: 'Jun', accuracy: 70.1, predictions: 41 }
          ],
          confusionMatrix: {
            homeWin: { predicted: 156, actual: 142 },
            draw: { predicted: 89, actual: 95 },
            awayWin: { predicted: 134, actual: 148 }
          },
          recentPerformance: [
            { date: '2024-01-01', accuracy: 68.5, brierScore: 0.234 },
            { date: '2024-01-08', accuracy: 71.2, brierScore: 0.221 },
            { date: '2024-01-15', accuracy: 69.8, brierScore: 0.245 },
            { date: '2024-01-22', accuracy: 73.1, brierScore: 0.198 },
            { date: '2024-01-29', accuracy: 70.5, brierScore: 0.232 }
          ]
        });
      } finally {
        setLoading(false);
      }
    };

    fetchPerformanceData();
  }, []);

  if (loading) {
    return <LoadingSpinner text="Loading performance data..." />;
  }

  if (error && !data) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-600 dark:text-red-400">Error loading performance data: {error}</p>
      </div>
    );
  }

  if (!data) return null;

  const renderCalibrationChart = () => {
    const maxValue = Math.max(...data.calibrationCurve.predicted, ...data.calibrationCurve.actual);
    
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Calibration Curve</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Shows how well predicted probabilities match actual outcomes. Perfect calibration follows the diagonal line.
        </p>
        <div className="relative h-64 bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <svg className="w-full h-full" viewBox="0 0 300 200">
            {/* Grid lines */}
            {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((value, i) => (
              <g key={i}>
                <line
                  x1={value * 250 + 25}
                  y1={25}
                  x2={value * 250 + 25}
                  y2={175}
                  stroke="currentColor"
                  strokeWidth="0.5"
                  className="text-gray-300 dark:text-gray-600"
                />
                <line
                  x1={25}
                  y1={175 - value * 150}
                  x2={275}
                  y2={175 - value * 150}
                  stroke="currentColor"
                  strokeWidth="0.5"
                  className="text-gray-300 dark:text-gray-600"
                />
              </g>
            ))}
            
            {/* Perfect calibration line */}
            <line
              x1={25}
              y1={175}
              x2={275}
              y2={25}
              stroke="currentColor"
              strokeWidth="2"
              strokeDasharray="5,5"
              className="text-gray-400 dark:text-gray-500"
            />
            
            {/* Actual calibration curve */}
            <polyline
              points={data.calibrationCurve.predicted.map((pred, i) => 
                `${pred * 250 + 25},${175 - data.calibrationCurve.actual[i] * 150}`
              ).join(' ')}
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              className="text-blue-600 dark:text-blue-400"
            />
            
            {/* Data points */}
            {data.calibrationCurve.predicted.map((pred, i) => (
              <circle
                key={i}
                cx={pred * 250 + 25}
                cy={175 - data.calibrationCurve.actual[i] * 150}
                r="4"
                fill="currentColor"
                className="text-blue-600 dark:text-blue-400"
              />
            ))}
          </svg>
        </div>
      </div>
    );
  };

  const renderMonthlyAccuracy = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Monthly Accuracy</h3>
      <div className="space-y-3">
        {data.monthlyAccuracy.map((month, i) => (
          <div key={i} className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300 w-8">
                {month.month}
              </span>
              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2 w-32">
                <div
                  className="bg-blue-600 dark:bg-blue-400 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${month.accuracy}%` }}
                ></div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm font-semibold text-gray-900 dark:text-white">
                {month.accuracy.toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {month.predictions} predictions
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderConfusionMatrix = () => {
    const total = data.confusionMatrix.homeWin.predicted + 
                 data.confusionMatrix.draw.predicted + 
                 data.confusionMatrix.awayWin.predicted;
    
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Prediction Distribution</h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {data.confusionMatrix.homeWin.predicted}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Home Wins</div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              ({((data.confusionMatrix.homeWin.predicted / total) * 100).toFixed(1)}%)
            </div>
          </div>
          <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
              {data.confusionMatrix.draw.predicted}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Draws</div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              ({((data.confusionMatrix.draw.predicted / total) * 100).toFixed(1)}%)
            </div>
          </div>
          <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {data.confusionMatrix.awayWin.predicted}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Away Wins</div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              ({((data.confusionMatrix.awayWin.predicted / total) * 100).toFixed(1)}%)
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderRecentPerformance = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Performance</h3>
      <div className="space-y-3">
        {data.recentPerformance.map((week, i) => (
          <div key={i} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {new Date(week.date).toLocaleDateString()}
            </div>
            <div className="flex space-x-4">
              <div className="text-right">
                <div className="text-sm font-semibold text-gray-900 dark:text-white">
                  {week.accuracy.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Accuracy</div>
              </div>
              <div className="text-right">
                <div className="text-sm font-semibold text-gray-900 dark:text-white">
                  {week.brierScore.toFixed(3)}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Brier Score</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const tabs = [
    { id: 'calibration', label: 'Calibration', component: renderCalibrationChart },
    { id: 'accuracy', label: 'Accuracy', component: renderMonthlyAccuracy },
    { id: 'confusion', label: 'Distribution', component: renderConfusionMatrix },
    { id: 'recent', label: 'Recent', component: renderRecentPerformance }
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Model Performance
        </h2>
        
        {/* Tab Navigation */}
        <div className="flex space-x-1 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as 'calibration' | 'accuracy' | 'confusion' | 'recent')}
              className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="p-6">
        {tabs.find(tab => tab.id === activeTab)?.component()}
      </div>
    </div>
  );
};

export default ModelPerformance;
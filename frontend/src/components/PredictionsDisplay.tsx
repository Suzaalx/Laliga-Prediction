'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Loader2, Calendar, TrendingUp, Target, AlertCircle } from 'lucide-react';

interface Prediction {
  date: string;
  home_team: string;
  away_team: string;
  prediction: string;
  probability: number;
  confidence: string;
}

const PredictionsDisplay: React.FC = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  useEffect(() => {
    // Set initial timestamp on client
    setLastUpdated(new Date().toLocaleString());
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      
      // For now, we'll load from our local predictions file
      // In a real app, this would be an API endpoint
      const response = await fetch('/api/predictions');
      
      if (!response.ok) {
        // Fallback to mock data if API is not available
        const mockPredictions: Prediction[] = [
          {
            date: '2025-01-15',
            home_team: 'Real Madrid',
            away_team: 'Barcelona',
            prediction: 'Over 2.5',
            probability: 0.72,
            confidence: 'High'
          },
          {
            date: '2025-01-16',
            home_team: 'Atletico Madrid',
            away_team: 'Sevilla',
            prediction: 'Under 2.5',
            probability: 0.65,
            confidence: 'Medium'
          },
          {
            date: '2025-01-17',
            home_team: 'Valencia',
            away_team: 'Real Sociedad',
            prediction: 'Over 2.5',
            probability: 0.58,
            confidence: 'Medium'
          },
          {
            date: '2025-01-18',
            home_team: 'Villarreal',
            away_team: 'Athletic Bilbao',
            prediction: 'Under 2.5',
            probability: 0.61,
            confidence: 'Medium'
          },
          {
            date: '2025-01-19',
            home_team: 'Real Betis',
            away_team: 'Getafe',
            prediction: 'Over 2.5',
            probability: 0.69,
            confidence: 'High'
          }
        ];
        setPredictions(mockPredictions);
        // lastUpdated is already set in useEffect
        setLoading(false);
        return;
      }

      const data = await response.json();
      setPredictions(data.predictions || []);
      setLastUpdated(data.last_updated || new Date().toLocaleString());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load predictions');
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence: string) => {
    switch (confidence.toLowerCase()) {
      case 'high':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300';
      case 'low':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300';
    }
  };

  const getPredictionColor = (prediction: string) => {
    return prediction.includes('Over') 
      ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300'
      : 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300';
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric'
    });
  };

  if (loading) {
    return (
      <Card className="w-full fade-in">
        <CardContent className="flex items-center justify-center p-8">
          <Loader2 className="h-8 w-8 animate-spin mr-2 shadow-glow" />
          <span className="gradient-text">Loading predictions...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full glass border-red-200 dark:border-red-800 shadow-glow-red slide-up">
        <CardContent className="p-6">
          <div className="text-center text-red-600 dark:text-red-400">
            <AlertCircle className="h-8 w-8 mx-auto mb-2" />
            <p className="font-semibold gradient-text-danger">Error loading predictions</p>
            <p className="text-sm mt-1">{error}</p>
            <button 
              onClick={fetchPredictions} 
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
            >
              Try Again
            </button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <Card className="glass">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 gradient-text">
            <Target className="h-5 w-5" />
            Upcoming Match Predictions
          </CardTitle>
          <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              <span>Last updated: {lastUpdated}</span>
            </div>
            <div className="flex items-center gap-1">
              <TrendingUp className="h-4 w-4" />
              <span>{predictions.length} predictions available</span>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Predictions Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {predictions.map((prediction, index) => (
          <Card key={index} className="glass card-hover">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <Badge variant="outline" className="text-xs">
                  {formatDate(prediction.date)}
                </Badge>
                <Badge className={getConfidenceColor(prediction.confidence)}>
                  {prediction.confidence}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Match Details */}
                <div className="text-center">
                  <div className="font-semibold text-lg gradient-text">
                    {prediction.home_team}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400 my-1">vs</div>
                  <div className="font-semibold text-lg gradient-text">
                    {prediction.away_team}
                  </div>
                </div>

                {/* Prediction */}
                <div className="text-center space-y-2">
                  <Badge className={`${getPredictionColor(prediction.prediction)} text-sm px-3 py-1`}>
                    {prediction.prediction}
                  </Badge>
                  <div className="text-2xl font-bold gradient-text-success">
                    {(prediction.probability * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Probability
                  </div>
                </div>

                {/* Confidence Bar */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                    <span>Confidence</span>
                    <span>{prediction.confidence}</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-300 shadow-glow ${
                        prediction.confidence === 'High' ? 'bg-gradient-to-r from-green-500 to-emerald-500 w-5/6' :
                        prediction.confidence === 'Medium' ? 'bg-gradient-to-r from-yellow-500 to-orange-500 w-3/5' :
                        'bg-gradient-to-r from-red-500 to-pink-500 w-2/5'
                      }`}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {predictions.length === 0 && (
        <Card>
          <CardContent className="text-center py-8">
            <Target className="h-12 w-12 mx-auto text-gray-400 mb-4" />
            <p className="text-gray-500 dark:text-gray-400">No predictions available at the moment.</p>
            <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
              Check back later for upcoming match predictions.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default PredictionsDisplay;
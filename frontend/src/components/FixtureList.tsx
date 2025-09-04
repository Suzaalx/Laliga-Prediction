'use client';

import { useEffect, useState } from 'react';
import LoadingSpinner from './LoadingSpinner';

interface Fixture {
  id: string;
  date: string;
  homeTeam: string;
  awayTeam: string;
  homeWinProb?: number;
  drawProb?: number;
  awayWinProb?: number;
  predictedScore?: {
    home: number;
    away: number;
  };
  confidence?: number;
  venue: string;
  status: 'upcoming' | 'live' | 'finished';
  actualScore?: {
    home: number;
    away: number;
  };
}

const FixtureList = () => {
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mockFixtures, setMockFixtures] = useState<Fixture[]>([]);

  useEffect(() => {
    // Generate mock fixtures with client-side dates
    const now = Date.now();
    const mockData = [
      {
        id: '1',
        date: new Date(now + 86400000).toISOString(), // Tomorrow
        homeTeam: 'Real Madrid',
        awayTeam: 'Barcelona',
        homeWinProb: 45.2,
        drawProb: 28.1,
        awayWinProb: 26.7,
        predictedScore: { home: 2, away: 1 },
        confidence: 78.5,
        venue: 'Santiago Bernabéu',
        status: 'upcoming' as const
      },
      {
        id: '2',
        date: new Date(now + 172800000).toISOString(),
        homeTeam: 'Atlético Madrid',
        awayTeam: 'Sevilla',
        homeWinProb: 52.8,
        drawProb: 25.4,
        awayWinProb: 21.8,
        predictedScore: { home: 1, away: 0 },
        confidence: 65.2,
        venue: 'Wanda Metropolitano',
        status: 'upcoming' as const
      },
      {
        id: '3',
        date: new Date(now + 259200000).toISOString(),
        homeTeam: 'Valencia',
        awayTeam: 'Real Sociedad',
        homeWinProb: 38.9,
        drawProb: 31.2,
        awayWinProb: 29.9,
        predictedScore: { home: 1, away: 1 },
        confidence: 58.7,
        venue: 'Mestalla',
        status: 'upcoming' as const
      }
    ];
    setMockFixtures(mockData);
    
    const fetchFixtures = async () => {
      try {
        const response = await fetch('/api/fixtures');
        if (!response.ok) {
          throw new Error('Failed to fetch fixtures');
        }
        const data = await response.json();
        setFixtures(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        // Use client-side generated mock data
        setFixtures(mockData);

      } finally {
        setLoading(false);
      }
    };

    fetchFixtures();
  }, []);

  const upcomingFixtures = fixtures.filter(fixture => fixture.status === 'upcoming');

  const getProbabilityColor = (prob: number) => {
    if (prob >= 50) return 'text-green-600 dark:text-green-400';
    if (prob >= 35) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 70) return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
    if (confidence >= 50) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
    return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
  };

  if (loading) {
    return <LoadingSpinner text="Loading fixtures..." />;
  }

  if (error && fixtures.length === 0) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-600 dark:text-red-400">Error loading fixtures: {error}</p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white text-center">
          Upcoming Matches
        </h2>
      </div>

      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        {upcomingFixtures.length === 0 ? (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            No fixtures found for the selected filter.
          </div>
        ) : (
          upcomingFixtures.map((fixture) => (
            <div key={fixture.id} className="p-8 hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50 dark:hover:from-gray-700/30 dark:hover:to-gray-600/30 transition-all duration-300">
              {/* Date */}
              <div className="text-center mb-6">
                <div className="text-lg font-medium text-gray-600 dark:text-gray-300">
                  {new Date(fixture.date).toLocaleDateString('en-US', {
                    weekday: 'long',
                    month: 'long',
                    day: 'numeric'
                  })}
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  {new Date(fixture.date).toLocaleTimeString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </div>
              </div>

              {/* Teams and Predicted Score */}
              <div className="flex items-center justify-center mb-8">
                <div className="text-center flex-1">
                  <div className="text-xl font-bold text-gray-900 dark:text-white mb-2">{fixture.homeTeam}</div>
                  <div className="text-3xl font-extrabold text-blue-600 dark:text-blue-400">
                    {fixture.predictedScore?.home ?? 0}
                  </div>
                </div>
                <div className="mx-8 text-2xl font-bold text-gray-400 dark:text-gray-500">VS</div>
                <div className="text-center flex-1">
                  <div className="text-xl font-bold text-gray-900 dark:text-white mb-2">{fixture.awayTeam}</div>
                  <div className="text-3xl font-extrabold text-purple-600 dark:text-purple-400">
                    {fixture.predictedScore?.away ?? 0}
                  </div>
                </div>
              </div>

              {/* Win Probabilities */}
              <div className="grid grid-cols-3 gap-6">
                <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-xl">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {(fixture.homeWinProb || 0).toFixed(0)}%
                  </div>
                  <div className="text-sm font-medium text-blue-700 dark:text-blue-300">Home Win</div>
                </div>
                <div className="text-center p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800/20 dark:to-gray-700/20 rounded-xl">
                  <div className="text-2xl font-bold text-gray-600 dark:text-gray-400">
                    {(fixture.drawProb || 0).toFixed(0)}%
                  </div>
                  <div className="text-sm font-medium text-gray-700 dark:text-gray-300">Draw</div>
                </div>
                <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-xl">
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {(fixture.awayWinProb || 0).toFixed(0)}%
                  </div>
                  <div className="text-sm font-medium text-purple-700 dark:text-purple-300">Away Win</div>
                </div>
              </div>

              {/* Confidence Badge */}
              <div className="text-center mt-6">
                <span className="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-gradient-to-r from-green-100 to-emerald-100 text-green-800 dark:from-green-900/30 dark:to-emerald-900/30 dark:text-green-400">
                  ⚡ {(fixture.confidence || 0).toFixed(0)}% Confidence
                </span>
              </div>

              {/* Venue - Simplified */}
              <div className="text-center mt-4 text-sm text-gray-500 dark:text-gray-400">
                📍 {fixture.venue}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default FixtureList;
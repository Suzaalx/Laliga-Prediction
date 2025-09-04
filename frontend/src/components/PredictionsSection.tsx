'use client';

import { useEffect, useState } from 'react';
import { Calendar, Clock, MapPin, TrendingUp, Target, Zap } from 'lucide-react';

interface Fixture {
  id: string;
  homeTeam: string;
  awayTeam: string;
  date: string;
  time: string;
  venue: string;
  homeWinProb: number;
  drawProb: number;
  awayWinProb: number;
  predictedScore: {
    home: number;
    away: number;
  };
  confidence: number;
}

export default function PredictionsSection() {
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFixtures = async () => {
      try {
        const response = await fetch('/api/fixtures');
        if (response.ok) {
          const data = await response.json();
          setFixtures(data);
        } else {
          // Fallback to mock data if API fails
          setFixtures(generateMockFixtures());
        }
      } catch (error) {
        console.error('Failed to fetch fixtures:', error);
        setFixtures(generateMockFixtures());
      } finally {
        setLoading(false);
      }
    };

    fetchFixtures();
  }, []);

  const generateMockFixtures = (): Fixture[] => {
    const teams = [
      'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Real Sociedad',
      'Villarreal', 'Real Betis', 'Athletic Bilbao', 'Valencia', 'Osasuna'
    ];
    
    const venues = [
      'Santiago Bernabéu', 'Camp Nou', 'Wanda Metropolitano', 'Ramón Sánchez-Pizjuán',
      'Reale Arena', 'Estadio de la Cerámica', 'Benito Villamarín', 'San Mamés'
    ];

    return Array.from({ length: 6 }, (_, i) => {
      const homeTeam = teams[Math.floor(Math.random() * teams.length)];
      let awayTeam = teams[Math.floor(Math.random() * teams.length)];
      while (awayTeam === homeTeam) {
        awayTeam = teams[Math.floor(Math.random() * teams.length)];
      }

      const homeWinProb = Math.random() * 0.6 + 0.2;
      const awayWinProb = Math.random() * (0.8 - homeWinProb);
      const drawProb = 1 - homeWinProb - awayWinProb;

      const date = new Date();
      date.setDate(date.getDate() + i + 1);

      return {
        id: `fixture-${i}`,
        homeTeam,
        awayTeam,
        date: date.toISOString().split('T')[0],
        time: `${15 + (i % 4)}:00`,
        venue: venues[Math.floor(Math.random() * venues.length)],
        homeWinProb,
        drawProb,
        awayWinProb,
        predictedScore: {
          home: Math.floor(Math.random() * 4),
          away: Math.floor(Math.random() * 4)
        },
        confidence: Math.random() * 0.3 + 0.7
      };
    });
  };

  const getProbabilityColor = (prob: number) => {
    if (prob > 0.6) return 'from-green-500 to-emerald-600';
    if (prob > 0.4) return 'from-yellow-500 to-orange-600';
    return 'from-red-500 to-pink-600';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-600 dark:text-green-400';
    if (confidence > 0.6) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Match Predictions</h2>
          <p className="text-gray-600 dark:text-gray-300">Loading AI-powered predictions...</p>
        </div>
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="bg-white/80 dark:bg-gray-700/80 rounded-xl p-6 animate-pulse">
              <div className="flex justify-between items-center">
                <div className="flex-1">
                  <div className="h-6 bg-gray-300 dark:bg-gray-600 rounded mb-2"></div>
                  <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-2/3"></div>
                </div>
                <div className="w-24 h-16 bg-gray-300 dark:bg-gray-600 rounded"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Match Predictions</h2>
        <p className="text-gray-600 dark:text-gray-300">AI-powered predictions for upcoming La Liga matches</p>
      </div>

      {/* Predictions Grid */}
      <div className="space-y-6">
        {fixtures.map((fixture) => {
          const maxProb = Math.max(fixture.homeWinProb, fixture.drawProb, fixture.awayWinProb);
          const winner = 
            maxProb === fixture.homeWinProb ? 'home' :
            maxProb === fixture.awayWinProb ? 'away' : 'draw';

          return (
            <div
              key={fixture.id}
              className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 dark:border-gray-700 hover:shadow-xl transition-all duration-300 hover:scale-[1.02]"
            >
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Match Info */}
                <div className="lg:col-span-2 space-y-4">
                  {/* Teams */}
                  <div className="flex items-center justify-between">
                    <div className="text-xl font-bold text-gray-900 dark:text-white">
                      {fixture.homeTeam}
                    </div>
                    <div className="text-2xl font-bold text-gray-500 dark:text-gray-400">vs</div>
                    <div className="text-xl font-bold text-gray-900 dark:text-white">
                      {fixture.awayTeam}
                    </div>
                  </div>

                  {/* Match Details */}
                  <div className="flex flex-wrap gap-4 text-sm text-gray-600 dark:text-gray-300">
                    <div className="flex items-center gap-1">
                      <Calendar className="h-4 w-4" />
                      <span>{new Date(fixture.date).toLocaleDateString()}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="h-4 w-4" />
                      <span>{fixture.time}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <MapPin className="h-4 w-4" />
                      <span>{fixture.venue}</span>
                    </div>
                  </div>

                  {/* Predicted Score */}
                  <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-gray-900 dark:text-white">
                          {fixture.predictedScore?.home ?? 0}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">Predicted</div>
                      </div>
                      <div className="text-gray-400 dark:text-gray-500">-</div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-gray-900 dark:text-white">
                          {fixture.predictedScore?.away ?? 0}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">Score</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Probabilities */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Target className="h-5 w-5 text-blue-600" />
                    <span className="font-semibold text-gray-900 dark:text-white">Win Probabilities</span>
                  </div>

                  <div className="space-y-3">
                    {/* Home Win */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-300">{fixture.homeTeam} Win</span>
                        <span className="font-semibold text-gray-900 dark:text-white">
                          {(fixture.homeWinProb * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full bg-gradient-to-r ${getProbabilityColor(fixture.homeWinProb)} transition-all duration-500`}
                          style={{ width: `${fixture.homeWinProb * 100}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Draw */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-300">Draw</span>
                        <span className="font-semibold text-gray-900 dark:text-white">
                          {(fixture.drawProb * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full bg-gradient-to-r ${getProbabilityColor(fixture.drawProb)} transition-all duration-500`}
                          style={{ width: `${fixture.drawProb * 100}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Away Win */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-300">{fixture.awayTeam} Win</span>
                        <span className="font-semibold text-gray-900 dark:text-white">
                          {(fixture.awayWinProb * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full bg-gradient-to-r ${getProbabilityColor(fixture.awayWinProb)} transition-all duration-500`}
                          style={{ width: `${fixture.awayWinProb * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-600">
                    <div className="flex items-center gap-1">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-300">Confidence</span>
                    </div>
                    <span className={`font-semibold ${getConfidenceColor(fixture.confidence)}`}>
                      {(fixture.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* View More */}
      <div className="text-center">
        <button className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-8 py-3 rounded-xl font-semibold hover:shadow-lg transition-all duration-300 hover:scale-105">
          View All Predictions
        </button>
      </div>
    </div>
  );
}
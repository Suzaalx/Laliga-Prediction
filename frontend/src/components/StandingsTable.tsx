'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Loader2, Trophy, TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface TeamStanding {
  position: number;
  team: string;
  matches_played: number;
  wins: number;
  draws: number;
  losses: number;
  goals_for: number;
  goals_against: number;
  goal_difference: number;
  points: number;
  form: string[];
  last_5_games: string;
}

const StandingsTable: React.FC = () => {
  const [standings, setStandings] = useState<TeamStanding[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSeason, setSelectedSeason] = useState('2024-25');

  useEffect(() => {
    fetchStandings();
  }, [selectedSeason]);

  const fetchStandings = async () => {
    try {
      setLoading(true);
      
      // For now, we'll use mock data since we don't have a standings API
      // In a real app, this would fetch from /api/standings
      const mockStandings: TeamStanding[] = [
        {
          position: 1,
          team: 'Real Madrid',
          matches_played: 20,
          wins: 15,
          draws: 3,
          losses: 2,
          goals_for: 45,
          goals_against: 18,
          goal_difference: 27,
          points: 48,
          form: ['W', 'W', 'D', 'W', 'W'],
          last_5_games: 'WWDWW'
        },
        {
          position: 2,
          team: 'Barcelona',
          matches_played: 20,
          wins: 14,
          draws: 4,
          losses: 2,
          goals_for: 42,
          goals_against: 20,
          goal_difference: 22,
          points: 46,
          form: ['W', 'D', 'W', 'W', 'L'],
          last_5_games: 'WDWWL'
        },
        {
          position: 3,
          team: 'Atletico Madrid',
          matches_played: 20,
          wins: 12,
          draws: 5,
          losses: 3,
          goals_for: 35,
          goals_against: 22,
          goal_difference: 13,
          points: 41,
          form: ['D', 'W', 'W', 'D', 'W'],
          last_5_games: 'DWWDW'
        },
        {
          position: 4,
          team: 'Athletic Bilbao',
          matches_played: 20,
          wins: 11,
          draws: 6,
          losses: 3,
          goals_for: 32,
          goals_against: 25,
          goal_difference: 7,
          points: 39,
          form: ['W', 'D', 'L', 'W', 'D'],
          last_5_games: 'WDLWD'
        },
        {
          position: 5,
          team: 'Real Sociedad',
          matches_played: 20,
          wins: 10,
          draws: 7,
          losses: 3,
          goals_for: 30,
          goals_against: 23,
          goal_difference: 7,
          points: 37,
          form: ['D', 'W', 'D', 'W', 'D'],
          last_5_games: 'DWDWD'
        },
        {
          position: 6,
          team: 'Villarreal',
          matches_played: 20,
          wins: 9,
          draws: 8,
          losses: 3,
          goals_for: 28,
          goals_against: 21,
          goal_difference: 7,
          points: 35,
          form: ['D', 'D', 'W', 'L', 'W'],
          last_5_games: 'DDWLW'
        },
        {
          position: 7,
          team: 'Real Betis',
          matches_played: 20,
          wins: 8,
          draws: 8,
          losses: 4,
          goals_for: 26,
          goals_against: 24,
          goal_difference: 2,
          points: 32,
          form: ['L', 'D', 'W', 'D', 'W'],
          last_5_games: 'LDWDW'
        },
        {
          position: 8,
          team: 'Valencia',
          matches_played: 20,
          wins: 7,
          draws: 9,
          losses: 4,
          goals_for: 25,
          goals_against: 23,
          goal_difference: 2,
          points: 30,
          form: ['D', 'L', 'D', 'W', 'D'],
          last_5_games: 'DLDWD'
        }
      ];
      
      setStandings(mockStandings);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load standings');
    } finally {
      setLoading(false);
    }
  };

  const getPositionColor = (position: number) => {
    if (position <= 4) return 'text-green-600 dark:text-green-400'; // Champions League
    if (position <= 6) return 'text-blue-600 dark:text-blue-400'; // Europa League
    if (position <= 7) return 'text-orange-600 dark:text-orange-400'; // Conference League
    if (position >= 18) return 'text-red-600 dark:text-red-400'; // Relegation
    return 'text-gray-600 dark:text-gray-400';
  };

  const getFormIcon = (result: string) => {
    switch (result) {
      case 'W':
        return <TrendingUp className="h-3 w-3 text-green-600" />;
      case 'L':
        return <TrendingDown className="h-3 w-3 text-red-600" />;
      case 'D':
        return <Minus className="h-3 w-3 text-yellow-600" />;
      default:
        return null;
    }
  };

  const getFormBadgeColor = (result: string) => {
    switch (result) {
      case 'W':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
      case 'L':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
      case 'D':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300';
    }
  };

  if (loading) {
    return (
      <Card className="w-full fade-in">
        <CardContent className="flex items-center justify-center p-8">
          <Loader2 className="h-8 w-8 animate-spin mr-2 shadow-glow" />
          <span className="gradient-text">Loading standings...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full glass border-red-200 dark:border-red-800 shadow-glow-red slide-up">
        <CardContent className="p-6">
          <div className="text-center text-red-600 dark:text-red-400">
            <p className="font-semibold gradient-text-danger">Error loading standings</p>
            <p className="text-sm mt-1">{error}</p>
            <button 
              onClick={fetchStandings} 
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
      {/* Header */}
      <Card className="glass card-hover">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 gradient-text">
            <Trophy className="h-5 w-5" />
            La Liga Standings - Season {selectedSeason}
          </CardTitle>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300">
              1-4: Champions League
            </Badge>
            <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300">
              5-6: Europa League
            </Badge>
            <Badge className="bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300">
              7: Conference League
            </Badge>
            <Badge className="bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300">
              18-20: Relegation
            </Badge>
          </div>
        </CardHeader>
      </Card>

      {/* Standings Table */}
      <Card className="glass card-hover">
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gradient-to-r from-blue-600 to-purple-600">
                <tr className="text-left text-xs font-medium text-white uppercase tracking-wider">
                  <th className="px-4 py-3">Pos</th>
                  <th className="px-4 py-3">Team</th>
                  <th className="px-4 py-3 text-center">MP</th>
                  <th className="px-4 py-3 text-center">W</th>
                  <th className="px-4 py-3 text-center">D</th>
                  <th className="px-4 py-3 text-center">L</th>
                  <th className="px-4 py-3 text-center">GF</th>
                  <th className="px-4 py-3 text-center">GA</th>
                  <th className="px-4 py-3 text-center">GD</th>
                  <th className="px-4 py-3 text-center">Pts</th>
                  <th className="px-4 py-3 text-center">Form</th>
                </tr>
              </thead>
              <tbody className="bg-white/50 dark:bg-gray-900/50 divide-y divide-gray-200 dark:divide-gray-700">
                {standings.map((team) => (
                  <tr key={team.position} className="hover:bg-gray-50/80 dark:hover:bg-gray-800/80 transition-all duration-200 card-hover">
                    <td className="px-4 py-4">
                      <span className={`font-bold shadow-lg ${getPositionColor(team.position)}`}>
                        {team.position}
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      <div className="font-medium gradient-text">
                        {team.team}
                      </div>
                    </td>
                    <td className="px-4 py-4 text-center text-sm text-gray-600 dark:text-gray-400 font-medium">
                      {team.matches_played}
                    </td>
                    <td className="px-4 py-4 text-center text-sm font-semibold gradient-text-success">
                      {team.wins}
                    </td>
                    <td className="px-4 py-4 text-center text-sm font-semibold gradient-text-warning">
                      {team.draws}
                    </td>
                    <td className="px-4 py-4 text-center text-sm font-semibold gradient-text-danger">
                      {team.losses}
                    </td>
                    <td className="px-4 py-4 text-center text-sm text-gray-600 dark:text-gray-400 font-medium">
                      {team.goals_for}
                    </td>
                    <td className="px-4 py-4 text-center text-sm text-gray-600 dark:text-gray-400 font-medium">
                      {team.goals_against}
                    </td>
                    <td className="px-4 py-4 text-center text-sm">
                      <span className={team.goal_difference >= 0 ? 'gradient-text-success font-bold' : 'gradient-text-danger font-bold'}>
                        {team.goal_difference > 0 ? '+' : ''}{team.goal_difference}
                      </span>
                    </td>
                    <td className="px-4 py-4 text-center">
                      <span className="font-bold text-lg gradient-text">
                        {team.points}
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      <div className="flex gap-1 justify-center">
                        {team.form.map((result, index) => (
                          <Badge
                            key={index}
                            className={`${getFormBadgeColor(result)} text-xs w-6 h-6 flex items-center justify-center p-0 shadow-lg transition-transform hover:scale-110`}
                          >
                            {result}
                          </Badge>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default StandingsTable;
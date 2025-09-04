'use client';

import { useEffect, useState } from 'react';
import { Trophy, TrendingUp, TrendingDown, Minus, Users, Target, Shield } from 'lucide-react';
import Link from 'next/link';

interface TeamRanking {
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
  recent_performance: 'up' | 'down' | 'stable';
}

export default function TeamsSection() {
  const [teams, setTeams] = useState<TeamRanking[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTeams = async () => {
      try {
        // Try to fetch from API first
        const response = await fetch('/api/teams/rankings');
        if (response.ok) {
          const data = await response.json();
          setTeams(data);
        } else {
          // Fallback to mock data
          setTeams(generateMockTeams());
        }
      } catch (error) {
        console.error('Failed to fetch teams:', error);
        setTeams(generateMockTeams());
      } finally {
        setLoading(false);
      }
    };

    fetchTeams();
  }, []);

  const generateMockTeams = (): TeamRanking[] => {
    const teamNames = [
      'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Athletic Bilbao', 'Real Sociedad',
      'Villarreal', 'Real Betis', 'Valencia', 'Sevilla', 'Osasuna',
      'Getafe', 'Girona', 'Celta Vigo', 'Las Palmas', 'Rayo Vallecano',
      'Mallorca', 'Alaves', 'Espanyol', 'Leganes', 'Valladolid'
    ];

    return teamNames.map((team, index) => {
      const matches = 15 + Math.floor(Math.random() * 5);
      const wins = Math.floor(Math.random() * (matches - index * 0.5));
      const losses = Math.floor(Math.random() * (matches - wins));
      const draws = matches - wins - losses;
      const goals_for = wins * 2 + draws + Math.floor(Math.random() * 10);
      const goals_against = losses * 2 + Math.floor(Math.random() * 10);
      
      const formResults = ['W', 'D', 'L'];
      const form = Array.from({ length: 5 }, () => 
        formResults[Math.floor(Math.random() * formResults.length)]
      );

      const performances: ('up' | 'down' | 'stable')[] = ['up', 'down', 'stable'];
      const recent_performance = performances[Math.floor(Math.random() * performances.length)];

      return {
        position: index + 1,
        team,
        matches_played: matches,
        wins,
        draws,
        losses,
        goals_for,
        goals_against,
        goal_difference: goals_for - goals_against,
        points: wins * 3 + draws,
        form,
        recent_performance
      };
    }).sort((a, b) => b.points - a.points).map((team, index) => ({ ...team, position: index + 1 }));
  };

  const getPositionColor = (position: number) => {
    if (position <= 4) return 'from-green-500 to-emerald-600'; // Champions League
    if (position <= 6) return 'from-blue-500 to-blue-600'; // Europa League
    if (position <= 7) return 'from-purple-500 to-purple-600'; // Conference League
    if (position >= 18) return 'from-red-500 to-red-600'; // Relegation
    return 'from-gray-500 to-gray-600'; // Mid-table
  };

  const getPositionBadge = (position: number) => {
    if (position <= 4) return { text: 'UCL', color: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' };
    if (position <= 6) return { text: 'UEL', color: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' };
    if (position <= 7) return { text: 'UECL', color: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200' };
    if (position >= 18) return { text: 'REL', color: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' };
    return null;
  };

  const getFormColor = (result: string) => {
    switch (result) {
      case 'W': return 'bg-green-500';
      case 'D': return 'bg-yellow-500';
      case 'L': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getPerformanceIcon = (performance: string) => {
    switch (performance) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'down': return <TrendingDown className="h-4 w-4 text-red-500" />;
      default: return <Minus className="h-4 w-4 text-gray-500" />;
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Team Rankings</h2>
          <p className="text-gray-600 dark:text-gray-300">Loading current La Liga standings...</p>
        </div>
        <div className="space-y-3">
          {[...Array(10)].map((_, i) => (
            <div key={i} className="bg-white/80 dark:bg-gray-700/80 rounded-xl p-4 animate-pulse">
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded"></div>
                <div className="flex-1">
                  <div className="h-5 bg-gray-300 dark:bg-gray-600 rounded mb-2"></div>
                  <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-2/3"></div>
                </div>
                <div className="w-16 h-8 bg-gray-300 dark:bg-gray-600 rounded"></div>
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
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Team Rankings</h2>
        <p className="text-gray-600 dark:text-gray-300">Current La Liga standings and team performance</p>
      </div>

      {/* Top Teams Highlight */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {teams.slice(0, 3).map((team, index) => {
          const medals = ['🥇', '🥈', '🥉'];
          const colors = [
            'from-yellow-400 to-yellow-600',
            'from-gray-400 to-gray-600', 
            'from-orange-400 to-orange-600'
          ];
          
          return (
            <Link key={team.team} href={`/teams/${encodeURIComponent(team.team)}`}>
              <div className={`bg-gradient-to-r ${colors[index]} p-6 rounded-2xl text-white hover:shadow-xl transition-all duration-300 hover:scale-105 cursor-pointer`}>
                <div className="flex items-center justify-between mb-4">
                  <div className="text-3xl">{medals[index]}</div>
                  <div className="text-right">
                    <div className="text-2xl font-bold">{team.points}</div>
                    <div className="text-sm opacity-90">points</div>
                  </div>
                </div>
                <div className="space-y-2">
                  <h3 className="text-xl font-bold">{team.team}</h3>
                  <div className="flex justify-between text-sm opacity-90">
                    <span>{team.wins}W {team.draws}D {team.losses}L</span>
                    <span>GD: {team.goal_difference > 0 ? '+' : ''}{team.goal_difference}</span>
                  </div>
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      {/* Full Table */}
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <Trophy className="h-5 w-5 text-yellow-500" />
            Complete Standings
          </h3>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr className="text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                <th className="px-6 py-3">Position</th>
                <th className="px-6 py-3">Team</th>
                <th className="px-6 py-3 text-center">MP</th>
                <th className="px-6 py-3 text-center">W</th>
                <th className="px-6 py-3 text-center">D</th>
                <th className="px-6 py-3 text-center">L</th>
                <th className="px-6 py-3 text-center">GF</th>
                <th className="px-6 py-3 text-center">GA</th>
                <th className="px-6 py-3 text-center">GD</th>
                <th className="px-6 py-3 text-center">Pts</th>
                <th className="px-6 py-3 text-center">Form</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-600">
              {teams.map((team) => {
                const badge = getPositionBadge(team.position);
                return (
                  <tr key={team.team} className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-lg bg-gradient-to-r ${getPositionColor(team.position)} flex items-center justify-center text-white font-bold text-sm`}>
                          {team.position}
                        </div>
                        {badge && (
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${badge.color}`}>
                            {badge.text}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <Link href={`/teams/${encodeURIComponent(team.team)}`}>
                        <div className="flex items-center gap-3 hover:text-blue-600 dark:hover:text-blue-400 transition-colors cursor-pointer">
                          <div className="font-medium text-gray-900 dark:text-white">{team.team}</div>
                          {getPerformanceIcon(team.recent_performance)}
                        </div>
                      </Link>
                    </td>
                    <td className="px-6 py-4 text-center text-sm text-gray-500 dark:text-gray-300">{team.matches_played}</td>
                    <td className="px-6 py-4 text-center text-sm font-medium text-green-600 dark:text-green-400">{team.wins}</td>
                    <td className="px-6 py-4 text-center text-sm font-medium text-yellow-600 dark:text-yellow-400">{team.draws}</td>
                    <td className="px-6 py-4 text-center text-sm font-medium text-red-600 dark:text-red-400">{team.losses}</td>
                    <td className="px-6 py-4 text-center text-sm text-gray-900 dark:text-white">{team.goals_for}</td>
                    <td className="px-6 py-4 text-center text-sm text-gray-900 dark:text-white">{team.goals_against}</td>
                    <td className={`px-6 py-4 text-center text-sm font-medium ${
                      team.goal_difference > 0 ? 'text-green-600 dark:text-green-400' :
                      team.goal_difference < 0 ? 'text-red-600 dark:text-red-400' :
                      'text-gray-500 dark:text-gray-300'
                    }`}>
                      {team.goal_difference > 0 ? '+' : ''}{team.goal_difference}
                    </td>
                    <td className="px-6 py-4 text-center text-sm font-bold text-gray-900 dark:text-white">{team.points}</td>
                    <td className="px-6 py-4">
                      <div className="flex justify-center gap-1">
                        {team.form.map((result, index) => (
                          <div
                            key={index}
                            className={`w-6 h-6 rounded-full ${getFormColor(result)} flex items-center justify-center text-white text-xs font-bold`}
                          >
                            {result}
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Legend */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-4">
        <h4 className="font-semibold text-gray-900 dark:text-white mb-3">Competition Qualification</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-gradient-to-r from-green-500 to-emerald-600"></div>
            <span className="text-gray-600 dark:text-gray-300">Champions League (1-4)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-gradient-to-r from-blue-500 to-blue-600"></div>
            <span className="text-gray-600 dark:text-gray-300">Europa League (5-6)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-gradient-to-r from-purple-500 to-purple-600"></div>
            <span className="text-gray-600 dark:text-gray-300">Conference League (7)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-gradient-to-r from-red-500 to-red-600"></div>
            <span className="text-gray-600 dark:text-gray-300">Relegation (18-20)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
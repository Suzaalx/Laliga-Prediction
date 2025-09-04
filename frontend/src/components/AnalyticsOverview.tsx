'use client';

import { useEffect, useState } from 'react';
import { TrendingUp, Users, Target, BarChart3, Activity, Trophy } from 'lucide-react';

interface LeagueAnalytics {
  total_matches: number;
  total_goals: number;
  avg_goals_per_match: number;
  home_win_percentage: number;
  away_win_percentage: number;
  draw_percentage: number;
  top_scorer: {
    player: string;
    team: string;
    goals: number;
  };
  most_goals_match: {
    home_team: string;
    away_team: string;
    total_goals: number;
    date: string;
  };
}

export default function AnalyticsOverview() {
  const [analytics, setAnalytics] = useState<LeagueAnalytics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await fetch('/api/analytics');
        if (response.ok) {
          const data = await response.json();
          setAnalytics(data);
        }
      } catch (error) {
        console.error('Failed to fetch analytics:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">League Analytics</h2>
          <p className="text-gray-600 dark:text-gray-300">Loading comprehensive league insights...</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="bg-white/80 dark:bg-gray-700/80 rounded-xl p-6 animate-pulse">
              <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded mb-4"></div>
              <div className="h-8 bg-gray-300 dark:bg-gray-600 rounded mb-2"></div>
              <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-2/3"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const stats = [
    {
      icon: Activity,
      label: 'Total Matches',
      value: analytics?.total_matches || 0,
      color: 'from-blue-500 to-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-900/20'
    },
    {
      icon: Target,
      label: 'Total Goals',
      value: analytics?.total_goals || 0,
      color: 'from-green-500 to-green-600',
      bgColor: 'bg-green-50 dark:bg-green-900/20'
    },
    {
      icon: BarChart3,
      label: 'Avg Goals/Match',
      value: analytics?.avg_goals_per_match?.toFixed(2) || '0.00',
      color: 'from-purple-500 to-purple-600',
      bgColor: 'bg-purple-50 dark:bg-purple-900/20'
    },
    {
      icon: TrendingUp,
      label: 'Home Win %',
      value: `${analytics?.home_win_percentage?.toFixed(1) || '0.0'}%`,
      color: 'from-orange-500 to-orange-600',
      bgColor: 'bg-orange-50 dark:bg-orange-900/20'
    },
    {
      icon: Users,
      label: 'Away Win %',
      value: `${analytics?.away_win_percentage?.toFixed(1) || '0.0'}%`,
      color: 'from-red-500 to-red-600',
      bgColor: 'bg-red-50 dark:bg-red-900/20'
    },
    {
      icon: Trophy,
      label: 'Draw %',
      value: `${analytics?.draw_percentage?.toFixed(1) || '0.0'}%`,
      color: 'from-indigo-500 to-indigo-600',
      bgColor: 'bg-indigo-50 dark:bg-indigo-900/20'
    }
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">League Analytics</h2>
        <p className="text-gray-600 dark:text-gray-300">Comprehensive insights from the current La Liga season</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div
              key={index}
              className={`${stat.bgColor} rounded-xl p-6 border border-gray-200 dark:border-gray-600 hover:shadow-lg transition-all duration-300 hover:scale-105`}
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`p-3 rounded-lg bg-gradient-to-r ${stat.color}`}>
                  <Icon className="h-6 w-6 text-white" />
                </div>
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-300">{stat.label}</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{stat.value}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Highlights */}
      {analytics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Scorer */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6 border border-yellow-200 dark:border-yellow-700">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-lg">
                <Trophy className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Top Scorer</h3>
            </div>
            <div className="space-y-2">
              <p className="text-xl font-bold text-gray-900 dark:text-white">
                {analytics.top_scorer?.player || 'N/A'}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                {analytics.top_scorer?.team || 'N/A'} • {analytics.top_scorer?.goals || 0} goals
              </p>
            </div>
          </div>

          {/* Highest Scoring Match */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 border border-green-200 dark:border-green-700">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg">
                <Target className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Highest Scoring Match</h3>
            </div>
            <div className="space-y-2">
              <p className="text-lg font-bold text-gray-900 dark:text-white">
                {analytics.most_goals_match?.home_team || 'N/A'} vs {analytics.most_goals_match?.away_team || 'N/A'}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                {analytics.most_goals_match?.total_goals || 0} goals • {analytics.most_goals_match?.date || 'N/A'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ArrowLeft, TrendingUp, Target, BarChart3, Calendar, Trophy, Home, Plane } from 'lucide-react';

interface TeamAnalytics {
  team_name: string;
  form_metrics: {
    goals_for_avg_5: number;
    goals_against_avg_5: number;
    goals_for_avg_10: number;
    goals_against_avg_10: number;
    home_ratio_5: number;
    home_ratio_10: number;
    total_records: number;
    data_range: {
      from: string;
      to: string;
    };
  };
  match_statistics: {
    total_matches: number;
    wins: number;
    draws: number;
    losses: number;
    win_percentage: number;
    goals_for: number;
    goals_against: number;
    goal_difference: number;
    goals_per_match: number;
    home_record: {
      matches: number;
      wins: number;
      draws: number;
      losses: number;
      goals_for: number;
      goals_against: number;
    };
    away_record: {
      matches: number;
      wins: number;
      draws: number;
      losses: number;
      goals_for: number;
      goals_against: number;
    };
  };
  performance_trends: {
    home_trends: {
      recent_form_5_games?: {
        avg_goals_for: number;
        avg_goals_against: number;
      };
      recent_form_10_games?: {
        avg_goals_for: number;
        avg_goals_against: number;
      };
    };
    away_trends: {
      recent_form_5_games?: {
        avg_goals_for: number;
        avg_goals_against: number;
      };
      recent_form_10_games?: {
        avg_goals_for: number;
        avg_goals_against: number;
      };
    };
  };
  recent_form: Array<{
    date: string;
    goals_for_r5: number;
    goals_against_r5: number;
    home_ratio_r5: number;
  }>;
  head_to_head: {
    total_opponents: number;
    most_played_against: {
      opponent: string;
      matches: number;
    };
  };
}

interface ApiResponse {
  success: boolean;
  team_name: string;
  analytics: TeamAnalytics;
}

export default function TeamDetailPage() {
  const params = useParams();
  const router = useRouter();
  const teamName = params.teamName as string;
  const [analytics, setAnalytics] = useState<TeamAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTeamAnalytics = async () => {
      if (!teamName) return;
      
      try {
        setLoading(true);
        const response = await fetch(`/api/team-analytics/teams/${encodeURIComponent(teamName)}/analytics`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch analytics: ${response.status}`);
        }
        
        const data: ApiResponse = await response.json();
        
        if (data.success && data.analytics) {
          setAnalytics(data.analytics);
        } else {
          throw new Error('Invalid response format');
        }
      } catch (err) {
        console.error('Error fetching team analytics:', err);
        setError(err instanceof Error ? err.message : 'Failed to load team analytics');
      } finally {
        setLoading(false);
      }
    };

    fetchTeamAnalytics();
  }, [teamName]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading team analytics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <p className="font-bold">Error Loading Analytics</p>
            <p>{error}</p>
          </div>
          <button
            onClick={() => router.push('/teams')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Back to Teams
          </button>
        </div>
      </div>
    );
  }

  if (!analytics) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 mb-4">No analytics data available for this team.</p>
          <button
            onClick={() => router.push('/teams')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Back to Teams
          </button>
        </div>
      </div>
    );
  }

  // Safely destructure with fallback values
  const form_metrics = analytics?.form_metrics;
  const match_statistics = analytics?.match_statistics;
  const recent_form = analytics?.recent_form || [];
  const head_to_head = analytics?.head_to_head;

  // Provide default values if form_metrics is undefined
  const safeFormMetrics = form_metrics || {
    goals_for_avg_5: 0,
    goals_against_avg_5: 0,
    goals_for_avg_10: 0,
    goals_against_avg_10: 0,
    home_ratio_5: 0,
    home_ratio_10: 0,
    total_records: 0,
    data_range: { from: 'N/A', to: 'N/A' }
  };

  // Provide default values for match_statistics if undefined
  const safeMatchStats = match_statistics || {
    total_matches: 0,
    wins: 0,
    draws: 0,
    losses: 0,
    win_percentage: 0,
    goals_for: 0,
    goals_against: 0,
    goal_difference: 0,
    goals_per_match: 0,
    home_record: { matches: 0, wins: 0, draws: 0, losses: 0, goals_for: 0, goals_against: 0 },
    away_record: { matches: 0, wins: 0, draws: 0, losses: 0, goals_for: 0, goals_against: 0 }
  };

  // Provide default values for head_to_head if undefined
  const safeHeadToHead = head_to_head || {
    total_opponents: 0,
    most_played_against: { opponent: 'N/A', matches: 0 }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => router.push('/teams')}
                className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
              >
                <ArrowLeft className="h-5 w-5 mr-2" />
                Back to Teams
              </button>
              <div className="h-6 border-l border-gray-300"></div>
              <h1 className="text-2xl font-bold text-gray-900">
                {teamName} Analytics
              </h1>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Matches</p>
                <p className="text-2xl font-bold text-gray-900">{safeMatchStats.total_matches}</p>
              </div>
              <BarChart3 className="h-8 w-8 text-blue-600" />
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Win Rate</p>
                <p className="text-2xl font-bold text-green-600">{safeMatchStats.win_percentage}%</p>
              </div>
              <Trophy className="h-8 w-8 text-green-600" />
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Goals Per Match</p>
                <p className="text-2xl font-bold text-orange-600">{safeMatchStats.goals_per_match}</p>
              </div>
              <Target className="h-8 w-8 text-orange-600" />
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Goal Difference</p>
                <p className={`text-2xl font-bold ${
                    safeMatchStats.goal_difference >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {safeMatchStats.goal_difference >= 0 ? '+' : ''}{safeMatchStats.goal_difference}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-purple-600" />
            </div>
          </div>
        </div>

        {/* Detailed Statistics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Match Record */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Match Record</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Wins</span>
                <span className="font-semibold text-green-600">{safeMatchStats.wins}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Draws</span>
                <span className="font-semibold text-yellow-600">{safeMatchStats.draws}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Losses</span>
                <span className="font-semibold text-red-600">{safeMatchStats.losses}</span>
              </div>
              <div className="border-t pt-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Goals For</span>
                  <span className="font-semibold text-blue-600">{safeMatchStats.goals_for}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Goals Against</span>
                  <span className="font-semibold text-red-600">{safeMatchStats.goals_against}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Form Metrics */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Form</h3>
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-600 mb-2">Last 5 Games Average</p>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Goals For</span>
                  <span className="font-semibold text-green-600">{(safeFormMetrics?.goals_for_avg_5 ?? 0).toFixed(1)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Goals Against</span>
                  <span className="font-semibold text-red-600">{(safeFormMetrics?.goals_against_avg_5 ?? 0).toFixed(1)}</span>
                </div>
              </div>
              <div className="border-t pt-4">
                <p className="text-sm text-gray-600 mb-2">Last 10 Games Average</p>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Goals For</span>
                  <span className="font-semibold text-green-600">{(safeFormMetrics?.goals_for_avg_10 ?? 0).toFixed(1)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Goals Against</span>
                  <span className="font-semibold text-red-600">{(safeFormMetrics?.goals_against_avg_10 ?? 0).toFixed(1)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Home vs Away Performance */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Home Record */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center mb-4">
              <Home className="h-5 w-5 text-blue-600 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">Home Record</h3>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Matches</span>
                <span className="font-semibold">{safeMatchStats.home_record.matches}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">W-D-L</span>
                <span className="font-semibold">
                  {safeMatchStats.home_record.wins}-{safeMatchStats.home_record.draws}-{safeMatchStats.home_record.losses}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Goals</span>
                <span className="font-semibold">
                  {safeMatchStats.home_record.goals_for}:{safeMatchStats.home_record.goals_against}
                </span>
              </div>
            </div>
          </div>

          {/* Away Record */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center mb-4">
              <Plane className="h-5 w-5 text-purple-600 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">Away Record</h3>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Matches</span>
                <span className="font-semibold">{safeMatchStats.away_record.matches}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">W-D-L</span>
                <span className="font-semibold">
                  {safeMatchStats.away_record.wins}-{safeMatchStats.away_record.draws}-{safeMatchStats.away_record.losses}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Goals</span>
                <span className="font-semibold">
                  {safeMatchStats.away_record.goals_for}:{safeMatchStats.away_record.goals_against}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Additional Info */}
        {head_to_head && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Head-to-Head Summary</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Total Opponents</span>
                <span className="font-semibold">{safeHeadToHead.total_opponents}</span>
              </div>
              {safeHeadToHead.most_played_against?.opponent && safeHeadToHead.most_played_against.opponent !== 'N/A' && (
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Most Played Against</span>
                    <span className="font-semibold">
                      {safeHeadToHead.most_played_against.opponent} ({safeHeadToHead.most_played_against.matches} matches)
                    </span>
                  </div>
                )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
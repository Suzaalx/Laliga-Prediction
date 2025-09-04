'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, BarChart3, Target, Trophy, Calendar, Users, Activity, AlertTriangle } from 'lucide-react';

interface LeagueAnalytics {
  season_summary: {
    total_matches: number;
    total_goals: number;
    avg_goals_per_match: number;
    total_teams: number;
    current_season: string;
    data_range: {
      from: string;
      to: string;
    };
  };
  team_performance: Array<{
    team: string;
    matches: number;
    wins: number;
    draws: number;
    losses: number;
    goals_for: number;
    goals_against: number;
    goal_difference: number;
    points: number;
    win_percentage: number;
  }>;
  scoring_trends: Array<{
    month: string;
    avg_goals: number;
    total_matches: number;
    home_advantage: number;
  }>;
  referee_stats: Array<{
    referee: string;
    matches: number;
    avg_cards_per_match: number;
    avg_goals_per_match: number;
  }>;
  venue_analysis: Array<{
    venue_type: string;
    matches: number;
    home_wins: number;
    away_wins: number;
    draws: number;
    home_win_percentage: number;
  }>;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<LeagueAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/analytics/league');
      
      if (!response.ok) {
        throw new Error(`Failed to fetch analytics: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success && data.analytics) {
        setAnalytics(data.analytics);
        setError(null);
      } else {
        throw new Error(data.error || 'Failed to load analytics data');
      }
    } catch (err) {
      console.error('Error fetching league analytics:', err);
      setError(err instanceof Error ? err.message : 'Failed to load league analytics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading league analytics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center max-w-md">
          <Alert className="mb-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              <strong>Error Loading Analytics:</strong> {error}
            </AlertDescription>
          </Alert>
          <button
            onClick={fetchAnalytics}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!analytics) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 mb-4">No analytics data available.</p>
          <button
            onClick={fetchAnalytics}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Refresh Data
          </button>
        </div>
      </div>
    );
  }

  const { season_summary, team_performance, scoring_trends, referee_stats, venue_analysis } = analytics;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">League Analytics</h1>
              <p className="text-gray-600 mt-1">
                Comprehensive insights from {season_summary.current_season} season data
              </p>
            </div>
            <Badge variant="outline" className="text-sm">
              Data: {season_summary.data_range.from} to {season_summary.data_range.to}
            </Badge>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Matches</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{season_summary.total_matches}</div>
              <p className="text-xs text-muted-foreground">
                Across {season_summary.total_teams} teams
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Goals</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{season_summary.total_goals}</div>
              <p className="text-xs text-muted-foreground">
                {season_summary.avg_goals_per_match.toFixed(2)} per match
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Teams</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{season_summary.total_teams}</div>
              <p className="text-xs text-muted-foreground">
                La Liga {season_summary.current_season}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Season Progress</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {Math.round((season_summary.total_matches / (season_summary.total_teams * (season_summary.total_teams - 1))) * 100)}%
              </div>
              <p className="text-xs text-muted-foreground">
                Matches completed
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Detailed Analytics */}
        <Tabs defaultValue="performance" className="space-y-4">
          <TabsList>
            <TabsTrigger value="performance">Team Performance</TabsTrigger>
            <TabsTrigger value="trends">Scoring Trends</TabsTrigger>
            <TabsTrigger value="referees">Referee Analysis</TabsTrigger>
            <TabsTrigger value="venues">Venue Statistics</TabsTrigger>
          </TabsList>

          <TabsContent value="performance" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>League Table & Performance</CardTitle>
                <CardDescription>Current standings and team statistics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-2">Team</th>
                        <th className="text-center p-2">Matches</th>
                        <th className="text-center p-2">W</th>
                        <th className="text-center p-2">D</th>
                        <th className="text-center p-2">L</th>
                        <th className="text-center p-2">GF</th>
                        <th className="text-center p-2">GA</th>
                        <th className="text-center p-2">GD</th>
                        <th className="text-center p-2">Points</th>
                        <th className="text-center p-2">Win %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {team_performance
                        .sort((a, b) => b.points - a.points)
                        .slice(0, 10)
                        .map((team, index) => (
                        <tr key={team.team} className="border-b hover:bg-gray-50">
                          <td className="p-2 font-medium">
                            <div className="flex items-center">
                              <span className="w-6 h-6 bg-blue-100 text-blue-800 rounded-full flex items-center justify-center text-xs mr-2">
                                {index + 1}
                              </span>
                              {team.team}
                            </div>
                          </td>
                          <td className="text-center p-2">{team.matches}</td>
                          <td className="text-center p-2 text-green-600">{team.wins}</td>
                          <td className="text-center p-2 text-yellow-600">{team.draws}</td>
                          <td className="text-center p-2 text-red-600">{team.losses}</td>
                          <td className="text-center p-2">{team.goals_for}</td>
                          <td className="text-center p-2">{team.goals_against}</td>
                          <td className="text-center p-2 font-medium">
                            <span className={team.goal_difference >= 0 ? 'text-green-600' : 'text-red-600'}>
                              {team.goal_difference > 0 ? '+' : ''}{team.goal_difference}
                            </span>
                          </td>
                          <td className="text-center p-2 font-bold">{team.points}</td>
                          <td className="text-center p-2">{team.win_percentage.toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="trends" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Scoring Trends Over Time</CardTitle>
                <CardDescription>Goals per match and home advantage trends</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={scoring_trends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="avg_goals" 
                        stroke="#8884d8" 
                        name="Avg Goals per Match"
                        strokeWidth={2}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="home_advantage" 
                        stroke="#82ca9d" 
                        name="Home Advantage %"
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="referees" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Referee Statistics</CardTitle>
                <CardDescription>Match control and consistency metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={referee_stats.slice(0, 10)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="referee" angle={-45} textAnchor="end" height={100} />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="avg_cards_per_match" fill="#ff8042" name="Avg Cards/Match" />
                      <Bar dataKey="avg_goals_per_match" fill="#8884d8" name="Avg Goals/Match" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="venues" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Home vs Away Performance</CardTitle>
                <CardDescription>Venue advantage analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={[
                            { name: 'Home Wins', value: venue_analysis.reduce((sum, v) => sum + v.home_wins, 0) },
                            { name: 'Away Wins', value: venue_analysis.reduce((sum, v) => sum + v.away_wins, 0) },
                            { name: 'Draws', value: venue_analysis.reduce((sum, v) => sum + v.draws, 0) }
                          ]}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent = 0 }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {venue_analysis.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="space-y-4">
                    <h4 className="font-semibold">Home Advantage Statistics</h4>
                    {venue_analysis.map((venue, index) => (
                      <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                        <span className="font-medium">{venue.venue_type}</span>
                        <div className="text-right">
                          <div className="font-bold text-green-600">{venue.home_win_percentage.toFixed(1)}%</div>
                          <div className="text-sm text-gray-600">{venue.matches} matches</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
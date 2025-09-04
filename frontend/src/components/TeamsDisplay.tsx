'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Loader2, Search, MapPin, Calendar, Building, Users, Trophy } from 'lucide-react';

interface TeamData {
  name: string;
  csv_name: string;
  official_name: string;
  founded: number | null;
  stadium: string;
  city: string;
  capacity: number | null;
  nickname: string;
  data_source: 'csv_only' | 'enriched';
  statistics?: {
    total_matches: number;
    home_matches: number;
    away_matches: number;
    data_available: boolean;
  };
}

interface TeamsSummary {
  total_teams: number;
  enriched_teams: number;
  csv_only_teams: number;
  enrichment_coverage: string;
  data_source: string;
  csv_file: string;
}

const TeamsDisplay: React.FC = () => {
  const router = useRouter();
  const [teams, setTeams] = useState<TeamData[]>([]);
  const [summary, setSummary] = useState<TeamsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetchTeamsData();
  }, []);

  const fetchTeamsData = async () => {
    try {
      setLoading(true);
      
      // Fetch teams data and summary in parallel
      const [teamsResponse, summaryResponse] = await Promise.all([
        fetch('/api/csv-teams/teams'),
        fetch('/api/csv-teams/teams-summary')
      ]);

      if (!teamsResponse.ok || !summaryResponse.ok) {
        throw new Error('Failed to fetch teams data');
      }

      const teamsData = await teamsResponse.json();
      const summaryData = await summaryResponse.json();

      setTeams(teamsData);
      setSummary(summaryData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };



  const filteredTeams = teams.filter(team =>
    team.official_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    team.city.toLowerCase().includes(searchTerm.toLowerCase()) ||
    team.nickname.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatCapacity = (capacity: number | null) => {
    if (!capacity) return 'Unknown';
    return capacity.toLocaleString();
  };

  if (loading) {
    return (
      <Card className="w-full glass">
        <CardContent className="flex items-center justify-center p-8">
          <Loader2 className="h-8 w-8 animate-spin shadow-glow" />
          <span className="ml-2 gradient-text">Loading teams data...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full glass border-red-200 dark:border-red-800 shadow-glow-red">
        <CardContent className="p-6">
          <div className="text-center text-red-600">
            <p className="font-semibold gradient-text-danger">Error loading teams data</p>
            <p className="text-sm mt-1">{error}</p>
            <button 
              onClick={fetchTeamsData} 
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 card-hover"
            >
              Try Again
            </button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6 fade-in">
      {/* Summary Card */}
      {summary && (
        <Card className="glass shadow-lg card-hover">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 gradient-text">
              <Trophy className="h-5 w-5" />
              Teams Data Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{summary.total_teams}</div>
                <div className="text-sm text-gray-600">Total Teams</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{summary.enriched_teams}</div>
                <div className="text-sm text-gray-600">Enriched</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">{summary.csv_only_teams}</div>
                <div className="text-sm text-gray-600">CSV Only</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">{summary.enrichment_coverage}</div>
                <div className="text-sm text-gray-600">Coverage</div>
              </div>
            </div>
            <div className="mt-4 text-sm text-gray-600">
              <p>Data Source: <Badge variant="outline">{summary.data_source}</Badge></p>
              <p>CSV File: <code className="text-xs bg-gray-100 px-1 rounded">{summary.csv_file}</code></p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Search and Teams List */}
      <Card className="glass shadow-lg">
        <CardHeader>
          <CardTitle className="gradient-text">La Liga Teams</CardTitle>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search teams by name, city, or nickname..."
              value={searchTerm}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchTerm(e.target.value)}
              className="input-enhanced pl-10"
            />
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredTeams.map((team) => (
              <Card 
                key={team.name} 
                className="glass cursor-pointer card-hover transition-all duration-300"
                onClick={() => router.push(`/teams/${encodeURIComponent(team.name)}`)}
              >
                <CardContent className="p-4">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-semibold text-lg gradient-text hover:text-blue-800 transition-colors">{team.official_name}</h3>
                    <Badge 
                      variant={team.data_source === 'enriched' ? 'default' : 'secondary'}
                      className="text-xs"
                    >
                      {team.data_source === 'enriched' ? (
                        <><span className="status-online mr-1"></span>Enhanced</>
                      ) : (
                        <><span className="status-warning mr-1"></span>Basic</>
                      )}
                    </Badge>
                  </div>
                  
                  <div className="space-y-1 text-sm text-gray-600">
                    <div className="flex items-center gap-1">
                      <MapPin className="h-3 w-3" />
                      <span>{team.city}</span>
                    </div>
                    
                    {team.founded && (
                      <div className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        <span>Founded {team.founded}</span>
                      </div>
                    )}
                    
                    <div className="flex items-center gap-1">
                      <Building className="h-3 w-3" />
                      <span>{team.stadium}</span>
                    </div>
                    
                    {team.capacity && (
                      <div className="flex items-center gap-1">
                        <Users className="h-3 w-3" />
                        <span>{formatCapacity(team.capacity)} capacity</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="mt-2">
                    <Badge variant="outline" className="text-xs">
                      {team.nickname}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          {filteredTeams.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <p>No teams found matching your search.</p>
            </div>
          )}
        </CardContent>
      </Card>


    </div>
  );
};

export default TeamsDisplay;
'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { AlertTriangle, CheckCircle, XCircle, TrendingUp, TrendingDown, Activity } from 'lucide-react';

interface PerformanceMetric {
  id: number;
  model_version: string;
  metric_name: string;
  metric_value: number;
  timestamp: string;
  environment: string;
}

interface FeatureDrift {
  id: number;
  model_version: string;
  feature_name: string;
  drift_score: number;
  p_value: number;
  timestamp: string;
  environment: string;
}

interface Alert {
  id: number;
  alert_type: string;
  severity: string;
  message: string;
  model_version: string;
  timestamp: string;
  resolved: boolean;
}

interface ServiceHealth {
  service_name: string;
  status: string;
  response_time_ms: number;
  error_rate: number;
  cpu_usage?: number;
  memory_usage?: number;
}

interface ChartDataPoint {
  date: string;
  [key: string]: string | number;
}

interface MonitoringData {
  performance_metrics: PerformanceMetric[];
  feature_drift: FeatureDrift[];
  active_alerts: Alert[];
  service_health: ServiceHealth[];
  current_health: ServiceHealth[];
  summary: {
    total_alerts: number;
    critical_alerts: number;
    drift_features: number;
    unhealthy_services: number;
  };
}

const MonitoringDashboard: React.FC = () => {
  const [monitoringData, setMonitoringData] = useState<MonitoringData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);
  const [currentTime, setCurrentTime] = useState<string>('');

  const fetchMonitoringData = async () => {
    try {
      const response = await fetch('/api/monitoring/dashboard?days=7');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setMonitoringData(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching monitoring data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch monitoring data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Set initial time on client
    setCurrentTime(new Date().toLocaleTimeString());
    
    fetchMonitoringData();

    // Set up auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchMonitoringData();
      setCurrentTime(new Date().toLocaleTimeString());
    }, 30000);
    setRefreshInterval(interval);

    return () => {
      if (interval) clearInterval(interval);
    };
  }, []);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return 'text-green-600';
      case 'degraded':
        return 'text-yellow-600';
      case 'unhealthy':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'degraded':
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'unhealthy':
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return <Activity className="h-4 w-4 text-gray-600" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return 'destructive';
      case 'high':
        return 'destructive';
      case 'medium':
        return 'default';
      case 'low':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const preparePerformanceChartData = () => {
    if (!monitoringData?.performance_metrics) return [];

    const groupedData: { [key: string]: ChartDataPoint } = {};
    
    monitoringData.performance_metrics.forEach(metric => {
      const date = new Date(metric.timestamp).toLocaleDateString();
      if (!groupedData[date]) {
        groupedData[date] = { date };
      }
      groupedData[date][metric.metric_name] = metric.metric_value;
    });

    return Object.values(groupedData);
  };

  const prepareDriftChartData = () => {
    if (!monitoringData?.feature_drift) return [];

    return monitoringData.feature_drift.map(drift => ({
      feature: drift.feature_name,
      drift_score: drift.drift_score,
      p_value: drift.p_value
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2">Loading monitoring data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription><strong>Error:</strong> {error}</AlertDescription>
      </Alert>
    );
  }

  if (!monitoringData) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription><strong>No Data:</strong> No monitoring data available</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Monitoring Dashboard</h1>
        <Badge variant="outline" className="text-sm">
          Last updated: {currentTime}
        </Badge>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Alerts</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{monitoringData.summary.total_alerts}</div>
            <p className="text-xs text-muted-foreground">
              {monitoringData.summary.critical_alerts} critical
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Drift Features</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{monitoringData.summary.drift_features}</div>
            <p className="text-xs text-muted-foreground">
              Features showing drift
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Service Health</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {monitoringData.current_health?.filter(s => s.status === 'healthy').length || 0}/
              {monitoringData.current_health?.length || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Services healthy
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Unhealthy Services</CardTitle>
            <XCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{monitoringData.summary.unhealthy_services}</div>
            <p className="text-xs text-muted-foreground">
              Require attention
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="drift">Feature Drift</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="health">Service Health</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* Service Health Overview */}
          <Card>
            <CardHeader>
              <CardTitle>Service Health Overview</CardTitle>
              <CardDescription>Current status of all services</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {monitoringData.current_health?.map((service, index) => (
                  <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(service.status)}
                      <div>
                        <p className="font-medium">{service.service_name}</p>
                        <p className={`text-sm ${getStatusColor(service.status)}`}>
                          {service.status}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">{service.response_time_ms.toFixed(1)}ms</p>
                      <p className="text-xs text-muted-foreground">
                        {(service.error_rate * 100).toFixed(1)}% errors
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recent Alerts */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Alerts</CardTitle>
              <CardDescription>Latest system alerts and warnings</CardDescription>
            </CardHeader>
            <CardContent>
              {monitoringData.active_alerts.length === 0 ? (
                <p className="text-muted-foreground">No active alerts</p>
              ) : (
                <div className="space-y-2">
                  {monitoringData.active_alerts.slice(0, 5).map((alert, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center space-x-3">
                        <Badge variant={getSeverityColor(alert.severity)}>
                          {alert.severity}
                        </Badge>
                        <div>
                          <p className="font-medium">{alert.message}</p>
                          <p className="text-sm text-muted-foreground">
                            {alert.model_version} • {formatTimestamp(alert.timestamp)}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Model Performance Metrics</CardTitle>
              <CardDescription>Performance trends over time</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={preparePerformanceChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="accuracy" stroke="#8884d8" name="Accuracy" />
                    <Line type="monotone" dataKey="brier_score" stroke="#82ca9d" name="Brier Score" />
                    <Line type="monotone" dataKey="log_loss" stroke="#ffc658" name="Log Loss" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="drift" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Feature Drift Detection</CardTitle>
              <CardDescription>Features showing statistical drift</CardDescription>
            </CardHeader>
            <CardContent>
              {monitoringData.feature_drift.length === 0 ? (
                <p className="text-muted-foreground">No feature drift detected</p>
              ) : (
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={prepareDriftChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="drift_score" fill="#8884d8" name="Drift Score" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>All Active Alerts</CardTitle>
              <CardDescription>Complete list of system alerts</CardDescription>
            </CardHeader>
            <CardContent>
              {monitoringData.active_alerts.length === 0 ? (
                <p className="text-muted-foreground">No active alerts</p>
              ) : (
                <div className="space-y-3">
                  {monitoringData.active_alerts.map((alert, index) => (
                    <Alert key={index}>
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        <div className="flex items-center space-x-2 mb-2">
                          <Badge variant={getSeverityColor(alert.severity)}>
                            {alert.severity}
                          </Badge>
                          <span className="font-medium">{alert.alert_type}</span>
                        </div>
                        <p>{alert.message}</p>
                        <p className="text-sm text-muted-foreground mt-1">
                          Model: {alert.model_version} • {formatTimestamp(alert.timestamp)}
                        </p>
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="health" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Detailed Service Health</CardTitle>
              <CardDescription>Comprehensive service health metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {monitoringData.current_health?.map((service, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(service.status)}
                        <h3 className="font-medium">{service.service_name}</h3>
                        <Badge variant={service.status === 'healthy' ? 'default' : 'destructive'}>
                          {service.status}
                        </Badge>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Response Time</p>
                        <p className="font-medium">{service.response_time_ms.toFixed(1)}ms</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Error Rate</p>
                        <p className="font-medium">{(service.error_rate * 100).toFixed(1)}%</p>
                      </div>
                      {service.cpu_usage && (
                        <div>
                          <p className="text-muted-foreground">CPU Usage</p>
                          <div className="flex items-center space-x-2">
                            <Progress value={service.cpu_usage} className="flex-1" />
                            <span className="font-medium">{service.cpu_usage.toFixed(1)}%</span>
                          </div>
                        </div>
                      )}
                      {service.memory_usage && (
                        <div>
                          <p className="text-muted-foreground">Memory Usage</p>
                          <div className="flex items-center space-x-2">
                            <Progress value={service.memory_usage} className="flex-1" />
                            <span className="font-medium">{service.memory_usage.toFixed(1)}%</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MonitoringDashboard;
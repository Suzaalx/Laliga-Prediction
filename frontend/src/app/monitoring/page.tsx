import MonitoringDashboard from '@/components/MonitoringDashboard';

export default function MonitoringPage() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">System Monitoring Dashboard</h1>
      <MonitoringDashboard />
    </div>
  );
}
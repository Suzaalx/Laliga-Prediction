import { Suspense } from 'react';
import Header from '@/components/Header';
import LoadingSpinner from '@/components/LoadingSpinner';
import StandingsTable from '@/components/StandingsTable';

export default function StandingsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-gray-900 dark:to-slate-800">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Page Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600 mb-4">
            League Standings
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Current La Liga table with team performance and statistics
          </p>
        </div>

        {/* Standings Content */}
        <div className="max-w-6xl mx-auto">
          <Suspense fallback={<LoadingSpinner />}>
            <StandingsTable />
          </Suspense>
        </div>
      </main>
    </div>
  );
}
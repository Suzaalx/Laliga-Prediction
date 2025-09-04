import { Suspense } from 'react';
import Header from '@/components/Header';
import TeamsDisplay from '@/components/TeamsDisplay';
import LoadingSpinner from '@/components/LoadingSpinner';

export default function TeamsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 dark:text-white mb-4">
            La Liga <span className="text-blue-600 dark:text-blue-400">Teams</span>
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Comprehensive team data and analytics from CSV match records with enriched information
          </p>
        </div>

        {/* Teams Display */}
        <Suspense fallback={<LoadingSpinner />}>
          <TeamsDisplay />
        </Suspense>
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-gray-600 dark:text-gray-400">
            <p>&copy; 2024 La Liga Predictions. Powered by advanced machine learning.</p>
            <p className="mt-2 text-sm">
              Built with Next.js, TypeScript, and the enhanced Dixon-Coles model
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
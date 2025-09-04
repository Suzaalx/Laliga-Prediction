'use client';

import { useState } from 'react';
import Header from '@/components/Header';
import AnalyticsOverview from '@/components/AnalyticsOverview';
import PredictionsSection from '@/components/PredictionsSection';
import TeamsSection from '@/components/TeamsSection';
import { BarChart3, Target, Users, TrendingUp } from 'lucide-react';

type TabType = 'analytics' | 'predictions' | 'teams';

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>('analytics');

  const tabs = [
    {
      id: 'analytics' as TabType,
      label: 'Analytics',
      icon: BarChart3,
      description: 'League insights and statistics'
    },
    {
      id: 'predictions' as TabType,
      label: 'Predictions',
      icon: Target,
      description: 'Match predictions and probabilities'
    },
    {
      id: 'teams' as TabType,
      label: 'Teams',
      icon: Users,
      description: 'Team performance and rankings'
    }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'analytics':
        return <AnalyticsOverview />;
      case 'predictions':
        return <PredictionsSection />;
      case 'teams':
        return <TeamsSection />;
      default:
        return <AnalyticsOverview />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <Header />
      
      {/* Hero Section */}
      <section className="relative py-16 px-4 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-purple-600/10 backdrop-blur-3xl"></div>
        <div className="relative max-w-6xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-full px-4 py-2 mb-6 border border-blue-200 dark:border-gray-700">
            <TrendingUp className="h-4 w-4 text-blue-600" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">AI-Powered Football Analytics</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent mb-6">
            La Liga Intelligence
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed">
            Advanced analytics, match predictions, and team insights powered by machine learning. 
            Discover the future of football analysis.
          </p>
        </div>
      </section>

      {/* Navigation Tabs */}
      <div className="max-w-6xl mx-auto px-4 mb-8">
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl p-2 border border-gray-200 dark:border-gray-700 shadow-lg">
          <div className="grid grid-cols-3 gap-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    relative p-4 rounded-xl transition-all duration-300 group
                    ${isActive 
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg transform scale-105' 
                      : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300'
                    }
                  `}
                >
                  <div className="flex flex-col items-center gap-2">
                    <Icon className={`h-6 w-6 ${isActive ? 'text-white' : 'text-gray-500 group-hover:text-gray-700 dark:group-hover:text-gray-200'}`} />
                    <div className="text-center">
                      <div className={`font-semibold ${isActive ? 'text-white' : 'text-gray-900 dark:text-gray-100'}`}>
                        {tab.label}
                      </div>
                      <div className={`text-xs ${isActive ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'}`}>
                        {tab.description}
                      </div>
                    </div>
                  </div>
                  {isActive && (
                    <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/20 to-purple-600/20 animate-pulse"></div>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 pb-20">
        <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-3xl border border-gray-200 dark:border-gray-700 shadow-xl overflow-hidden">
          <div className="p-8">
            {renderTabContent()}
          </div>
        </div>
      </main>
    </div>
  );
}

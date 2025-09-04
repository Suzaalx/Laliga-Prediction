import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/model/performance`, {
      headers: {
        'Content-Type': 'application/json',
      },
      next: { revalidate: 3600 } // Revalidate every hour
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching model performance:', error);
    
    // Return mock data as fallback
    const mockPerformance = {
      calibrationCurve: {
        predicted: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        actual: [0.08, 0.18, 0.32, 0.41, 0.52, 0.58, 0.72, 0.79, 0.91]
      },
      monthlyAccuracy: [
        { month: 'Jan', accuracy: 65.2, predictions: 45 },
        { month: 'Feb', accuracy: 68.1, predictions: 42 },
        { month: 'Mar', accuracy: 71.3, predictions: 48 },
        { month: 'Apr', accuracy: 69.8, predictions: 46 },
        { month: 'May', accuracy: 72.5, predictions: 44 },
        { month: 'Jun', accuracy: 70.1, predictions: 41 }
      ],
      confusionMatrix: {
        homeWin: { predicted: 156, actual: 142 },
        draw: { predicted: 89, actual: 95 },
        awayWin: { predicted: 134, actual: 148 }
      },
      recentPerformance: [
        { date: '2024-01-01', accuracy: 68.5, brierScore: 0.234 },
        { date: '2024-01-08', accuracy: 71.2, brierScore: 0.221 },
        { date: '2024-01-15', accuracy: 69.8, brierScore: 0.245 },
        { date: '2024-01-22', accuracy: 73.1, brierScore: 0.198 },
        { date: '2024-01-29', accuracy: 70.5, brierScore: 0.232 }
      ],
      modelMetrics: {
        precision: 0.687,
        recall: 0.692,
        f1Score: 0.689,
        auc: 0.734
      },
      featureImportance: [
        { feature: 'home_form_goals_scored', importance: 0.156 },
        { feature: 'away_form_goals_conceded', importance: 0.142 },
        { feature: 'home_advantage', importance: 0.128 },
        { feature: 'head_to_head_home_wins', importance: 0.098 },
        { feature: 'recent_momentum', importance: 0.087 }
      ]
    };

    return NextResponse.json(mockPerformance);
  }
}
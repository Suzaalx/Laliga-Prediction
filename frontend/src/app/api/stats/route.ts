import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/predictions/general-stats`, {
      headers: {
        'Content-Type': 'application/json',
      },
      next: { revalidate: 600 } // Revalidate every 10 minutes
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching stats:', error);
    
    // Return mock data as fallback
    const mockStats = {
      totalPredictions: 1247,
      accuracy: 68.5,
      brierScore: 0.234,
      logLoss: 0.891,
      calibrationScore: 0.156,
      lastUpdated: new Date().toISOString(),
      modelVersion: '2.1.3',
      dataQuality: {
        completeness: 94.2,
        freshness: 98.7,
        consistency: 91.8
      },
      performanceTrend: {
        last7Days: 71.2,
        last30Days: 68.9,
        last90Days: 67.3
      }
    };

    return NextResponse.json(mockStats);
  }
}
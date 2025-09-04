import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/fixtures`, {
      headers: {
        'Content-Type': 'application/json',
      },
      // Add cache control for better performance
      next: { revalidate: 300 } // Revalidate every 5 minutes
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching fixtures:', error);
    
    // Return mock data as fallback
    const mockFixtures = [
      {
        id: 1,
        home_team: 'Real Madrid',
        away_team: 'Barcelona',
        match_date: '2024-02-15T20:00:00Z',
        status: 'upcoming',
        predictions: {
          home_win: 0.45,
          draw: 0.25,
          away_win: 0.30
        },
        confidence: 0.78,
        venue: 'Santiago Bernabéu'
      },
      {
        id: 2,
        home_team: 'Atlético Madrid',
        away_team: 'Sevilla',
        match_date: '2024-02-16T18:30:00Z',
        status: 'upcoming',
        predictions: {
          home_win: 0.52,
          draw: 0.28,
          away_win: 0.20
        },
        confidence: 0.71,
        venue: 'Wanda Metropolitano'
      },
      {
        id: 3,
        home_team: 'Valencia',
        away_team: 'Real Sociedad',
        match_date: '2024-02-17T16:15:00Z',
        status: 'upcoming',
        predictions: {
          home_win: 0.38,
          draw: 0.32,
          away_win: 0.30
        },
        confidence: 0.65,
        venue: 'Mestalla'
      }
    ];

    return NextResponse.json(mockFixtures);
  }
}
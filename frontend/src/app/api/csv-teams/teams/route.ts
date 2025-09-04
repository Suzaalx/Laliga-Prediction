import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/csv-teams/teams`, {
      headers: {
        'Content-Type': 'application/json',
      },
      next: { revalidate: 300 } // Revalidate every 5 minutes
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching CSV teams data:', error);
    return NextResponse.json(
      { error: 'Failed to fetch teams data' },
      { status: 500 }
    );
  }
}
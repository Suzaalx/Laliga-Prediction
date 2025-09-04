import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    console.log('Fetching league analytics from backend...');
    
    const response = await fetch(`${BACKEND_URL}/api/analytics/league`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store', // Ensure fresh data
    });

    if (!response.ok) {
      console.error(`Backend responded with status: ${response.status}`);
      return NextResponse.json(
        { success: false, error: `Backend error: ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log('Successfully fetched league analytics');
    
    return NextResponse.json({
      success: true,
      analytics: data
    });
    
  } catch (error) {
    console.error('Error fetching league analytics:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch league analytics' },
      { status: 500 }
    );
  }
}
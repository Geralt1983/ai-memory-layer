import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Check memory API health
    let memoryHealth = false;
    try {
      const response = await fetch('http://localhost:8001/health', {
        signal: AbortSignal.timeout(5000)
      });
      memoryHealth = response.ok;
    } catch (error) {
      console.log('Memory API not available');
    }

    return NextResponse.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      services: {
        nextjs: true,
        memory_api: memoryHealth
      }
    });
  } catch (error) {
    return NextResponse.json(
      { 
        status: 'error', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      },
      { status: 500 }
    );
  }
}
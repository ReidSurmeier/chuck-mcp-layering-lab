import { NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8001";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ resultId: string }> },
) {
  const apiKey = process.env.BACKEND_API_KEY;
  if (!apiKey) {
    return new Response(JSON.stringify({ error: "Missing authentication" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const { resultId } = await params;

  // Sanitize
  if (!/^[a-zA-Z0-9-]+$/.test(resultId)) {
    return new Response(JSON.stringify({ error: "Invalid result ID" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  try {
    const backendRes = await fetch(`${BACKEND}/api/result/${resultId}`, {
      headers: { "X-API-Key": apiKey },
    });

    if (!backendRes.ok) {
      return new Response(await backendRes.text(), {
        status: backendRes.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    const body = await backendRes.arrayBuffer();
    return new Response(body, {
      status: 200,
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "private, max-age=1800",
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Backend unreachable";
    return new Response(JSON.stringify({ error: message }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }
}

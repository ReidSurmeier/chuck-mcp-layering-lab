import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8001";

export async function POST(req: NextRequest) {
  const body = await req.formData();
  const apiKey = process.env.BACKEND_API_KEY ?? req.headers.get("X-API-Key");

  let backendRes: Response;
  try {
    backendRes = await fetch(`${BACKEND}/api/preview-stream`, {
      method: "POST",
      headers: {
        ...(apiKey ? { "X-API-Key": apiKey } : {}),
      },
      body,
      // @ts-expect-error - duplex needed for streaming request
      duplex: "half",
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Backend unreachable";
    return new Response(JSON.stringify({ error: message }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (!backendRes.ok) {
    const errorBody = await backendRes.text().catch(() => backendRes.statusText);
    return new Response(errorBody, {
      status: backendRes.status,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Manually pipe the SSE stream to avoid undici "other side closed" errors.
  // Direct body passthrough fails when backend closes the connection after
  // sending all data — standalone Next.js can't handle the close gracefully.
  const readable = new ReadableStream({
    async start(controller) {
      const reader = backendRes.body?.getReader();
      if (!reader) {
        controller.close();
        return;
      }
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          controller.enqueue(value);
        }
      } catch {
        // Backend closed connection — expected after stream completes
      } finally {
        controller.close();
      }
    },
  });

  return new Response(readable, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

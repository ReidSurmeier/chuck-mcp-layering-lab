import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8001";
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp", "image/tiff"];
const ALLOWED_FIELDS = ["image", "document"];

function validateFormData(formData: FormData): { valid: boolean; error?: string } {
  let fileCount = 0;
  let totalSize = 0;

  for (const [fieldName, value] of formData.entries()) {
    if (!ALLOWED_FIELDS.includes(fieldName)) {
      return { valid: false, error: `Disallowed field: ${fieldName}` };
    }

    if (value instanceof File) {
      fileCount++;
      if (fileCount > 1) {
        return { valid: false, error: "Multiple files not allowed" };
      }

      if (!ALLOWED_MIME_TYPES.includes(value.type)) {
        return { valid: false, error: `Invalid file type: ${value.type}` };
      }

      const fileSize = value.size;
      totalSize += fileSize;
      if (fileSize > MAX_FILE_SIZE) {
        return { valid: false, error: "File exceeds 10MB limit" };
      }
    } else if (typeof value === "string") {
      totalSize += Buffer.byteLength(value, "utf8");
    }

    if (totalSize > MAX_FILE_SIZE) {
      return { valid: false, error: "Request body exceeds 10MB limit" };
    }
  }

  if (fileCount === 0) {
    return { valid: false, error: "No file provided" };
  }

  return { valid: true };
}

export async function POST(request: NextRequest) {
  const apiKey = process.env.BACKEND_API_KEY;
  if (!apiKey) {
    return new NextResponse(JSON.stringify({ error: "Missing authentication" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const body = await request.formData();

  const validation = validateFormData(body);
  if (!validation.valid) {
    return new NextResponse(JSON.stringify({ error: validation.error }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const res = await fetch(`${BACKEND_URL}/api/preview`, {
    method: "POST",
    headers: { "X-API-Key": apiKey },
    body,
  });

  if (!res.ok) {
    return NextResponse.json({ error: "Backend preview failed" }, { status: res.status });
  }

  const blob = await res.blob();
  const manifest = res.headers.get("X-Manifest");

  return new NextResponse(blob, {
    status: 200,
    headers: {
      "Content-Type": "image/png",
      ...(manifest ? { "X-Manifest": manifest } : {}),
    },
  });
}

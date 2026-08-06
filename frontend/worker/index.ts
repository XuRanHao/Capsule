/** Cloudflare Worker entry point for the vinext-starter template. */
import { handleImageOptimization, DEFAULT_DEVICE_SIZES, DEFAULT_IMAGE_SIZES } from "vinext/server/image-optimization";
import handler from "vinext/server/app-router-entry";

interface Env {
  ASSETS: Fetcher;
  DB: D1Database;
  IMAGES: {
    input(stream: ReadableStream): {
      transform(options: Record<string, unknown>): {
        output(options: { format: string; quality: number }): Promise<{ response(): Response }>;
      };
    };
  };
}

interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}

// Image security config. SVG sources with .svg extension auto-skip the
// optimization endpoint on the client side (served directly, no proxy).
// To route SVGs through the optimizer (with security headers), set
// dangerouslyAllowSVG: true in next.config.js and uncomment below:
// const imageConfig: ImageConfig = { dangerouslyAllowSVG: true };

const worker = {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    // Production `vinext start` does not use Vite's development proxy. Keep
    // the public browser on one origin while forwarding application API calls
    // to the local Python service where all data and computation live.
    if (url.pathname === "/health" || url.pathname.startsWith("/api/")) {
      const apiUrl = new URL(request.url);
      apiUrl.protocol = "http:";
      apiUrl.hostname = "127.0.0.1";
      apiUrl.port = "8010";
      const headers = new Headers(request.headers);
      headers.set("x-forwarded-host", url.host);
      headers.set("x-forwarded-proto", url.protocol.slice(0, -1));
      const proxyRequest: RequestInit & { duplex?: "half" } = {
        headers,
        method: request.method,
      };
      if (request.body !== null) {
        proxyRequest.body = request.body;
        // Node's fetch implementation requires this flag when forwarding a
        // streaming body. Keeping the stream avoids buffering large videos in
        // the frontend process before sending them to the local API.
        proxyRequest.duplex = "half";
      }
      return fetch(new Request(apiUrl, proxyRequest));
    }

    if (url.pathname === "/_vinext/image") {
      const allowedWidths = [...DEFAULT_DEVICE_SIZES, ...DEFAULT_IMAGE_SIZES];
      return handleImageOptimization(request, {
        fetchAsset: (path) => env.ASSETS.fetch(new Request(new URL(path, request.url))),
        transformImage: async (body, { width, format, quality }) => {
          const result = await env.IMAGES.input(body).transform(width > 0 ? { width } : {}).output({ format, quality });
          return result.response();
        },
      }, allowedWidths);
    }

    return handler.fetch(request, env, ctx);
  },
};

export default worker;

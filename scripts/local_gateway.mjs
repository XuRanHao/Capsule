import http from "node:http";

// Listen on the LAN by default so nearby users can upload directly without
// routing large files through the public Cloudflare tunnel.
const listenHost = process.env.CAPSULE_GATEWAY_HOST ?? "0.0.0.0";
const listenPort = Number(process.env.CAPSULE_GATEWAY_PORT ?? "3000");
const frontendOrigin = new URL(
  process.env.CAPSULE_FRONTEND_ORIGIN ?? "http://127.0.0.1:3001",
);
const apiOrigin = new URL(
  process.env.CAPSULE_API_ORIGIN ?? "http://127.0.0.1:8010",
);

function targetFor(pathname) {
  return pathname === "/health" || pathname.startsWith("/api/")
    ? apiOrigin
    : frontendOrigin;
}

function proxy(request, response) {
  const requestUrl = new URL(request.url ?? "/", "http://capsule.local");
  const target = targetFor(requestUrl.pathname);
  const headers = { ...request.headers, host: target.host };
  const forwardedHost = request.headers.host;

  if (forwardedHost) headers["x-forwarded-host"] = forwardedHost;
  headers["x-forwarded-proto"] = request.headers["x-forwarded-proto"] ?? "http";

  const upstream = http.request(
    {
      protocol: target.protocol,
      hostname: target.hostname,
      port: target.port,
      method: request.method,
      path: request.url,
      headers,
    },
    (upstreamResponse) => {
      response.writeHead(
        upstreamResponse.statusCode ?? 502,
        upstreamResponse.statusMessage,
        upstreamResponse.headers,
      );
      upstreamResponse.pipe(response);
    },
  );

  upstream.on("error", (error) => {
    if (!response.headersSent) {
      response.writeHead(502, { "content-type": "application/json" });
      response.end(JSON.stringify({ detail: { message: error.message } }));
      return;
    }
    response.destroy(error);
  });

  request.on("aborted", () => upstream.destroy());
  request.pipe(upstream);
}

const server = http.createServer(proxy);
server.requestTimeout = 0;
server.headersTimeout = 65_000;

server.on("clientError", (_error, socket) => {
  if (socket.writable) socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
});

server.listen(listenPort, listenHost, () => {
  console.log(
    `Capsule gateway listening on http://${listenHost}:${listenPort} ` +
      `(frontend ${frontendOrigin}, api ${apiOrigin})`,
  );
});

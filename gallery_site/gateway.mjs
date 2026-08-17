import http from "node:http";

const listenHost = process.env.GATEWAY_HOST || "0.0.0.0";
const listenPort = Number(process.env.GATEWAY_PORT || 3000);
const sitePort = Number(process.env.SITE_PORT || 3001);
const bodyPort = Number(process.env.BODY_PORT || 7861);
const jobPort = Number(process.env.JOB_PORT || 7862);

function proxy(request, response, port) {
  const upstream = http.request(
    {
      hostname: "127.0.0.1",
      port,
      path: request.url,
      method: request.method,
      headers: { ...request.headers, host: `127.0.0.1:${port}` },
    },
    (upstreamResponse) => {
      response.writeHead(upstreamResponse.statusCode || 502, upstreamResponse.headers);
      upstreamResponse.pipe(response);
    },
  );
  upstream.on("error", (error) => {
    if (!response.headersSent) {
      response.writeHead(502, { "content-type": "application/json; charset=utf-8" });
    }
    response.end(JSON.stringify({ error: "upstream_unavailable", message: error.message }));
  });
  request.pipe(upstream);
}

http
  .createServer((request, response) => {
    const path = request.url || "/";
    const isBodyRequest =
      path.startsWith("/api/body/") || path.startsWith("/body-output/");
    const isJobRequest =
      path === "/api/jobs" ||
      path.startsWith("/api/jobs/") ||
      path.startsWith("/job-output/");
    proxy(
      request,
      response,
      isBodyRequest ? bodyPort : isJobRequest ? jobPort : sitePort,
    );
  })
  .listen(listenPort, listenHost, () => {
    console.log(
      JSON.stringify({
        event: "gateway_listening",
        address: `http://${listenHost}:${listenPort}`,
        site: `http://127.0.0.1:${sitePort}`,
        body: `http://127.0.0.1:${bodyPort}`,
        jobs: `http://127.0.0.1:${jobPort}`,
      }),
    );
  });

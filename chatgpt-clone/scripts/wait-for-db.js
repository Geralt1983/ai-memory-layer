const http = require("http");
function ping() {
  return new Promise((res) => { http.get({ host: "localhost", port: 3000, path: "/api/health" }, (r) => { res(r.statusCode === 200); }).on("error", () => res(false)); });
}
(async () => {
  const start = Date.now();
  while (Date.now() - start < 60000) {
    const ok = await ping();
    if (ok) process.exit(0);
    await new Promise(r => setTimeout(r, 1500));
  }
  console.error("DB health did not become ready within 60s");
  process.exit(1);
})();
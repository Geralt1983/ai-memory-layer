export const runtime = "nodejs";

export async function POST() {
  if (!process.env.ALLOW_DB_MIGRATION_API) return new Response("Forbidden", { status: 403 });
  const { spawn } = await import("child_process");
  const cmd = process.platform === "win32" ? "npx.cmd" : "npx";
  const child = spawn(cmd, ["prisma", "migrate", "deploy"], { env: process.env, cwd: process.cwd() });
  let out = ""; let err = "";
  child.stdout.on("data", (d) => (out += d.toString()));
  child.stderr.on("data", (d) => (err += d.toString()));
  const code: number = await new Promise((res) => child.on("close", res as any));
  if (code !== 0) return new Response(JSON.stringify({ ok: false, code, err }), { status: 500 });
  return Response.json({ ok: true, out });
}
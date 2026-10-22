import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const mode = process.argv[2];
if (!["pre-commit", "pre-push"].includes(mode)) {
  throw new Error("Expected pre-commit or pre-push");
}
const revisions = mode === "pre-commit"
  ? [execFileSync("git", ["write-tree"], { encoding: "utf8" }).trim()]
  : readFileSync(0, "utf8").trim().split("\n").filter(Boolean)
    .map((line) => line.split(/\s+/)[1]).filter((sha) => !/^0+$/.test(sha));

for (const revision of new Set(revisions)) {
  if (!/^[a-f0-9]{40,64}$/.test(revision)) throw new Error("Invalid Git object ID");
  const directory = mkdtempSync(join(tmpdir(), "hands-mode-check-"));
  try {
    const snapshot = join(directory, "snapshot");
    const archive = join(directory, "tree.tar");
    mkdirSync(snapshot);
    execFileSync("git", ["archive", "--format=tar", `--output=${archive}`, revision]);
    execFileSync("tar", ["-xf", archive, "-C", snapshot]);
    console.log(`Checking ${mode} snapshot ${revision}`);
    const result = spawnSync("npm", ["run", "check"], { cwd: snapshot, stdio: "inherit" });
    if (result.status !== 0) {
      process.exitCode = result.status ?? 1;
      break;
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

import { spawn } from "node:child_process";
import { readdir } from "node:fs/promises";
import { resolve } from "node:path";


async function collectTests(directory) {
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }

  const tests = [];
  for (const entry of entries) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      tests.push(...(await collectTests(path)));
    } else if (entry.isFile() && entry.name.endsWith(".test.ts")) {
      tests.push(path);
    }
  }
  return tests;
}


const roots =
  process.argv.length > 2
    ? process.argv.slice(2).map((path) => resolve(path))
    : [resolve("tests")];
const tests = (await Promise.all(roots.map(collectTests))).flat().sort();

if (tests.length === 0) {
  console.error("No frontend tests found; refusing a false-green test run.");
  process.exitCode = 1;
} else {
  const child = spawn(
    process.execPath,
    ["--experimental-strip-types", "--test", ...tests],
    { stdio: "inherit" }
  );
  child.on("error", (error) => {
    console.error(error);
    process.exitCode = 1;
  });
  child.on("exit", (code) => {
    process.exitCode = code ?? 1;
  });
}

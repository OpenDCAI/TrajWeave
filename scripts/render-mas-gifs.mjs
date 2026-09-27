import { existsSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const remotionEntry = join(root, "src/remotion/index.ts");
const outputRoot = join(root, ".remotion-output/mas-dataflow");
const assetRoot = join(root, "assets/diagrams");
const remotionBin = join(root, "node_modules/.bin/remotion");
const gifskiBin = existsSync(join(root, "node_modules/.bin/gifski"))
  ? join(root, "node_modules/.bin/gifski")
  : join(root, "node_modules/gifski/bin/debian/gifski");

const compositions = [
  { id: "MasDataFlowEN", gif: "mas-dataflow-en.gif", mp4: "mas-dataflow-en.mp4", frames: "frames-en" },
  { id: "MasDataFlowZH", gif: "mas-dataflow-zh.gif", mp4: "mas-dataflow-zh.mp4", frames: "frames-zh" },
];

const run = (command, args, options = {}) => {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    env: {
      ...process.env,
      NO_COLOR: "1",
    },
    ...options,
  });
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed with exit code ${result.status}`);
  }
};

if (!existsSync(remotionBin)) {
  throw new Error("Remotion CLI is missing. Run `npm install` first.");
}

if (!existsSync(gifskiBin)) {
  throw new Error("gifski is missing. Run `npm install` first.");
}

rmSync(outputRoot, { recursive: true, force: true });
mkdirSync(outputRoot, { recursive: true });
mkdirSync(assetRoot, { recursive: true });

for (const composition of compositions) {
  const mp4Path = join(outputRoot, composition.mp4);
  const framesDir = join(outputRoot, composition.frames);
  const gifPath = join(assetRoot, composition.gif);

  mkdirSync(framesDir, { recursive: true });

  run(remotionBin, [
    "render",
    remotionEntry,
    composition.id,
    mp4Path,
    "--codec=h264",
    "--crf=18",
    "--pixel-format=yuv420p",
    "--overwrite",
  ]);

  run("ffmpeg", [
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    mp4Path,
    "-vf",
    "fps=12",
    join(framesDir, "frame_%04d.png"),
  ]);

  const frameFiles = readdirSync(framesDir)
    .filter((name) => name.endsWith(".png"))
    .sort()
    .map((name) => join(framesDir, name));

  run(gifskiBin, [
    "--fps",
    "12",
    "--quality",
    "90",
    "--width",
    "1000",
    "--output",
    gifPath,
    ...frameFiles,
  ]);
}

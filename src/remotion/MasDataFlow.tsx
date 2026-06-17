import React from "react";
import {
  AbsoluteFill,
  Easing,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

type Language = "en" | "zh";

type Stage = {
  title: string;
  body: string;
};

type Labels = {
  title: string;
  eyebrow: string;
  subtitle: string;
  inference: string;
  training: string;
  task: Stage;
  protocol: Stage;
  policies: Stage;
  trajectory: Stage;
  reward: Stage;
  trainer: Stage;
  update: Stage;
  solver: string;
  verifier: string;
  tool: string;
  action: string;
  observation: string;
  credit: string;
  feedback: string;
};

const labels: Record<Language, Labels> = {
  en: {
    title: "TrajWeave MAS Data Flow",
    eyebrow: "trajectory-first multi-agent RL",
    subtitle: "Agent actions are woven into trajectories, scored, assigned credit, and sent back through VERL.",
    inference: "inference",
    training: "training",
    task: { title: "Task", body: "sample + environment" },
    protocol: { title: "Coordination", body: "who acts, who sees what" },
    policies: { title: "Agent Policies", body: "solver / verifier / tools" },
    trajectory: { title: "Trajectory Weave", body: "turns + messages + tool calls" },
    reward: { title: "Reward + Credit", body: "agent-wise advantage" },
    trainer: { title: "VERL Trainer", body: "DataProto batches" },
    update: { title: "Policy Update", body: "next rollout cycle" },
    solver: "Solver",
    verifier: "Verifier",
    tool: "Tool",
    action: "action",
    observation: "observation",
    credit: "credit",
    feedback: "feedback",
  },
  zh: {
    title: "TrajWeave MAS 数据流",
    eyebrow: "以轨迹为中心的多 Agent 强化学习",
    subtitle: "Agent 行为先被编织成轨迹，再进入奖励、归因和 VERL 训练，最后回流更新策略。",
    inference: "推理流",
    training: "训练流",
    task: { title: "任务", body: "样本 + 环境" },
    protocol: { title: "协同协议", body: "谁行动、谁看什么" },
    policies: { title: "Agent 策略", body: "求解器 / 验证器 / 工具" },
    trajectory: { title: "轨迹编织", body: "轮次 / 消息 / 工具" },
    reward: { title: "奖励 + 归因", body: "按 Agent 算优势" },
    trainer: { title: "VERL 训练器", body: "DataProto 批次" },
    update: { title: "策略更新", body: "进入下一轮 rollout" },
    solver: "求解器",
    verifier: "验证器",
    tool: "工具",
    action: "动作",
    observation: "观测",
    credit: "归因",
    feedback: "回流",
  },
};

const palette = {
  paper: "#FBFCF8",
  ink: "#0F172A",
  muted: "#64748B",
  faint: "#E2E8F0",
  rail: "#CBD5E1",
  blue: "#2563EB",
  teal: "#0F766E",
  amber: "#D97706",
  rose: "#E11D48",
  violet: "#6D28D9",
  green: "#16A34A",
  blueFill: "#EFF6FF",
  tealFill: "#ECFDF5",
  amberFill: "#FFFBEB",
  violetFill: "#F5F3FF",
};

const fontStack =
  "'Inter', 'DejaVu Sans', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', Arial, sans-serif";

type Point = {
  x: number;
  y: number;
};

type Cubic = [Point, Point, Point, Point];

const clampProgress = (value: number) => Math.max(0, Math.min(1, value));

const cubicPoint = ([p0, p1, p2, p3]: Cubic, progress: number): Point => {
  const t = clampProgress(progress);
  const mt = 1 - t;
  return {
    x: mt ** 3 * p0.x + 3 * mt ** 2 * t * p1.x + 3 * mt * t ** 2 * p2.x + t ** 3 * p3.x,
    y: mt ** 3 * p0.y + 3 * mt ** 2 * t * p1.y + 3 * mt * t ** 2 * p2.y + t ** 3 * p3.y,
  };
};

const cubicPath = ([p0, p1, p2, p3]: Cubic) =>
  `M ${p0.x} ${p0.y} C ${p1.x} ${p1.y}, ${p2.x} ${p2.y}, ${p3.x} ${p3.y}`;

const streamPaths: Array<{ curve: Cubic; color: string; width: number; delay: number }> = [
  {
    curve: [
      { x: 128, y: 224 },
      { x: 250, y: 125 },
      { x: 412, y: 122 },
      { x: 537, y: 235 },
    ],
    color: palette.blue,
    width: 10,
    delay: 0,
  },
  {
    curve: [
      { x: 272, y: 309 },
      { x: 374, y: 286 },
      { x: 430, y: 394 },
      { x: 552, y: 332 },
    ],
    color: palette.teal,
    width: 10,
    delay: 0.18,
  },
  {
    curve: [
      { x: 271, y: 383 },
      { x: 390, y: 430 },
      { x: 436, y: 230 },
      { x: 571, y: 288 },
    ],
    color: palette.amber,
    width: 10,
    delay: 0.34,
  },
];

const outboundPaths: Array<{ curve: Cubic; color: string; width: number; delay: number }> = [
  {
    curve: [
      { x: 606, y: 238 },
      { x: 686, y: 185 },
      { x: 725, y: 186 },
      { x: 790, y: 223 },
    ],
    color: palette.violet,
    width: 9,
    delay: 0.08,
  },
  {
    curve: [
      { x: 607, y: 330 },
      { x: 682, y: 382 },
      { x: 725, y: 385 },
      { x: 790, y: 358 },
    ],
    color: palette.green,
    width: 9,
    delay: 0.22,
  },
];

const feedbackPath: Cubic = [
  { x: 856, y: 439 },
  { x: 936, y: 452 },
  { x: 948, y: 148 },
  { x: 312, y: 151 },
];

const easeOut = Easing.bezier(0.16, 1, 0.3, 1);

const stageEntrance = (frame: number, fps: number, delaySeconds: number) =>
  interpolate(frame, [delaySeconds * fps, delaySeconds * fps + 0.65 * fps], [0.78, 1], {
    easing: easeOut,
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

const Card = ({
  x,
  y,
  w,
  h,
  tone,
  title,
  body,
  index,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  tone: string;
  title: string;
  body: string;
  index: number;
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const enter = stageEntrance(frame, fps, 0.1 + index * 0.08);

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y + interpolate(enter, [0, 1], [12, 0]),
        width: w,
        height: h,
        opacity: enter,
        background: "rgba(255,255,255,0.88)",
        border: `1.5px solid ${palette.faint}`,
        boxShadow: "0 16px 40px rgba(15, 23, 42, 0.07)",
        padding: "16px 18px",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          width: 36,
          height: 5,
          background: tone,
          marginBottom: 14,
        }}
      />
      <div style={{ color: palette.ink, fontSize: 24, lineHeight: 1.05, fontWeight: 650 }}>{title}</div>
      {body ? (
        <div style={{ color: palette.muted, fontSize: 14, lineHeight: 1.35, marginTop: 8 }}>{body}</div>
      ) : null}
    </div>
  );
};

const AgentChip = ({
  x,
  y,
  label,
  color,
  index,
}: {
  x: number;
  y: number;
  label: string;
  color: string;
  index: number;
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const enter = stageEntrance(frame, fps, 0.35 + index * 0.08);

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: 126,
        height: 42,
        opacity: enter,
        transform: `translateX(${interpolate(enter, [0, 1], [-14, 0])}px)`,
        border: `1.5px solid ${color}33`,
        background: "#FFFFFF",
        color,
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "0 14px",
        boxSizing: "border-box",
        fontSize: 16,
        fontWeight: 650,
        boxShadow: "0 10px 22px rgba(15, 23, 42, 0.05)",
      }}
    >
      <span style={{ width: 10, height: 10, background: color }} />
      <span>{label}</span>
    </div>
  );
};

const StageLabel = ({
  x,
  y,
  label,
  color,
}: {
  x: number;
  y: number;
  label: string;
  color: string;
}) => (
  <div
    style={{
      position: "absolute",
      left: x,
      top: y,
      color,
      fontSize: 13,
      fontWeight: 750,
      letterSpacing: 0,
      textTransform: "uppercase",
    }}
  >
    {label}
  </div>
);

const MicroLabel = ({
  x,
  y,
  label,
  color,
}: {
  x: number;
  y: number;
  label: string;
  color: string;
}) => (
  <div
    style={{
      position: "absolute",
      left: x,
      top: y,
      color,
      fontSize: 12,
      fontWeight: 700,
      background: "#FFFFFF",
      border: `1px solid ${color}33`,
      padding: "4px 8px",
    }}
  >
    {label}
  </div>
);

const AnimatedPath = ({
  curve,
  color,
  width,
  delay,
  dashed = false,
}: {
  curve: Cubic;
  color: string;
  width: number;
  delay: number;
  dashed?: boolean;
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const intro = stageEntrance(frame, fps, 0.55 + delay);
  const loop = ((frame + delay * fps * 18) % durationInFrames) / durationInFrames;

  return (
    <>
      <path
        d={cubicPath(curve)}
        fill="none"
        stroke={color}
        strokeWidth={width}
        strokeLinecap="round"
        opacity={0.16 * intro}
      />
      <path
        d={cubicPath(curve)}
        fill="none"
        stroke={color}
        strokeWidth={width * 0.52}
        strokeLinecap="round"
        pathLength={1}
        strokeDasharray={dashed ? "0.04 0.055" : "0.09 0.08"}
        strokeDashoffset={-loop}
        opacity={0.92 * intro}
      />
    </>
  );
};

const MovingToken = ({
  curve,
  color,
  offset,
  label,
}: {
  curve: Cubic;
  color: string;
  offset: number;
  label?: string;
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const progress = ((frame / durationInFrames + offset) % 1);
  const point = cubicPoint(curve, progress);
  const pulse = interpolate(Math.sin(progress * Math.PI * 2), [-1, 1], [0.9, 1.12]);

  return (
    <g transform={`translate(${point.x} ${point.y}) scale(${pulse})`}>
      <circle r="11" fill="#FFFFFF" stroke={color} strokeWidth="4" />
      <circle r="4" fill={color} />
      {label ? (
        <text x="16" y="5" fill={color} fontSize="12" fontWeight="700" fontFamily={fontStack}>
          {label}
        </text>
      ) : null}
    </g>
  );
};

const WeaveCore = ({ title, body }: Stage) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const enter = stageEntrance(frame, fps, 0.55);
  const breathe = interpolate(Math.sin((frame / fps) * Math.PI), [-1, 1], [0.98, 1.02]);

  return (
    <div
      style={{
        position: "absolute",
        left: 458,
        top: 196,
        width: 178,
        height: 154,
        opacity: enter,
        transform: `scale(${breathe})`,
        transformOrigin: "50% 50%",
        background: "rgba(255,255,255,0.92)",
        border: `1.5px solid ${palette.faint}`,
        boxShadow: "0 24px 50px rgba(15, 23, 42, 0.09)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        padding: 18,
        boxSizing: "border-box",
      }}
    >
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 12px)", gap: 6, marginBottom: 14 }}>
        {Array.from({ length: 12 }).map((_, index) => (
          <span
            key={index}
            style={{
              width: 12,
              height: 12,
              background:
                index % 4 === 0
                  ? palette.blue
                  : index % 4 === 1
                    ? palette.teal
                    : index % 4 === 2
                      ? palette.amber
                      : palette.violet,
              opacity: 0.18 + ((index + Math.floor(frame / 8)) % 4) * 0.18,
            }}
          />
        ))}
      </div>
      <div style={{ color: palette.ink, fontSize: 24, lineHeight: 1.06, fontWeight: 760 }}>{title}</div>
      <div style={{ color: palette.muted, fontSize: 13.5, lineHeight: 1.32, marginTop: 8 }}>{body}</div>
    </div>
  );
};

const DataPlane = ({ t }: { t: Labels }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const intro = stageEntrance(frame, fps, 0.2);
  const feedbackLoop = ((frame + 30) % 150) / 150;
  const feedbackPoint = cubicPoint(feedbackPath, feedbackLoop);

  return (
    <svg
      width="1000"
      height="560"
      viewBox="0 0 1000 560"
      style={{ position: "absolute", left: 0, top: 0, opacity: intro }}
    >
      <defs>
        <pattern id="dot-grid" width="28" height="28" patternUnits="userSpaceOnUse">
          <circle cx="2" cy="2" r="1.2" fill="#CBD5E1" opacity="0.5" />
        </pattern>
      </defs>
      <rect x="0" y="0" width="1000" height="560" fill="url(#dot-grid)" opacity="0.52" />
      <path d="M 56 486 H 930" stroke={palette.rail} strokeWidth="1.4" strokeDasharray="6 10" />
      <path d="M 56 154 H 930" stroke={palette.rail} strokeWidth="1.4" strokeDasharray="6 10" />

      {streamPaths.map((path) => (
        <AnimatedPath
          key={`${path.color}-${path.delay}`}
          curve={path.curve}
          color={path.color}
          width={path.width}
          delay={path.delay}
        />
      ))}
      {outboundPaths.map((path) => (
        <AnimatedPath
          key={`${path.color}-${path.delay}`}
          curve={path.curve}
          color={path.color}
          width={path.width}
          delay={path.delay}
        />
      ))}

      <path
        d={cubicPath(feedbackPath)}
        fill="none"
        stroke={palette.rose}
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray="0.045 0.035"
        strokeDashoffset={-feedbackLoop}
        pathLength={1}
        opacity="0.68"
      />
      <circle cx={feedbackPoint.x} cy={feedbackPoint.y} r="8" fill="#FFFFFF" stroke={palette.rose} strokeWidth="4" />

      {streamPaths.map((path, index) => (
        <MovingToken
          key={`token-${path.color}-${index}`}
          curve={path.curve}
          color={path.color}
          offset={index * 0.19}
        />
      ))}
      {outboundPaths.map((path, index) => (
        <MovingToken
          key={`out-token-${path.color}-${index}`}
          curve={path.curve}
          color={path.color}
          offset={0.28 + index * 0.23}
        />
      ))}
    </svg>
  );
};

export const MasDataFlow = ({ language }: { language: Language }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = labels[language];
  const titleEnter = stageEntrance(frame, fps, 0);

  return (
    <AbsoluteFill style={{ background: palette.paper, fontFamily: fontStack }}>
      <DataPlane t={t} />

      <div
        style={{
          position: "absolute",
          left: 54,
          top: 30,
          opacity: titleEnter,
          transform: `translateY(${interpolate(titleEnter, [0, 1], [14, 0])}px)`,
        }}
      >
        <div style={{ color: palette.teal, fontSize: 13, fontWeight: 780, textTransform: "uppercase" }}>
          {t.eyebrow}
        </div>
        <div style={{ color: palette.ink, fontSize: 34, lineHeight: 1.02, marginTop: 6, fontWeight: 780 }}>
          {t.title}
        </div>
        <div style={{ color: palette.muted, fontSize: 16, lineHeight: 1.35, marginTop: 10, width: 650 }}>
          {t.subtitle}
        </div>
      </div>

      <StageLabel x={56} y={137} label={t.inference} color={palette.blue} />
      <StageLabel x={56} y={468} label={t.training} color={palette.green} />

      <Card x={64} y={190} w={150} h={106} tone={palette.blue} title={t.task.title} body={t.task.body} index={0} />
      <Card
        x={224}
        y={112}
        w={172}
        h={110}
        tone={palette.violet}
        title={t.protocol.title}
        body={t.protocol.body}
        index={1}
      />
      <Card
        x={224}
        y={254}
        w={200}
        h={220}
        tone={palette.teal}
        title={t.policies.title}
        body=""
        index={2}
      />

      <AgentChip x={244} y={340} label={t.solver} color={palette.blue} index={0} />
      <AgentChip x={244} y={385} label={t.verifier} color={palette.teal} index={1} />
      <AgentChip x={244} y={430} label={t.tool} color={palette.amber} index={2} />

      <WeaveCore title={t.trajectory.title} body={t.trajectory.body} />

      <Card
        x={754}
        y={174}
        w={190}
        h={118}
        tone={palette.violet}
        title={t.reward.title}
        body={t.reward.body}
        index={3}
      />
      <Card
        x={754}
        y={320}
        w={190}
        h={112}
        tone={palette.green}
        title={t.trainer.title}
        body={t.trainer.body}
        index={4}
      />
      <Card
        x={652}
        y={440}
        w={198}
        h={104}
        tone={palette.rose}
        title={t.update.title}
        body={t.update.body}
        index={5}
      />

      <MicroLabel x={440} y={158} label={t.observation} color={palette.blue} />
      <MicroLabel x={856} y={130} label={t.feedback} color={palette.rose} />
    </AbsoluteFill>
  );
};

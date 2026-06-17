import React from "react";
import {
  AbsoluteFill,
  Easing,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

type Language = "en" | "zh";

type Labels = {
  title: string;
  subtitle: string;
  task: string;
  taskBody: string;
  teamSpec: string;
  agentSpec: string;
  coordination: string;
  coordinationBody: string;
  memory: string;
  memoryBody: string;
  ledger: string;
  ledgerBody: string;
  reward: string;
  credit: string;
  dataproto: string;
  trainer: string;
  update: string;
  solver: string;
  verifier: string;
  searcher: string;
  edge: string;
  message: string;
  toolCall: string;
  logprob: string;
  mask: string;
  advantage: string;
  workers: string;
};

const copy: Record<Language, Labels> = {
  en: {
    title: "TrajWeave: Multi-Agent RL Loop",
    subtitle: "Protocol, memory, agent turns, credit assignment, and VERL worker groups in one trajectory.",
    task: "Task / Env",
    taskBody: "problem, tools, state",
    teamSpec: "TeamSpec",
    agentSpec: "AgentSpec",
    coordination: "Coordination",
    coordinationBody: "next agent, visibility, stop rule",
    memory: "Memory / Blackboard",
    memoryBody: "shared facts + private scratchpads",
    ledger: "Trajectory Ledger",
    ledgerBody: "each turn is a trainable record",
    reward: "Reward",
    credit: "Credit Allocator",
    dataproto: "DataProto",
    trainer: "VERL Trainer",
    update: "Policy Update",
    solver: "Solver",
    verifier: "Verifier",
    searcher: "Searcher",
    edge: "edge",
    message: "message",
    toolCall: "tool call",
    logprob: "logprob",
    mask: "loss mask",
    advantage: "agent-wise advantage",
    workers: "worker groups",
  },
  zh: {
    title: "TrajWeave：多 Agent 强化学习闭环",
    subtitle: "把协议、记忆、Agent 轮次、归因和 VERL worker group 放进同一条轨迹里。",
    task: "任务 / 环境",
    taskBody: "题目、工具、状态",
    teamSpec: "TeamSpec",
    agentSpec: "AgentSpec",
    coordination: "协同协议",
    coordinationBody: "下个 agent、可见性、停止条件",
    memory: "记忆 / 黑板",
    memoryBody: "共享事实 + 私有草稿",
    ledger: "轨迹账本",
    ledgerBody: "每个 turn 都是可训练记录",
    reward: "奖励",
    credit: "归因分配器",
    dataproto: "DataProto",
    trainer: "VERL 训练器",
    update: "策略更新",
    solver: "求解器",
    verifier: "验证器",
    searcher: "搜索器",
    edge: "边",
    message: "消息",
    toolCall: "工具调用",
    logprob: "logprob",
    mask: "loss mask",
    advantage: "按 Agent 算优势",
    workers: "worker groups",
  },
};

const colors = {
  bg: "#FFFFFF",
  ink: "#111827",
  sub: "#5B6472",
  line: "#D9E1EC",
  grid: "#EDF2F7",
  blue: "#2563EB",
  teal: "#0F766E",
  amber: "#D97706",
  violet: "#7C3AED",
  rose: "#E11D48",
  green: "#16A34A",
  slate: "#334155",
  blueSoft: "#EFF6FF",
  tealSoft: "#ECFDF5",
  amberSoft: "#FFF7ED",
  violetSoft: "#F5F3FF",
  roseSoft: "#FFF1F2",
  greenSoft: "#F0FDF4",
};

const fontStack =
  "'Inter', 'DejaVu Sans', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', Arial, sans-serif";

type Point = {
  x: number;
  y: number;
};

type PathSpec = {
  id: string;
  start: Point;
  c1: Point;
  c2: Point;
  end: Point;
  color: string;
  delay: number;
};

const paths: PathSpec[] = [
  {
    id: "task-to-protocol",
    start: { x: 183, y: 177 },
    c1: { x: 224, y: 134 },
    c2: { x: 266, y: 136 },
    end: { x: 304, y: 165 },
    color: colors.blue,
    delay: 0,
  },
  {
    id: "team-to-ledger",
    start: { x: 211, y: 339 },
    c1: { x: 302, y: 331 },
    c2: { x: 372, y: 258 },
    end: { x: 489, y: 230 },
    color: colors.teal,
    delay: 0.16,
  },
  {
    id: "protocol-to-ledger",
    start: { x: 454, y: 169 },
    c1: { x: 490, y: 150 },
    c2: { x: 512, y: 173 },
    end: { x: 542, y: 202 },
    color: colors.violet,
    delay: 0.28,
  },
  {
    id: "memory-to-ledger",
    start: { x: 452, y: 397 },
    c1: { x: 500, y: 410 },
    c2: { x: 520, y: 347 },
    end: { x: 564, y: 325 },
    color: colors.amber,
    delay: 0.4,
  },
  {
    id: "ledger-to-credit",
    start: { x: 703, y: 211 },
    c1: { x: 747, y: 180 },
    c2: { x: 765, y: 152 },
    end: { x: 806, y: 147 },
    color: colors.rose,
    delay: 0.52,
  },
  {
    id: "ledger-to-dataproto",
    start: { x: 705, y: 310 },
    c1: { x: 748, y: 324 },
    c2: { x: 766, y: 314 },
    end: { x: 807, y: 306 },
    color: colors.green,
    delay: 0.64,
  },
  {
    id: "trainer-feedback",
    start: { x: 888, y: 432 },
    c1: { x: 950, y: 455 },
    c2: { x: 944, y: 113 },
    end: { x: 340, y: 129 },
    color: colors.rose,
    delay: 0.76,
  },
];

const pathD = (path: PathSpec) =>
  `M ${path.start.x} ${path.start.y} C ${path.c1.x} ${path.c1.y}, ${path.c2.x} ${path.c2.y}, ${path.end.x} ${path.end.y}`;

const cubicPoint = (path: PathSpec, t: number): Point => {
  const p = Math.max(0, Math.min(1, t));
  const mt = 1 - p;
  return {
    x:
      mt ** 3 * path.start.x +
      3 * mt ** 2 * p * path.c1.x +
      3 * mt * p ** 2 * path.c2.x +
      p ** 3 * path.end.x,
    y:
      mt ** 3 * path.start.y +
      3 * mt ** 2 * p * path.c1.y +
      3 * mt * p ** 2 * path.c2.y +
      p ** 3 * path.end.y,
  };
};

const enter = (frame: number, fps: number, delay: number) =>
  interpolate(frame, [delay * fps, delay * fps + 0.5 * fps], [0, 1], {
    easing: Easing.bezier(0.16, 1, 0.3, 1),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

const loop = (frame: number, fps: number, delay: number) => ((frame / fps + delay) % 3) / 3;

const Panel = ({
  x,
  y,
  w,
  h,
  title,
  tone,
  children,
  delay,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  tone: string;
  children: React.ReactNode;
  delay: number;
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = enter(frame, fps, delay);

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y + interpolate(p, [0, 1], [10, 0]),
        width: w,
        height: h,
        opacity: p,
        border: `1px solid ${colors.line}`,
        background: "rgba(255,255,255,0.94)",
        boxShadow: "0 12px 34px rgba(15, 23, 42, 0.07)",
        padding: 12,
        boxSizing: "border-box",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ width: 28, height: 5, background: tone }} />
        <span style={{ color: colors.ink, fontSize: 18, fontWeight: 760 }}>{title}</span>
      </div>
      <div style={{ marginTop: 10 }}>{children}</div>
    </div>
  );
};

const Pill = ({
  text,
  color,
  soft,
}: {
  text: string;
  color: string;
  soft: string;
}) => (
  <span
    style={{
      display: "inline-flex",
      alignItems: "center",
      height: 22,
      padding: "0 8px",
      border: `1px solid ${color}33`,
      background: soft,
      color,
      fontSize: 11,
      fontWeight: 720,
      marginRight: 5,
      marginBottom: 6,
      whiteSpace: "nowrap",
    }}
  >
    {text}
  </span>
);

const AgentRow = ({
  name,
  role,
  color,
  delay,
}: {
  name: string;
  role: string;
  color: string;
  delay: number;
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = enter(frame, fps, delay);

  return (
    <div
      style={{
        height: 30,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        border: `1px solid ${colors.line}`,
        padding: "0 9px",
        marginTop: 7,
        opacity: p,
        transform: `translateX(${interpolate(p, [0, 1], [-8, 0])}px)`,
        background: "#FFFFFF",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
        <span style={{ width: 9, height: 9, background: color }} />
        <span style={{ color: colors.ink, fontSize: 13, fontWeight: 730 }}>{name}</span>
      </div>
      <span style={{ color: colors.sub, fontSize: 11 }}>{role}</span>
    </div>
  );
};

const Sparkline = ({ color, values }: { color: string; values: number[] }) => (
  <div style={{ display: "flex", alignItems: "end", gap: 4, height: 34 }}>
    {values.map((value, index) => (
      <span
        key={index}
        style={{
          width: 8,
          height: 7 + value * 22,
          background: color,
          opacity: 0.3 + index * 0.08,
        }}
      />
    ))}
  </div>
);

const AnimatedSvg = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <svg width="1000" height="560" viewBox="0 0 1000 560" style={{ position: "absolute", inset: 0 }}>
      <defs>
        <pattern id="grid" width="22" height="22" patternUnits="userSpaceOnUse">
          <path d="M 22 0 H 0 V 22" fill="none" stroke={colors.grid} strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="1000" height="560" fill="url(#grid)" opacity="0.45" />
      <path d="M 38 96 H 960" stroke={colors.line} strokeDasharray="6 9" />
      <path d="M 38 506 H 960" stroke={colors.line} strokeDasharray="6 9" />

      {paths.map((path) => {
        const p = enter(frame, fps, 0.2 + path.delay * 0.2);
        const l = loop(frame, fps, path.delay);
        const point = cubicPoint(path, l);
        return (
          <g key={path.id} opacity={p}>
            <path d={pathD(path)} fill="none" stroke={path.color} strokeWidth="8" strokeLinecap="round" opacity="0.12" />
            <path
              d={pathD(path)}
              fill="none"
              stroke={path.color}
              strokeWidth="3.5"
              strokeLinecap="round"
              pathLength={1}
              strokeDasharray="0.08 0.07"
              strokeDashoffset={-l}
              opacity="0.9"
            />
            <circle cx={point.x} cy={point.y} r="10" fill="#FFFFFF" stroke={path.color} strokeWidth="3" />
            <circle cx={point.x} cy={point.y} r="4" fill={path.color} />
          </g>
        );
      })}
    </svg>
  );
};

const LedgerRow = ({
  index,
  agent,
  detail,
  color,
  tags,
}: {
  index: number;
  agent: string;
  detail: string;
  color: string;
  tags: string[];
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = enter(frame, fps, 0.55 + index * 0.16);
  const active = Math.floor((frame / 18) % 4) === index;

  return (
    <div
      style={{
        height: 51,
        border: `1px solid ${active ? color : colors.line}`,
        background: active ? `${color}10` : "#FFFFFF",
        marginBottom: 7,
        padding: "6px 8px",
        boxSizing: "border-box",
        opacity: p,
        transform: `translateY(${interpolate(p, [0, 1], [8, 0])}px)`,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ width: 8, height: 8, background: color }} />
          <span style={{ color: colors.ink, fontSize: 12.5, fontWeight: 760 }}>{agent}</span>
        </div>
        <span style={{ color: colors.sub, fontSize: 10.5 }}>turn_{String(index + 1).padStart(2, "0")}</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 5 }}>
        <span style={{ color: colors.sub, fontSize: 11, width: 72 }}>{detail}</span>
        {tags.map((tag) => (
          <span
            key={tag}
            style={{
              color,
              border: `1px solid ${color}30`,
              background: "#FFFFFF",
              fontSize: 9.5,
              padding: "2px 5px",
              whiteSpace: "nowrap",
            }}
          >
            {tag}
          </span>
        ))}
      </div>
    </div>
  );
};

const ScanBar = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const y = interpolate(frame % durationInFrames, [0, durationInFrames], [144, 421]);

  return (
    <div
      style={{
        position: "absolute",
        left: 494,
        top: y,
        width: 205,
        height: 26,
        border: `1px solid ${colors.blue}55`,
        background: "rgba(37, 99, 235, 0.06)",
        boxShadow: "0 0 18px rgba(37,99,235,0.10)",
      }}
    />
  );
};

export const MasDataFlow = ({ language }: { language: Language }) => {
  const t = copy[language];
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const titleP = enter(frame, fps, 0);

  return (
    <AbsoluteFill style={{ background: colors.bg, fontFamily: fontStack }}>
      <AnimatedSvg />

      <div
        style={{
          position: "absolute",
          left: 40,
          top: 24,
          opacity: titleP,
          transform: `translateY(${interpolate(titleP, [0, 1], [10, 0])}px)`,
        }}
      >
        <div style={{ color: colors.teal, fontSize: 12, fontWeight: 780, textTransform: "uppercase" }}>
          TrajWeave / VERL-MAS
        </div>
        <div style={{ color: colors.ink, fontSize: 29, fontWeight: 790, lineHeight: 1.05 }}>{t.title}</div>
        <div style={{ color: colors.sub, fontSize: 14.5, lineHeight: 1.3, marginTop: 6, width: 720 }}>{t.subtitle}</div>
      </div>

      <Panel x={40} y={123} w={165} h={98} title={t.task} tone={colors.blue} delay={0.05}>
        <div style={{ color: colors.sub, fontSize: 12.5, marginBottom: 8 }}>{t.taskBody}</div>
        <Pill text="env_obs" color={colors.blue} soft={colors.blueSoft} />
        <Pill text="tools" color={colors.amber} soft={colors.amberSoft} />
        <Pill text="done" color={colors.green} soft={colors.greenSoft} />
      </Panel>

      <Panel x={40} y={248} w={178} h={200} title={t.teamSpec} tone={colors.teal} delay={0.16}>
        <Pill text={t.agentSpec} color={colors.teal} soft={colors.tealSoft} />
        <Pill text="policy_group" color={colors.violet} soft={colors.violetSoft} />
        <AgentRow name={t.solver} role="wg_0" color={colors.blue} delay={0.26} />
        <AgentRow name={t.verifier} role="wg_1" color={colors.teal} delay={0.34} />
        <AgentRow name={t.searcher} role="tool" color={colors.amber} delay={0.42} />
        <div style={{ color: colors.sub, fontSize: 10.5, marginTop: 8 }}>shared / non-shared / heterogeneous</div>
      </Panel>

      <Panel x={292} y={120} w={175} h={130} title={t.coordination} tone={colors.violet} delay={0.22}>
        <div style={{ color: colors.sub, fontSize: 12.2, lineHeight: 1.35 }}>{t.coordinationBody}</div>
        <div style={{ marginTop: 9 }}>
          <Pill text="fixed" color={colors.violet} soft={colors.violetSoft} />
          <Pill text="router" color={colors.rose} soft={colors.roseSoft} />
          <Pill text="max_turn" color={colors.slate} soft="#F8FAFC" />
        </div>
      </Panel>

      <Panel x={292} y={360} w={175} h={96} title={t.memory} tone={colors.amber} delay={0.36}>
        <div style={{ color: colors.sub, fontSize: 12.2, lineHeight: 1.35 }}>{t.memoryBody}</div>
        <div style={{ marginTop: 8, display: "flex", gap: 4 }}>
          {[colors.blue, colors.teal, colors.amber, colors.violet, colors.rose].map((color, index) => (
            <span key={index} style={{ width: 22, height: 8, background: color, opacity: 0.35 + index * 0.11 }} />
          ))}
        </div>
      </Panel>

      <ScanBar />
      <Panel x={486} y={118} w={225} h={326} title={t.ledger} tone={colors.blue} delay={0.28}>
        <div style={{ color: colors.sub, fontSize: 12.2, marginBottom: 9 }}>{t.ledgerBody}</div>
        <LedgerRow
          index={0}
          agent={t.solver}
          detail={`${t.edge}: S->V`}
          color={colors.blue}
          tags={[t.message, t.logprob]}
        />
        <LedgerRow
          index={1}
          agent={t.verifier}
          detail={`${t.edge}: V->S`}
          color={colors.teal}
          tags={[t.mask, "approve/reject"]}
        />
        <LedgerRow
          index={2}
          agent={t.searcher}
          detail={t.toolCall}
          color={colors.amber}
          tags={["obs", "tool_result"]}
        />
        <LedgerRow
          index={3}
          agent={t.credit}
          detail={t.advantage}
          color={colors.rose}
          tags={["reward", "adv"]}
        />
      </Panel>

      <Panel x={792} y={104} w={168} h={106} title={t.reward} tone={colors.rose} delay={0.46}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Sparkline color={colors.rose} values={[0.15, 0.7, 0.35, 0.92, 0.52, 0.78]} />
          <div style={{ color: colors.rose, fontSize: 24, fontWeight: 800 }}>+1.0</div>
        </div>
        <Pill text="exact / judge / tool" color={colors.rose} soft={colors.roseSoft} />
      </Panel>

      <Panel x={792} y={230} w={168} h={102} title={t.credit} tone={colors.green} delay={0.54}>
        <Pill text="agent-wise" color={colors.green} soft={colors.greenSoft} />
        <Pill text="turn-wise" color={colors.blue} soft={colors.blueSoft} />
        <div style={{ display: "flex", gap: 6, marginTop: 5 }}>
          <Pill text="mu_k" color={colors.green} soft={colors.greenSoft} />
          <Pill text="sigma_k" color={colors.rose} soft={colors.roseSoft} />
        </div>
      </Panel>

      <Panel x={792} y={352} w={168} h={78} title={t.dataproto} tone={colors.violet} delay={0.62}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 4 }}>
          {Array.from({ length: 18 }).map((_, index) => (
            <span
              key={index}
              style={{
                height: 8,
                background: index % 3 === 0 ? colors.blue : index % 3 === 1 ? colors.teal : colors.violet,
                opacity: 0.25 + ((index + Math.floor(frame / 6)) % 5) * 0.12,
              }}
            />
          ))}
        </div>
      </Panel>

      <Panel x={756} y={454} w={204} h={72} title={t.trainer} tone={colors.slate} delay={0.7}>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <Pill text="wg_0" color={colors.blue} soft={colors.blueSoft} />
          <Pill text="wg_1" color={colors.teal} soft={colors.tealSoft} />
          <span style={{ color: colors.sub, fontSize: 11 }}>{t.workers}</span>
        </div>
      </Panel>

      <div
        style={{
          position: "absolute",
          left: 540,
          top: 465,
          width: 172,
          height: 56,
          border: `1px solid ${colors.line}`,
          background: "#FFFFFF",
          boxShadow: "0 12px 34px rgba(15, 23, 42, 0.07)",
          padding: "10px 12px",
          boxSizing: "border-box",
        }}
      >
        <div style={{ width: 30, height: 4, background: colors.green, marginBottom: 8 }} />
        <div style={{ color: colors.ink, fontSize: 17, fontWeight: 760 }}>{t.update}</div>
      </div>
    </AbsoluteFill>
  );
};

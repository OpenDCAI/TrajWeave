import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from "remotion";

type Language = "en" | "zh";

type Labels = {
  title: string;
  subtitle: string;
  inference: string;
  training: string;
  task: string;
  sample: string;
  orchestrator: string;
  schedule: string;
  agents: string;
  solver: string;
  verifier: string;
  tools: string;
  final: string;
  decision: string;
  trajectory: string;
  turns: string;
  reward: string;
  advantage: string;
  trainer: string;
  dataproto: string;
  policies: string;
  next: string;
};

const labels: Record<Language, Labels> = {
  en: {
    title: "TrajWeave MAS Data Flow",
    subtitle: "Inference trajectories feed reward, credit assignment, and VERL-backed policy updates.",
    inference: "Inference flow",
    training: "Training flow",
    task: "Task",
    sample: "benchmark sample",
    orchestrator: "Orchestrator",
    schedule: "who speaks next",
    agents: "Agent Team",
    solver: "Solver",
    verifier: "Verifier",
    tools: "Tools",
    final: "Final Output",
    decision: "answer or action",
    trajectory: "MultiAgentTrajectory",
    turns: "turns, tokens, tool calls",
    reward: "Reward + Credit",
    advantage: "agent-wise advantage",
    trainer: "VERL Trainer",
    dataproto: "DataProto batches",
    policies: "Updated Policies",
    next: "next rollout cycle",
  },
  zh: {
    title: "TrajWeave MAS 数据流",
    subtitle: "推理轨迹进入奖励、归因和 VERL 后端训练，再更新多 Agent 策略。",
    inference: "推理流",
    training: "训练流",
    task: "任务",
    sample: "评测样本",
    orchestrator: "编排器",
    schedule: "决定谁先说",
    agents: "Agent 团队",
    solver: "求解器",
    verifier: "验证器",
    tools: "工具调用",
    final: "最终输出",
    decision: "答案或动作",
    trajectory: "多 Agent 轨迹",
    turns: "轮次、token、工具调用",
    reward: "奖励 + 归因",
    advantage: "按 Agent 算优势",
    trainer: "VERL 训练器",
    dataproto: "DataProto 批次",
    policies: "策略更新",
    next: "进入下一轮 rollout",
  },
};

const colors = {
  bg: "#FAFBFF",
  ink: "#111827",
  muted: "#5B6472",
  blue: "#1A73E8",
  blueFill: "#E8F1FF",
  green: "#18A058",
  greenFill: "#E9F8EF",
  orange: "#F59E0B",
  orangeFill: "#FFF4DB",
  purple: "#7C3AED",
  purpleFill: "#F1EAFE",
  border: "#D7DEE8",
  grayFill: "#F3F6FA",
};

const fontStack =
  "'Inter', 'DejaVu Sans', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', Arial, sans-serif";

const Box = ({
  x,
  y,
  w,
  h,
  fill,
  title,
  subtitle,
  children,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  title: string;
  subtitle?: string;
  children?: React.ReactNode;
}) => (
  <div
    style={{
      position: "absolute",
      left: x,
      top: y,
      width: w,
      height: h,
      border: `2px solid ${colors.border}`,
      background: fill,
      display: "flex",
      flexDirection: "column",
      justifyContent: "center",
      padding: "0 24px",
      boxSizing: "border-box",
    }}
  >
    <div style={{ color: colors.ink, fontSize: 23, lineHeight: 1.05, whiteSpace: "nowrap" }}>{title}</div>
    {subtitle ? (
      <div style={{ color: colors.muted, fontSize: 15, lineHeight: 1.3, marginTop: 8, whiteSpace: "nowrap" }}>
        {subtitle}
      </div>
    ) : null}
    {children}
  </div>
);

const Line = ({
  x,
  y,
  w,
  h,
  color,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
}) => (
  <div
    style={{
      position: "absolute",
      left: x,
      top: y,
      width: w,
      height: h,
      background: color,
    }}
  />
);

const Arrow = ({ x, y, color, rotate = 0 }: { x: number; y: number; color: string; rotate?: number }) => (
  <div
    style={{
      position: "absolute",
      left: x,
      top: y,
      width: 0,
      height: 0,
      borderTop: "11px solid transparent",
      borderBottom: "11px solid transparent",
      borderLeft: `18px solid ${color}`,
      transform: `rotate(${rotate}deg)`,
      transformOrigin: "50% 50%",
    }}
  />
);

const Dot = ({ x, y, color }: { x: number; y: number; color: string }) => (
  <div
    style={{
      position: "absolute",
      left: x,
      top: y,
      width: 22,
      height: 22,
      borderRadius: 999,
      background: color,
      boxShadow: `0 0 0 6px ${color}1F`,
    }}
  />
);

const MovingDots = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const loop = (frame % durationInFrames) / durationInFrames;
  const pulse = interpolate(Math.sin(loop * Math.PI * 2), [-1, 1], [0.8, 1.08]);

  const inferenceX = interpolate(loop, [0, 1], [64, 885]);
  const captureY = interpolate(loop, [0, 1], [224, 316]);
  const trainingX = interpolate(loop, [0, 1], [130, 875]);
  const feedbackY = interpolate(loop, [0, 1], [444, 222]);

  return (
    <div style={{ transform: `scale(${pulse})`, transformOrigin: "500px 280px" }}>
      <Dot x={inferenceX} y={158} color={colors.blue} />
      <Dot x={562} y={captureY} color={colors.purple} />
      <Dot x={trainingX} y={438} color={colors.green} />
      <Dot x={925} y={feedbackY} color={colors.orange} />
    </div>
  );
};

export const MasDataFlow = ({ language }: { language: Language }) => {
  const t = labels[language];

  return (
    <AbsoluteFill style={{ background: colors.bg, fontFamily: fontStack }}>
      <div style={{ position: "absolute", left: 42, top: 26, color: colors.ink, fontSize: 30, fontWeight: 600 }}>
        {t.title}
      </div>
      <div style={{ position: "absolute", left: 44, top: 66, color: colors.muted, fontSize: 17 }}>{t.subtitle}</div>
      <div style={{ position: "absolute", left: 54, top: 104, color: colors.blue, fontSize: 18, fontWeight: 600 }}>
        {t.inference}
      </div>
      <div style={{ position: "absolute", left: 54, top: 360, color: colors.green, fontSize: 18, fontWeight: 600 }}>
        {t.training}
      </div>

      <Box x={60} y={130} w={135} h={72} fill={colors.blueFill} title={t.task} subtitle={t.sample} />
      <Box x={245} y={130} w={160} h={72} fill={colors.blueFill} title={t.orchestrator} subtitle={t.schedule} />
      <Box x={455} y={120} w={210} h={92} fill={colors.purpleFill} title={t.agents}>
        <div style={{ display: "flex", gap: 28, marginTop: 10, color: colors.purple, fontSize: 15 }}>
          <span>{t.solver}</span>
          <span>{t.verifier}</span>
        </div>
        <div style={{ color: colors.purple, fontSize: 15, marginTop: 4 }}>{t.tools}</div>
      </Box>
      <Box x={725} y={130} w={175} h={72} fill={colors.blueFill} title={t.final} subtitle={t.decision} />
      <Box x={405} y={265} w={245} h={78} fill={colors.grayFill} title={t.trajectory} subtitle={t.turns} />
      <Box x={120} y={402} w={210} h={82} fill={colors.greenFill} title={t.reward} subtitle={t.advantage} />
      <Box x={405} y={402} w={190} h={82} fill={colors.greenFill} title={t.trainer} subtitle={t.dataproto} />
      <Box x={675} y={402} w={220} h={82} fill={colors.orangeFill} title={t.policies} subtitle={t.next} />

      <Line x={195} y={165} w={50} h={4} color={colors.blue} />
      <Arrow x={226} y={154} color={colors.blue} />
      <Line x={405} y={165} w={50} h={4} color={colors.blue} />
      <Arrow x={436} y={154} color={colors.blue} />
      <Line x={665} y={165} w={60} h={4} color={colors.blue} />
      <Arrow x={706} y={154} color={colors.blue} />

      <Line x={560} y={212} w={4} h={53} color={colors.purple} />
      <Arrow x={552} y={238} color={colors.purple} rotate={90} />

      <Line x={405} y={343} w={4} h={34} color={colors.green} />
      <Line x={226} y={377} w={183} h={4} color={colors.green} />
      <Line x={226} y={381} w={4} h={21} color={colors.green} />
      <Arrow x={218} y={384} color={colors.green} rotate={90} />
      <Line x={330} y={443} w={75} h={4} color={colors.green} />
      <Arrow x={386} y={432} color={colors.green} />
      <Line x={595} y={443} w={80} h={4} color={colors.green} />
      <Arrow x={654} y={432} color={colors.green} />

      <Line x={895} y={443} w={30} h={4} color={colors.orange} />
      <Line x={925} y={235} w={4} h={212} color={colors.orange} />
      <Line x={620} y={235} w={305} h={4} color={colors.orange} />
      <Line x={620} y={212} w={4} h={23} color={colors.orange} />
      <Arrow x={612} y={211} color={colors.orange} rotate={-90} />

      <MovingDots />
    </AbsoluteFill>
  );
};

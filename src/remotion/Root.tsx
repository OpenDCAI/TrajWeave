import { Composition } from "remotion";
import { MasDataFlow } from "./MasDataFlow";

export const Root = () => {
  return (
    <>
      <Composition
        id="MasDataFlowEN"
        component={MasDataFlow}
        durationInFrames={180}
        fps={30}
        width={1000}
        height={560}
        defaultProps={{ language: "en" as const }}
      />
      <Composition
        id="MasDataFlowZH"
        component={MasDataFlow}
        durationInFrames={180}
        fps={30}
        width={1000}
        height={560}
        defaultProps={{ language: "zh" as const }}
      />
    </>
  );
};

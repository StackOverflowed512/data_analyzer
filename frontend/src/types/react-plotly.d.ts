declare module "react-plotly.js" {
  import { Component } from "react";

  interface PlotParams {
    data: any[];
    layout?: any;
    frames?: any[];
    config?: any;
    style?: React.CSSProperties;
    className?: string;
    onAfterExport?: () => void;
    onAfterPlot?: () => void;
    onAnimated?: () => void;
    onAnimatingFrame?: () => void;
    onBeforeExport?: () => void;
    onButtonClicked?: () => void;
    onClick?: () => void;
    onClickAnnotation?: () => void;
    onDeselect?: () => void;
    onDoubleClick?: () => void;
    onFramework?: () => void;
    onHover?: () => void;
    onInitialized?: () => void;
    onLegendClick?: () => void;
    onLegendDoubleClick?: () => void;
    onPurge?: () => void;
    onRedraw?: () => void;
    onRelayout?: () => void;
    onRestyle?: () => void;
    onSelected?: () => void;
    onSelecting?: () => void;
    onSliderChange?: () => void;
    onSliderEnd?: () => void;
    onSliderStart?: () => void;
    onTransitioning?: () => void;
    onTransitionInterrupted?: () => void;
    onUnhover?: () => void;
    onUpdate?: () => void;
    onWebGlContextLost?: () => void;
    revision?: number;
    useResizeHandler?: boolean;
    debug?: boolean;
    divId?: string;
  }

  export default class Plot extends Component<PlotParams> {}
}

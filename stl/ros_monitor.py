from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import jax.numpy as jnp
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from event_calculus import EventCalculusConfig, LiveTensorEventCalculus
from stl_prob import (
    ProbabilityConfig,
    ScallopFactExporter,
    scallop_facts_to_event_calculus_facts,
)

from .monitor import MonitorResult, STLDistanceMonitor
from .visualizer import STLRuntimeVisualizer


DistanceMonitorFactory = Callable[..., STLDistanceMonitor]


class STLCGDistanceMonitor(Node):
    """
    Thin ROS2 adapter for the salvaged STL monitor.

    It subscribes to pose topics and delegates all STL work to STLDistanceMonitor.
    """

    def __init__(
        self,
        monitor_factory: DistanceMonitorFactory = STLDistanceMonitor,
        visualizer: Optional[STLRuntimeVisualizer] = None,
    ):
        super().__init__("stlcg_distance_monitor")

        self.declare_parameter("threshold_m", 100.0)
        self.declare_parameter("window_sec", 5.0)
        self.declare_parameter("sample_rate_hz", 10.0)
        self.declare_parameter("rules_path", "")
        self.declare_parameter("visualization_dir", "")
        self.declare_parameter("scallop_fact_export_enabled", True)
        self.declare_parameter("scallop_fact_export_path", "")
        self.declare_parameter("scallop_fact_signal", "instant")
        self.declare_parameter("scallop_fact_scale", 1.0)
        self.declare_parameter("scallop_fact_midpoint", 0.0)
        self.declare_parameter("scallop_fact_include_groups", True)
        self.declare_parameter("scallop_fact_reset_on_start", True)
        self.declare_parameter("event_calculus_live_enabled", True)
        self.declare_parameter("event_calculus_hold_threshold", 0.5)

        self.threshold_m = float(self.get_parameter("threshold_m").value)
        self.window_sec = float(self.get_parameter("window_sec").value)
        self.sample_rate_hz = float(self.get_parameter("sample_rate_hz").value)
        self.scallop_fact_export_enabled = bool(
            self.get_parameter("scallop_fact_export_enabled").value
        )
        self.scallop_fact_signal = str(
            self.get_parameter("scallop_fact_signal").value
        ).strip()
        self.scallop_fact_scale = float(self.get_parameter("scallop_fact_scale").value)
        self.scallop_fact_midpoint = float(
            self.get_parameter("scallop_fact_midpoint").value
        )
        self.scallop_fact_include_groups = bool(
            self.get_parameter("scallop_fact_include_groups").value
        )
        self.scallop_fact_reset_on_start = bool(
            self.get_parameter("scallop_fact_reset_on_start").value
        )
        self.event_calculus_live_enabled = bool(
            self.get_parameter("event_calculus_live_enabled").value
        )
        self.event_calculus_hold_threshold = float(
            self.get_parameter("event_calculus_hold_threshold").value
        )
        configured_rules_path = str(self.get_parameter("rules_path").value).strip()
        configured_visualization_dir = str(
            self.get_parameter("visualization_dir").value
        ).strip()
        configured_fact_export_path = str(
            self.get_parameter("scallop_fact_export_path").value
        ).strip()

        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be greater than 0.0")
        self.sample_period_sec = 1.0 / self.sample_rate_hz
        self.window_steps = max(2, int(self.window_sec * self.sample_rate_hz))

        monitor_kwargs = {}
        if configured_rules_path:
            monitor_kwargs["rules_path"] = Path(configured_rules_path)
        self.monitor = monitor_factory(
            threshold_m=self.threshold_m,
            window_steps=self.window_steps,
            **monitor_kwargs,
        )
        self.visualizer = visualizer or STLRuntimeVisualizer(
            output_dir=(
                Path(configured_visualization_dir)
                if configured_visualization_dir
                else Path(__file__).with_name("stl_viz")
            )
        )
        default_fact_export_path = (
            Path(__file__).resolve().parents[1]
            / "stl_prob"
            / "exports"
            / "stl_live_facts.scl"
        )
        self.scallop_fact_exporter = (
            ScallopFactExporter(
                configured_fact_export_path or default_fact_export_path,
                config=ProbabilityConfig(
                    scale=self.scallop_fact_scale,
                    midpoint=self.scallop_fact_midpoint,
                    signal=self.scallop_fact_signal,
                ),
                include_groups=self.scallop_fact_include_groups,
                reset_on_start=self.scallop_fact_reset_on_start,
            )
            if self.scallop_fact_export_enabled
            else None
        )
        self.live_event_calculus = (
            LiveTensorEventCalculus(
                config=EventCalculusConfig(
                    hold_threshold=self.event_calculus_hold_threshold,
                )
            )
            if self.event_calculus_live_enabled
            else None
        )

        self.origin: Optional[jnp.ndarray] = None
        self.drone1: Optional[jnp.ndarray] = None
        self.drone2: Optional[jnp.ndarray] = None
        self._origin_received = False
        self._drone1_received = False
        self._drone2_received = False

        self.create_subscription(PoseStamped, "/origin/pose", self.origin_callback, 10)
        self.create_subscription(PoseStamped, "/Drone_1/pose", self.drone1_callback, 10)
        self.create_subscription(PoseStamped, "/Drone_2/pose", self.drone2_callback, 10)
        self.create_timer(self.sample_period_sec, self.sample_and_monitor)

        self.visualizer.render(
            compiled_rules=self.monitor.compiled_rules,
            combine_operator=self.monitor.combine_operator,
            result=None,
            status_message="waiting for pose topics",
        )
        self.get_logger().info("Salvaged STL distance monitor started.")
        self.get_logger().info(f"threshold_m={self.threshold_m}")
        self.get_logger().info(f"window_sec={self.window_sec}")
        self.get_logger().info(f"sample_rate_hz={self.sample_rate_hz}")
        self.get_logger().info(f"window_steps={self.window_steps}")
        self.get_logger().info(f"rules_path={self.monitor.rules_path}")
        self.get_logger().info(f"visualization_svg={self.visualizer.artifacts.svg_path}")
        self.get_logger().info(f"visualization_json={self.visualizer.artifacts.json_path}")
        if self.scallop_fact_exporter is not None:
            self.get_logger().info(
                f"scallop_fact_export_path={self.scallop_fact_exporter.output_path}"
            )
            self.get_logger().info(
                "scallop_fact_export="
                f"signal={self.scallop_fact_signal} "
                f"scale={self.scallop_fact_scale} "
                f"midpoint={self.scallop_fact_midpoint} "
                f"include_groups={self.scallop_fact_include_groups}"
            )
        if self.live_event_calculus is not None:
            self.get_logger().info(
                "event_calculus_live=true "
                f"hold_threshold={self.event_calculus_hold_threshold}"
            )

    @staticmethod
    def _pose_to_array(msg: PoseStamped) -> jnp.ndarray:
        p = msg.pose.position
        return jnp.array([p.x, p.y, p.z], dtype=jnp.float32)

    def origin_callback(self, msg: PoseStamped) -> None:
        self.origin = self._pose_to_array(msg)
        if not self._origin_received:
            self._origin_received = True
            self.get_logger().info("Received first pose on /origin/pose")

    def drone1_callback(self, msg: PoseStamped) -> None:
        self.drone1 = self._pose_to_array(msg)
        if not self._drone1_received:
            self._drone1_received = True
            self.get_logger().info("Received first pose on /Drone_1/pose")

    def drone2_callback(self, msg: PoseStamped) -> None:
        self.drone2 = self._pose_to_array(msg)
        if not self._drone2_received:
            self._drone2_received = True
            self.get_logger().info("Received first pose on /Drone_2/pose")

    def sample_and_monitor(self) -> None:
        if self.origin is None or self.drone1 is None or self.drone2 is None:
            self.get_logger().debug("Waiting for all pose topics...")
            return

        result = self.monitor.add_sample(
            self.origin,
            self.drone1,
            self.drone2,
            sample_time_ns=self.get_clock().now().nanoseconds,
        )
        self.visualizer.render(
            compiled_rules=self.monitor.compiled_rules,
            combine_operator=self.monitor.combine_operator,
            result=result,
        )
        if not result.is_ready:
            self.get_logger().info(
                f"Collecting samples: {result.collected_samples}/{result.window_steps}"
            )
            return
        exported_fact_count = self._export_scallop_facts(result)
        self._log_result(result)
        if exported_fact_count is not None:
            self.get_logger().info(
                f"scallop_facts_exported={exported_fact_count} "
                f"path={self.scallop_fact_exporter.output_path}"
            )

    def _export_scallop_facts(self, result: MonitorResult) -> Optional[int]:
        if self.scallop_fact_exporter is None:
            return None
        facts = self.scallop_fact_exporter.export_result(result)
        self._process_event_calculus_facts(facts)
        return len(facts)

    def _process_event_calculus_facts(self, facts) -> None:
        if self.live_event_calculus is None or not facts:
            return
        result = self.live_event_calculus.feed(
            scallop_facts_to_event_calculus_facts(facts)
        )
        if not result.intervals:
            self.get_logger().info("event_calculus_intervals=none")
            return
        lines = ["Event Calculus true intervals:"]
        for interval in result.intervals:
            lines.append(
                "  "
                f"{interval.source_kind}:{interval.name} "
                f"[{interval.start_sample_index}, {interval.end_sample_index}) "
                f"p_min={interval.min_probability:.3f} "
                f"p_max={interval.max_probability:.3f}"
            )
        self.get_logger().info("\n".join(lines))

    def _log_result(self, result: MonitorResult) -> None:
        lines = ["", "Instantaneous rule evaluations:"]
        for rule in result.rule_evaluations:
            lines.append(
                f"  {rule.label}: value={rule.current_value:.2f} m, "
                f"rho_now={rule.instant_robustness:.2f}"
            )
        lines.append("STL window robustness:")
        for rule in result.rule_evaluations:
            start, end = rule.interval
            lines.append(
                f"  {rule.label}, {rule.temporal_op}[{start},{end}]: "
                f"rho={rule.window_robustness:.2f}"
            )
        lines.append("Grouped formulas:")
        for group in result.group_evaluations:
            lines.append(
                f"  {group.name}: rules={group.rule_count}, "
                f"rho={group.window_robustness:.2f}, "
                f"ok_now={group.ok_now}, ok_window={group.ok_window}"
            )
        lines.append(f"sample_index={result.sample_index}, timestamp_ns={result.timestamp_ns}")
        lines.append(f"ok_now={result.ok_now}, ok_window={result.ok_window}")
        self.get_logger().info("\n".join(lines))

    def destroy_node(self):
        self.monitor.close()
        return super().destroy_node()

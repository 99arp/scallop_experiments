from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import jax.numpy as jnp
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

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

        self.threshold_m = float(self.get_parameter("threshold_m").value)
        self.window_sec = float(self.get_parameter("window_sec").value)
        self.sample_rate_hz = float(self.get_parameter("sample_rate_hz").value)
        configured_rules_path = str(self.get_parameter("rules_path").value).strip()
        configured_visualization_dir = str(
            self.get_parameter("visualization_dir").value
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
        self._log_result(result)

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

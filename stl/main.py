#!/usr/bin/env python3

from pathlib import Path

import rclpy

from .ros_monitor import STLCGDistanceMonitor
from .visualizer import STLRuntimeVisualizer


def main(args=None):
    rclpy.init(args=args)
    visualizer = STLRuntimeVisualizer(output_dir=Path(__file__).with_name("stl_viz"))
    node = STLCGDistanceMonitor(visualizer=visualizer)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 用法：python extract_from_bag_time.py D:\GraduationProjectCode\dataset\real_data\wheelchock.bag -o extracted_data/extracted_data6 --start_time 9 --end_time 14

import pyrealsense2 as rs
import cv2
import numpy as np
import os
import argparse
import pandas as pd

def extract_bag(bag_path, output_dir, step=1, max_frames=None, start_time=None, end_time=None):
    rgb_dir = os.path.join(output_dir, 'rgb')
    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    timestamps_list = []

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)

    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale}")

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()
    print("Camera intrinsics:")
    print(f"  fx={color_intr.fx}, fy={color_intr.fy}")
    print(f"  cx={color_intr.ppx}, cy={color_intr.ppy}")
    print(f"  width={color_intr.width}, height={color_intr.height}")

    with open(os.path.join(output_dir, 'camera_intrinsics.txt'), 'w') as f:
        f.write(f"fx {color_intr.fx}\n")
        f.write(f"fy {color_intr.fy}\n")
        f.write(f"cx {color_intr.ppx}\n")
        f.write(f"cy {color_intr.ppy}\n")
        f.write(f"width {color_intr.width}\n")
        f.write(f"height {color_intr.height}\n")
        f.write(f"depth_scale {depth_scale}\n")

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)

    frame_count = 0
    saved_count = 0
    first_timestamp = None
    print("开始提取帧...")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if not frames:
                break

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            timestamp_ms = color_frame.get_timestamp()
            if first_timestamp is None:
                first_timestamp = timestamp_ms
                print(f"首帧时间戳（毫秒）: {first_timestamp}")

            rel_sec = (timestamp_ms - first_timestamp) / 1000.0

            # 时间区间筛选
            if start_time is not None and rel_sec < start_time:
                frame_count += 1
                continue
            if end_time is not None and rel_sec > end_time:
                print(f"已达到结束时间 {end_time}s，停止提取。")
                break

            # 步长筛选
            if frame_count % step == 0:
                rgb_image = np.asanyarray(color_frame.get_data())
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                depth_image = np.asanyarray(depth_frame.get_data())

                rgb_filename = os.path.join(rgb_dir, f"frame_{saved_count:06d}.png")
                depth_filename = os.path.join(depth_dir, f"frame_{saved_count:06d}.png")
                cv2.imwrite(rgb_filename, bgr_image)
                cv2.imwrite(depth_filename, depth_image)

                timestamps_list.append([saved_count, timestamp_ms, rel_sec])

                saved_count += 1

                if saved_count % 100 == 0:
                    print(f"已保存 {saved_count} 帧 (相对时间 {rel_sec:.2f}s)")

                if max_frames is not None and saved_count >= max_frames:
                    print(f"达到最大保存帧数 {max_frames}，停止提取。")
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"发生异常: {e}")
    finally:
        pipeline.stop()
        if timestamps_list:
            timestamps_df = pd.DataFrame(timestamps_list, columns=['frame_idx', 'timestamp_ms', 'rel_sec'])
            timestamps_path = os.path.join(output_dir, 'timestamps.csv')
            timestamps_df.to_csv(timestamps_path, index=False)
            print(f"时间戳已保存到 {timestamps_path}")
        print(f"共读取 {frame_count} 帧，实际保存 {saved_count} 帧到 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 RealSense .bag 文件中提取对齐的 RGB 和深度图（支持相对时间）")
    parser.add_argument("bag_file", help="输入 .bag 文件路径")
    parser.add_argument("-o", "--output", default="extracted_data", help="输出目录")
    parser.add_argument("--step", type=int, default=1, help="每隔 step 帧保存一帧")
    parser.add_argument("--max_frames", type=int, default=None, help="最多保存的帧数")
    parser.add_argument("--start_time", type=float, default=None, help="起始时间（秒，相对于视频开始）")
    parser.add_argument("--end_time", type=float, default=None, help="结束时间（秒，相对于视频开始）")
    args = parser.parse_args()

    extract_bag(args.bag_file, args.output, step=args.step, max_frames=args.max_frames,
                start_time=args.start_time, end_time=args.end_time)
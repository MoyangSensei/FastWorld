"""Build the validation CAN inputs from official nuScenes records."""

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import pickle

import numpy as np
from pyquaternion import Quaternion

MONITOR_FIELDS = (
    "brake", "brake_switch", "throttle", "vehicle_speed", "steering",
    "yaw_rate", "left_signal", "right_signal",
)
PEDAL_FIELDS = ("brake_sensor", "throttle_sensor", "steering_sensor")
MONITOR_LAGS = (100_000, 600_000, 1_100_000, 1_600_000)
PEDAL_WINDOWS = ((100_000, 200_000), (200_000, 600_000), (600_000, 1_100_000), (1_100_000, 1_600_000))
POSE_LAGS = (0, 250_000, 500_000, 1_000_000, 2_000_000)
RECENT_MASKS = (4, 9, 14, 19, 24, 33, 40, 47)


def read_infos(path):
    with path.open("rb") as stream:
        return pickle.load(stream)["infos"]


def read_stream(root, scene, kind):
    path = root / f"{scene}_{kind}.json"
    records = json.loads(path.read_text()) if path.is_file() else []
    records.sort(key=lambda row: row["utime"])
    return [row["utime"] for row in records], records


def read_at(stream, cutoff, max_age):
    times, records = stream
    index = bisect_right(times, cutoff) - 1
    if index >= 0 and cutoff - times[index] <= max_age:
        return records[index]
    return None


def observed_motion(rows, output, indices):
    previous = previous_velocity = previous_dt = None
    previous_displacement = np.zeros(3)
    for index in sorted(indices, key=lambda i: rows[i]["timestamp"]):
        row = rows[index]
        ego_rotation = Quaternion(row["ego2global_rotation"])
        lidar_rotation = Quaternion(row["lidar2ego_rotation"])
        position = np.asarray(row["ego2global_translation"], dtype=np.float64)
        lidar_position = position + ego_rotation.rotate(row["lidar2ego_translation"])
        to_lidar = (ego_rotation * lidar_rotation).inverse.rotation_matrix
        yaw = ego_rotation.yaw_pitch_roll[0]
        stamp = row["timestamp"]
        dt = (stamp - previous[0]) / 1e6 if previous else 0
        valid = .35 <= dt <= .65
        displacement = lidar_position - previous[2] if valid else np.zeros(3)
        velocity = (position - previous[1]) / dt if valid else np.zeros(3)
        acceleration = ((velocity - previous_velocity) / ((dt + previous_dt) / 2)
                        if valid and previous_velocity is not None else np.zeros(3))
        yaw_rate = ((yaw - previous[3] + np.pi) % (2 * np.pi) - np.pi) / dt if valid else 0
        prior = previous_displacement if valid else np.zeros(3)
        history = np.stack((to_lidar @ prior, to_lidar @ displacement))[:, :2].reshape(4)
        speed = np.linalg.norm(velocity[:2])
        output[index] = np.r_[history, (to_lidar @ velocity)[:2], (to_lidar @ acceleration)[:2],
                              yaw_rate, 4.084, 1.85, speed, yaw_rate / max(speed, .5)]
        previous = stamp, position, lidar_position, yaw
        previous_velocity = velocity if valid else None
        previous_dt = dt if valid else None
        previous_displacement = displacement


def store_observation(destination, values):
    if np.isfinite(values).all():
        destination[:-1] = values
        destination[-1] = 1


def store_window(destination, times, values, stamp, near, far):
    start = np.searchsorted(times, stamp - far)
    end = np.searchsorted(times, stamp - near, side="right")
    if end > start:
        window = values[start:end]
        if np.isfinite(window).all():
            store_observation(destination, np.r_[window.mean(0), np.ptp(window, axis=0)])


def collect(rows, scene_names, can_root):
    size = len(rows)
    motion = np.empty((size, 13), dtype=np.float32)
    monitor_features = np.zeros((size, 4, 9), dtype=np.float32)
    pedal_features = np.zeros((size, 4, 7), dtype=np.float32)
    recent = np.zeros((size, 48), dtype=np.float32)
    scenes = defaultdict(list)
    for index, row in enumerate(rows):
        scenes[row["scene_token"]].append(index)

    for scene, indices in scenes.items():
        observed_motion(rows, motion, indices)
        name = scene_names[scene]
        pose = read_stream(can_root, name, "pose")
        steering = read_stream(can_root, name, "steeranglefeedback")
        monitor = read_stream(can_root, name, "vehicle_monitor")
        times, pedal_records = read_stream(can_root, name, "zoesensors")
        pedal_times = np.asarray(times, dtype=np.int64)
        pedal = np.asarray([[row[field] for field in PEDAL_FIELDS] for row in pedal_records])
        # The earlier pedal windows were evaluated in float32. The recent
        # windows use the original JSON precision before writing float32.
        pedal_f32 = pedal.astype(np.float32)
        for index in indices:
            stamp = rows[index]["timestamp"]
            record = read_at(pose, stamp, 200_000)
            if record is not None:
                values = np.r_[record["accel"][:2], record["vel"][0]]
                if np.isfinite(values).all():
                    motion[index, [6, 7, 11]] = values
            record = read_at(steering, stamp, 200_000)
            if record is not None and np.isfinite(record["value"]):
                sign = -1 if rows[index]["map_location"].startswith("singapore") else 1
                motion[index, 12] = sign * 2 * record["value"] / 2.588

            for column, lag in enumerate(MONITOR_LAGS):
                record = read_at(monitor, stamp - lag, 650_000)
                if record is not None:
                    values = np.asarray([record[field] for field in MONITOR_FIELDS], dtype=np.float32)
                    store_observation(monitor_features[index, column], values)
            for column, (near, far) in enumerate(PEDAL_WINDOWS):
                store_window(pedal_features[index, column], pedal_times, pedal_f32, stamp, near, far)

            for column, lag in enumerate(POSE_LAGS):
                record = read_at(pose, stamp - lag, 120_000)
                if record is not None:
                    store_observation(recent[index, column * 5:column * 5 + 5], np.r_[record["vel"][:2], record["accel"][:2]])
            record = read_at(monitor, stamp, 200_000)
            if record is not None:
                store_observation(recent[index, 25:34], [record[field] for field in MONITOR_FIELDS])
            store_window(recent[index, 34:41], pedal_times, pedal, stamp, 0, 100_000)
            store_window(recent[index, 41:48], pedal_times, pedal, stamp, 0, 300_000)

    return motion, monitor_features, pedal_features, recent


def scale_block(train, validation, mask_train, mask_val, minimum_scale):
    observed = train[mask_train]
    if len(observed) == 0:
        raise ValueError("A CAN stream is missing from the train split. Check the CAN bus download and path.")
    mean = observed.mean(0)
    scale = observed.std(0).clip(minimum_scale)
    return np.clip((validation - mean) / scale, -8, 8) * mask_val[..., None]


def prepare(train_info, val_info, scene_json, can_root, output):
    scene_names = {row["token"]: row["name"] for row in json.loads(scene_json.read_text())}
    train = read_infos(train_info)
    val = read_infos(val_info)
    _, train_monitor, train_pedal, train_recent = collect(train, scene_names, can_root)
    motion, monitors, pedals, recent = collect(val, scene_names, can_root)
    for train_block, val_block, minimum in ((train_monitor, monitors, .05), (train_pedal, pedals, .005)):
        val_block[..., :-1] = scale_block(train_block[..., :-1], val_block[..., :-1],
                                          train_block[..., -1].astype(bool), val_block[..., -1].astype(bool), minimum)
    start = 0
    for end in RECENT_MASKS:
        recent[:, start:end] = scale_block(train_recent[:, start:end], recent[:, start:end],
                                           train_recent[:, end].astype(bool), recent[:, end].astype(bool), .005)
        start = end + 1
    controls = np.concatenate((monitors.reshape(len(val), 36), pedals.reshape(len(val), 28), recent), axis=1)
    if not np.isfinite(motion).all() or not np.isfinite(controls).all():
        raise ValueError("Non-finite CAN features.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, tokens=np.asarray([row["token"] for row in val]),
                            timestamps=np.asarray([row["timestamp"] for row in val], dtype=np.int64),
                            motion=motion, controls=controls)
    print(f"Prepared {len(val)} validation samples at {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("train-info", "val-info", "scene-json", "can-root", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.train_info, args.val_info, args.scene_json, args.can_root, args.output)


if __name__ == "__main__":
    main()

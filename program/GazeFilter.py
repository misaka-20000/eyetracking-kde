"""
@file: GazeFilter.py
@desc: 设备参数及凝视点筛选
"""

import numpy as np


class Tracker:  # 眼动仪参数
    def __init__(self):
        self.sample_freq = 30  # 采样频率
        self.resolution = (1920, 1080)  # 分辨率
        self.monitor_size = (597.7, 336.2)  # 屏幕尺寸
        self.distance = 700.0  # z向距离


class IVTClassifier:
    def __init__(self, threshold: float = 30.0):
        self.tracker = Tracker()

        # 速度阈值
        threshold_px = self.tracker.distance * np.tan(np.radians(threshold * 0.001)) * self.tracker.resolution[0] / self.tracker.monitor_size[0]
        self.threshold = threshold_px

    @staticmethod
    def compute_velocity(x, y, timestamp):  # 计算速度
        x_velocity = np.append(0, np.abs(np.diff(x) / np.diff(timestamp)))
        y_velocity = np.append(0, np.abs(np.diff(y) / np.diff(timestamp)))
        velocity = np.sqrt((np.square(x_velocity)) + (np.square(y_velocity)))
        return x_velocity, y_velocity, velocity

    def predict(self, x, y, timestamps):  # 给出注视点的注释模式
        x = x * self.tracker.resolution[0]
        y = y * self.tracker.resolution[1]
        _, _, vel = self.compute_velocity(x, y, timestamps)
        fixations = [1 if v <= self.threshold else 0 for v in vel]
        return fixations


if __name__ == '__main__':
    classifier = IVTClassifier()
"""
@file: kde_model.py
@desc: 基于所有的被试数据计算kde，并计算设定阈值下的区域占比结果，保存到csv文件
"""

import numpy as np
import pandas as pd
from Subject import Subject
import os
import multiprocessing
from tqdm import tqdm
from scipy.stats import gaussian_kde
import time
import itertools


def load_processed_subject_data(subject_folder):
    subject_name = os.path.basename(subject_folder)
    subject_instance = Subject(subject_folder)  # 假设 Subject 类可以接受文件夹路径作为参数
    subject_instance.gaze_data = {}

    # 读取眼动数据
    for file in os.listdir(subject_folder):
        if file.startswith('gaze_') and file.endswith('.csv'):
            key = int(file.replace('gaze_', '').replace('.csv', ''))
            gaze_file_path = os.path.join(subject_folder, file)
            subject_instance.gaze_data[key] = pd.read_csv(gaze_file_path)

    return subject_instance


def load_all_subjects_data(base_path="../processed_data"):
    # 获取所有被试的姓名文件夹
    subject_folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if
                       os.path.isdir(os.path.join(base_path, f))]

    # 使用多进程加载所有被试数据
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # 通过 `tqdm` 显示进度条
        subjects = list(tqdm(pool.imap(load_processed_subject_data, subject_folders), total=len(subject_folders), desc="加载被试数据"))

    # 过滤掉加载失败的被试
    return [subject for subject in subjects if subject is not None]


def kde_one_frame(video_idx, frame_idx, subjects, threshold=0.68):
    # print("开始计算kde")
    pts = []
    for subject_instance in subjects:
        gaze_df = subject_instance.gaze_data[video_idx]
        # frame_data = gaze_df[gaze_df['frame'] == frame_idx]
        frame_data = gaze_df[(gaze_df['frame'] == frame_idx) & (gaze_df['fixation'] == 1)]  # 只考虑凝视点
        pts.append(frame_data[['x', 'y']].values)

    if pts:
        pts = np.vstack(pts)
    else:
        pts = np.array([])

    if pts.size <= 1:
        # print(f'视频{video_idx}，帧号{frame_idx}没有足够的数据')
        return -1  # 凝视点太少，无法计算kde

    kde_model = gaussian_kde(pts.T, bw_method='silverman')
    x_grid, y_grid = np.mgrid[0.:1:500j, 0.:1:500j]  # 这里的 (0., 1.) 可以根据数据范围调整
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])  # 网格坐标
    density_values = kde_model(grid_coords).reshape(x_grid.shape)  # 密度值
    density_values /= np.sum(density_values)  # 归一化

    sorted_indices = np.argsort(density_values)[::-1]  # 按密度值从大到小排序
    sorted_density = density_values[sorted_indices]
    cumulative_prob = np.cumsum(sorted_density)  # 累积概率

    threshold_index = np.searchsorted(cumulative_prob, 0.68)
    num_points = threshold_index + 1
    # (f"目标区域的点数: {num_points}")
    # 计算面积比
    area_ratio = num_points / 250000

    # print(f'视频{video_idx}，帧号{frame_idx},面积比:{area_ratio}')
    return area_ratio


def process_task(args):
    """任务处理函数，适配并行"""
    video_idx, frame_idx = args
    try:
        return (video_idx, frame_idx, kde_one_frame(video_idx, frame_idx, subjects))
    except:
        return (video_idx, frame_idx, -1)


def init_worker(subjects_):
    """多进程初始化，共享subjects数据"""
    global subjects
    subjects = subjects_

if __name__ == '__main__':
    subjects = load_all_subjects_data("../processed_data")
    max_video_idx = 7
    max_frame_idx = 2100
    results = {}

    # 生成所有(video_idx, frame_idx)组合
    tasks = list(itertools.product(range(1, max_video_idx + 1), range(1, max_frame_idx + 1)))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_worker, initargs=(subjects,)
    ) as pool:
        # 使用imap_unordered加速任务分发
        for result in tqdm(
                pool.imap_unordered(process_task, tasks),
                total=len(tasks),
                desc="计算KDE"
        ):
            video_idx, frame_idx, ratio = result
            if video_idx not in results:
                results[video_idx] = {}
            results[video_idx][frame_idx] = ratio

    data = []
    for video_idx, frames in results.items():
        for frame_idx, ratio in frames.items():
            data.append([video_idx, frame_idx, ratio])

    df = pd.DataFrame(data, columns=["video_idx", "frame_idx", "area_ratio"])

    # 保存到 CSV
    df.to_csv("kde_results.csv", index=False)

    print(subjects)

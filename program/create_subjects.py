"""
@file: datasource.py
@desc: 所有被试的眼动数据预处理以及保存
"""

import numpy as np
import pandas as pd
from Subject import Subject
import os
import multiprocessing
from tqdm import tqdm
import time


def load_subject_data(subject_folder):
    subject_instance = Subject(subject_folder)
    subject_instance.get_video_frames()
    subject_instance.get_gaze_data()
    subject_instance.add_fixation_type()
    return subject_instance


def load_all_subjects_data(base_path):
    # 获取所有被试的姓名文件夹
    subject_folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if
                       os.path.isdir(os.path.join(base_path, f))]

    # 使用多进程加载所有被试数据
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # 通过 `tqdm` 显示进度条
        subjects = list(tqdm(pool.imap(load_subject_data, subject_folders), total=len(subject_folders), desc="加载被试数据"))

    # 过滤掉加载失败的被试
    return [subject for subject in subjects if subject is not None]


def save_subject_data(subjects, save_path="../processed_data"):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for subject_instance in subjects:
        # 创建保存路径（使用被试的姓名或文件夹名）
        subject_name = os.path.basename(subject_instance.path)
        subject_folder = os.path.join(save_path, subject_name)
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)

        # 保存眼动数据
        for key, gaze_df in subject_instance.gaze_data.items():
            gaze_file_path = os.path.join(subject_folder, f'gaze_{key}.csv')
            gaze_df.to_csv(gaze_file_path, index=False)


if __name__ == '__main__':
    base_path = '../gaze_data'
    time1 = time.time()
    subjects = load_all_subjects_data(base_path)
    time2 = time.time()
    print(f'加载数据用时：{time2 - time1}秒')
    save_subject_data(subjects, "../processed_data")
    print(subjects)
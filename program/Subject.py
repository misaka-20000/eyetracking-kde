"""
@file: Subject.py
@desc: 处理一个被试的数据
"""

import pandas as pd
import numpy as np
import os
import GazeFilter

pd.options.mode.chained_assignment = None


class Subject:
    def __init__(self, path):
        self.path = path
        self.video_frames = {}
        self.gaze_data = {}
        self.video_num = 7

    def get_video_frames(self):
        for i in range(1, self.video_num + 1):  # 假设文件编号从1到video_num
            video_file = os.path.join(self.path, f'{i}_eye_video_timestamp.csv')
            if os.path.exists(video_file):
                try:
                    video_df = pd.read_csv(video_file, names=['timestamp', 'frame'])
                    video_df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
                    self.video_frames[i] = video_df
                except Exception as e:
                    print(f"无法读取文件 {video_file}: {e}")
            else:
                print(f"文件 {video_file} 不存在")

    def get_gaze_data(self):
        gaze_file = os.path.join(self.path, 'eye_track_data.csv')
        if os.path.exists(gaze_file):
            try:
                # 读取眼动数据
                gaze_df = pd.read_csv(gaze_file, skiprows=1, names=['timestamp', 'x', 'y'])
                gaze_st_time = pd.read_csv(gaze_file, header=None, nrows=1).iloc[0, 0]  # 单独读取开始的时间戳
                delta = gaze_st_time - gaze_df.iloc[0, 0]/1000
                gaze_df['timestamp'] = gaze_df['timestamp']/1000 + delta
                gaze_df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
                for key, df in self.video_frames.items():
                    t0 = df.iloc[0, 0]
                    t1 = df.iloc[-1, 0]
                    self.gaze_data[key] = gaze_df[(gaze_df['timestamp'] >= t0) & (gaze_df['timestamp'] <= t1)]

                    self.gaze_data[key]['frame'] = 1  # 默认值
                    total_frame = df['frame'].values.max()
                    for idx in range(total_frame):  # 为眼动数据匹配对应视频帧
                        if idx + 1 in df['frame'].values and idx + 2 in df['frame'].values:
                            t0 = df[df['frame'] == idx + 1]['timestamp'].values[0]
                            t1 = df[df['frame'] == idx + 2]['timestamp'].values[0]
                            self.gaze_data[key].loc[(self.gaze_data[key]['timestamp'] >= t0) & (self.gaze_data[key]['timestamp'] < t1), 'frame'] = idx + 1
                        elif idx + 1 == total_frame:
                            t0 = df[df['frame'] == idx + 1]['timestamp'].values[0]
                            self.gaze_data[key].loc[(self.gaze_data[key]['timestamp'] >= t0), 'frame'] = idx + 1
            except Exception as e:
                print(f"无法读取文件 {gaze_file}: {e}")
        else:
            print(f"文件 {gaze_file} 不存在")

    def add_fixation_type(self):
        ivt = GazeFilter.IVTClassifier()
        for key, df in self.gaze_data.items():
            fixation = ivt.predict(df['x'].values, df['y'].values, df['timestamp'].values)
            self.gaze_data[key]['fixation'] = fixation


if __name__ == '__main__':
    sub = Subject('../gaze_data/王卓')
    sub.get_video_frames()
    sub.get_gaze_data()
    sub.add_fixation_type()
    print(" ")
    print("   12   ")

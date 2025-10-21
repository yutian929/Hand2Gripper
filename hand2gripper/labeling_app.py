import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from typing import List, Tuple
from utils.visualize import vis_hand_2D_skeleton_contact
from utils.common import read_color_image, read_depth_image
from processor_config import DataManager
import mediapy
import tqdm

class LabelingApp:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def _process_single_sample(self, sample_id: int, color_image: np.ndarray, depth_image: np.ndarray):
        color = read_color_image(color_image)
        depth = read_depth_image(depth_image)
        bboxes = self.data_manager._read_bbox(sample_id)  # List[np.ndarray]
        is_right = self.data_manager._read_is_right(sample_id)  # List[bool]
        img_size = self.data_manager._read_img_size(sample_id)  # np.ndarray
        joints_2d = self.data_manager._read_joints_2d(sample_id)  # List[np.ndarray]
        contact_joint_out = self.data_manager._read_contact_joint_out(sample_id)  # List[np.ndarray]
        vertices_aligned = self.data_manager._read_vertices_aligned(sample_id)  # List[np.ndarray]
        joints = self.data_manager._read_joints(sample_id)  # List[np.ndarray]
        contact_out = self.data_manager._read_contact_out(sample_id)  # List[np.ndarray]
        assert len(bboxes) == len(is_right) == len(joints_2d) == len(contact_joint_out) == len(vertices_aligned) == len(joints) == len(contact_out), "Number of bboxes, is_right, joints_2d, contact_joint_out, vertices_aligned, joints, and contact_out must be the same"
        
        for hand_id in range(len(is_right)):
            bbox = bboxes[hand_id]
            is_right_hand = is_right[hand_id]
            joints_2d_hand = joints_2d[hand_id]
            contact_joint_out_hand = contact_joint_out[hand_id]
            selected_gripper_joints_seq = self._choose_gripper_joints_seq(sample_id, hand_id, color, bbox, is_right_hand, joints_2d_hand, contact_joint_out_hand)
            print(f"Selected gripper joints sequence for hand {hand_id}: {selected_gripper_joints_seq}")
            # TODO: Save the label results
            self._save_label_results(sample_id, hand_id, selected_gripper_joints_seq, vertices_aligned[hand_id], joints[hand_id], contact_out[hand_id], contact_joint_out[hand_id])
    
    def _choose_gripper_joints_seq(self, sample_id: int, hand_id: int, color: np.ndarray, bbox: np.ndarray, is_right_hand: bool, joints_2d_hand: np.ndarray, contact_joint_out_hand: np.ndarray) -> List[int]:
        """
        让用户通过文本框输入夹指关节点ID
        
        Args:
            color: 彩色图像
            bbox: 边界框 [x1, y1, x2, y2]
            is_right_hand: 是否为右手
            joints_2d_hand: 2D关节点坐标 (21, 2)
            contact_joint_out_hand: 接触关节点输出 (21,)
        Returns:
            选择的2个关节点ID列表
        """
        hand_2d_skeleton_image = vis_hand_2D_skeleton_contact(color, joints_2d_hand, bbox, is_right_hand, contact_joint_out_hand, eval_threshold=0.1)
        
        def draw_gripper_joints_seq(image: np.ndarray, joints_seq: List[int], joints_2d_hand: np.ndarray) -> np.ndarray:
            """在图像上绘制L和R标记"""
            left_joint_id, right_joint_id = joints_seq
            cv2.putText(image, f"L", (int(joints_2d_hand[left_joint_id][0]), int(joints_2d_hand[left_joint_id][1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, f"R", (int(joints_2d_hand[right_joint_id][0]), int(joints_2d_hand[right_joint_id][1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return image
        
        while True:
            cv2.imshow("gripper_annotated_image", hand_2d_skeleton_image)
            cv2.waitKey(100)
            
            try:
                user_input = input("Enter the gripper joints sequence (2 integers separated by blank): ")
                gripper_joints_seq = [int(x) for x in user_input.split(' ')]
                
                if len(gripper_joints_seq) != 2 or not all(0 <= joint_id <= 20 for joint_id in gripper_joints_seq):
                    print("Invalid input. Please enter 2 joint IDs between 0 and 20 (e.g., 1 2).")
                    continue
                
                # 显示标注后的图像
                annotated_image = draw_gripper_joints_seq(hand_2d_skeleton_image.copy(), gripper_joints_seq, joints_2d_hand)
                cv2.imshow("gripper_annotated_image", annotated_image)
                cv2.waitKey(100)
                
                # 确认选择（按Enter或输入y确认）
                confirm = input("Confirm the selection? (y/n or press Enter): ").lower().strip()
                if confirm in ['', 'y', 'yes']:
                    break
                print("Selection cancelled. Please try again.")
                
            except ValueError:
                print("Invalid input. Please enter 2 integers separated by blank (e.g., 1 2).")
        
        cv2.destroyAllWindows()
        return gripper_joints_seq

if __name__ == '__main__':
    data_manager = DataManager()
    labeling_app = LabelingApp(data_manager)
    video_path = "/home/yutian/projs/Hand2Gripper/hand2gripper/raw/0/video.mp4"
    depth_npy_path = "/home/yutian/projs/Hand2Gripper/hand2gripper/raw/0/depth.npy"
    video = mediapy.read_video(video_path)
    depth_npy = np.load(depth_npy_path)
    assert len(video) == len(depth_npy), "Number of frames in video and depth image must be the same"
    for idx in tqdm.tqdm(range(len(video)), desc="LabelingApp: Processing samples"):
        color_image = video[idx]
        depth_image = depth_npy[idx]  # meters
        labeling_app._process_single_sample(idx, color_image, depth_image)

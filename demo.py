import os
import cv2
import numpy as np
from hand2gripper.processor_config import DataManager, HandProcessorConfig, ContactProcessorConfig
from hand2gripper.hand_processor import HandProcessor
from hand2gripper.contact_processor import ContactProcessor
from hand2gripper.utils.visualize import vis_hand_2D_skeleton_contact, vis_choose_gripper
from hand2gripper.utils.common import read_color_image, read_depth_image
import matplotlib.pyplot as plt

class HandContactInference:
    def __init__(self, sample_id: int, color_image_path: str, depth_image_path: str):
        # 初始化 DataManager 和处理器配置
        self.sample_id = sample_id
        self.color_image_path = color_image_path
        self.depth_image_path = depth_image_path
        
        # 配置 HandProcessor 和 ContactProcessor
        self.data_manager = DataManager(sample_id)
        self.hand_processor_config = HandProcessorConfig(sample_id)
        self.contact_processor_config = ContactProcessorConfig(sample_id)
        
        # 初始化 HandProcessor 和 ContactProcessor 实例
        self.hand_processor = HandProcessor(self.hand_processor_config)
        self.contact_processor = ContactProcessor(self.contact_processor_config, self.data_manager)

    def run_inference(self):
        # 读取图像
        color_image = read_color_image(self.color_image_path)
        depth_image = np.zeros((color_image.shape[0], color_image.shape[1]))
        breakpoint()
        # 进行手部检测和重建
        hand_results = self.hand_processor._process_single_sample(self.sample_id, color_image, depth_image)
        
        # 进行接触点估计
        contact_results = self.contact_processor._process_single_sample(self.sample_id, color_image, depth_image)
        
        # 可视化结果
        self.visualize_results(hand_results, contact_results, color_image)

    def visualize_results(self, hand_results, contact_results, color_image):
        for idx, hand_data in enumerate(hand_results):
            # 提取 2D 关键点和手部信息
            joints_2d = hand_data['joints_2d']
            bbox = hand_data['bbox']
            is_right = hand_data['is_right']
            contact_joint_out = contact_results[idx]['raw_outputs']['contact_out']  # 假设 contact_out 包含接触点结果
            
            # 可视化 2D 骨架和接触点
            vis_image = vis_hand_2D_skeleton_contact(color_image, joints_2d, bbox, is_right, contact_joint_out)
            
            # 可视化选择的夹爪关节
            gripper_base_left_right_ids = [hand_data['gripper_base_joint_id'], hand_data['gripper_left_joint_id'], hand_data['gripper_right_joint_id']]
            vis_image = vis_choose_gripper(vis_image, joints_2d, gripper_base_left_right_ids)
            
            # 显示图像
            self.show_image(vis_image, idx)

    def show_image(self, vis_image, idx):
        """显示处理后的图像"""
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Sample {self.sample_id} - Hand {idx + 1}")
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # 示例：在给定样本上运行推理
    sample_id = "xxx"  # 修改为所需的样本 ID
    color_image_path = "/data/epic_kitchen/OpenDataLab___VISOR/raw/VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/rgb_frames/train/P01/P01_01/P01_01_frame_0000000298.jpg"  # 修改为颜色图像路径
    depth_image_path = "/path/to/depth_image.npy"  # 修改为深度图像路径

    # 创建 HandContactInference 实例并运行推理
    inference = HandContactInference(sample_id, color_image_path, depth_image_path)
    inference.run_inference()

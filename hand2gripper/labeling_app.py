import cv2
import os
import numpy as np
from typing import List, Tuple
from utils.visualize import vis_hand_2D_skeleton_contact
from utils.common import read_color_image, read_depth_image, _to_numpy
from processor_config import DataManager, LabelingAppConfig
import mediapy
import tqdm

class LabelingApp:
    def __init__(self, labeling_app_config: LabelingAppConfig, data_manager: DataManager):
        self.labeling_app_config = labeling_app_config
        self.labeling_app_results_dir = self.labeling_app_config.labeling_app_results_dir
        self.labeling_app_color_image_dir = self.labeling_app_config.labeling_app_color_image_dir
        self.labeling_app_depth_npy_dir = self.labeling_app_config.labeling_app_depth_npy_dir
        self.data_manager = data_manager
        self.last_chosen_gripper_joints_seq = [0, 0, 0]

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
            selected_gripper_joints_seq = self._choose_gripper_joints_seq(sample_id, hand_id, color, bboxes[hand_id], is_right[hand_id], joints_2d[hand_id], contact_joint_out[hand_id])
            # selected_gripper_joints_seq = [0, 4, 4] if is_right[hand_id] == 1 else [0 , 4, 15]
            print(f"Selected gripper joints sequence for hand {hand_id}: {selected_gripper_joints_seq}")
            self._save_label_results(sample_id, hand_id, bboxes[hand_id],is_right[hand_id], vertices_aligned[hand_id], joints[hand_id], contact_out[hand_id], contact_joint_out[hand_id], selected_gripper_joints_seq, self.labeling_app_results_dir)
            self._save_color_image(sample_id, hand_id, color, self.labeling_app_color_image_dir)
            self._save_depth_npy(sample_id, hand_id, depth, self.labeling_app_depth_npy_dir)
    
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
            """在图像上绘制Base, Left, Right标记"""
            base_joint_id, left_joint_id, right_joint_id = joints_seq
            cv2.putText(image, f"B", (int(joints_2d_hand[base_joint_id][0]), int(joints_2d_hand[base_joint_id][1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, f"L", (int(joints_2d_hand[left_joint_id][0]), int(joints_2d_hand[left_joint_id][1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, f"R", (int(joints_2d_hand[right_joint_id][0]), int(joints_2d_hand[right_joint_id][1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return image
        
        while True:
            cv2.imshow(f"gripper_annotated_image {sample_id}_{hand_id}", hand_2d_skeleton_image)
            cv2.waitKey(100)
            
            try:
                user_input = input(f"Enter the gripper joints sequence (3 integers separated by blank, last chosen(Base, Left, Right): {self.last_chosen_gripper_joints_seq}): ")
                if user_input == '':
                    gripper_joints_seq = self.last_chosen_gripper_joints_seq
                else:
                    gripper_joints_seq = [int(x) for x in user_input.split(' ')]
                
                if len(gripper_joints_seq) != 3 or not all(0 <= joint_id <= 20 for joint_id in gripper_joints_seq):
                    print("Invalid input. Please enter 3 joint IDs (base, left, right) between 0 and 20 (e.g., 1 2 3).")
                    continue
                
                annotated_image = draw_gripper_joints_seq(hand_2d_skeleton_image.copy(), gripper_joints_seq, joints_2d_hand)
                cv2.imshow(f"gripper_annotated_image {sample_id}_{hand_id}", annotated_image)
                cv2.waitKey(100)
                
                confirm = input("Confirm the selection? (y/n or press Enter): ").lower().strip()
                if confirm in ['', 'y', 'yes']:
                    self.last_chosen_gripper_joints_seq = gripper_joints_seq
                    break
                print("Selection cancelled. Please try again.")
                
            except ValueError:
                print("Invalid input. Please enter 3 integers (base, left, right) separated by blank (e.g., 1 2 3).")
        
        cv2.destroyAllWindows()
        return gripper_joints_seq
    
    def _save_label_results(self, sample_id: int, hand_id: int, bbox: np.ndarray, is_right_hand: bool, vertices_aligned: np.ndarray, joints: np.ndarray, contact_out: np.ndarray, contact_joint_out: np.ndarray, selected_gripper_joints_seq: List[int], save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        data = {
            'bbox': _to_numpy(bbox).astype(np.int32),
            'is_right_hand': _to_numpy(is_right_hand),
            'vertices_aligned': _to_numpy(vertices_aligned),
            'joints': _to_numpy(joints),
            'contact_out': _to_numpy(contact_out),
            'contact_joint_out': _to_numpy(contact_joint_out),
            'gripper_left_joint_id': _to_numpy(selected_gripper_joints_seq[1]),
            'gripper_right_joint_id': _to_numpy(selected_gripper_joints_seq[2]),
            'gripper_base_joint_id': _to_numpy(selected_gripper_joints_seq[0]),
        }
        out_path = os.path.join(save_dir, f"{sample_id}_{hand_id}.npz")
        np.savez_compressed(out_path, **data)
    
    def _save_color_image(self, sample_id: int, hand_id: int, color: np.ndarray, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{sample_id}_{hand_id}.png")
        cv2.imwrite(out_path, color)
    
    def _save_depth_npy(self, sample_id: int, hand_id: int, depth: np.ndarray, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{sample_id}_{hand_id}.npy")
        np.save(out_path, depth)


if __name__ == '__main__':
    raw_samples_root_dir = "/home/yutian/projs/Hand2Gripper/hand2gripper/raw"
    for samples_id in os.listdir(raw_samples_root_dir):
        # if samples_id != '10':
        #     continue
        if os.path.isdir(os.path.join(raw_samples_root_dir, samples_id)):
            labeling_app_config = LabelingAppConfig(samples_id)
            data_manager = DataManager(samples_id)
            labeling_app = LabelingApp(labeling_app_config, data_manager)
            video_path = os.path.join(raw_samples_root_dir, samples_id, "video.mp4")
            depth_npy_path = os.path.join(raw_samples_root_dir, samples_id, "depth.npy")
            video = mediapy.read_video(video_path)
            depth_npy = np.load(depth_npy_path)  # meters
            assert len(video) == len(depth_npy), "Number of frames in video and depth image must be the same"
            for idx in tqdm.tqdm(range(len(video)), desc=f"Labeling App: Processing samples {os.path.join(raw_samples_root_dir, samples_id)}"):
                if len(data_manager._read_labeling_app_results(idx)) > 0:
                    continue
                color_image = video[idx]
                depth_image = depth_npy[idx]  # meters
                labeling_app._process_single_sample(idx, color_image, depth_image)
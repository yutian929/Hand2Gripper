import os
import numpy as np
import torch
import cv2
import mediapy
import tqdm
from typing import Union
from models.model import Hand2GripperModel
from processor_config import Hand2GripperProcessorConfig, DataManager
from utils.common import read_color_image, read_depth_image, _to_numpy
from utils.visualize import vis_choose_gripper

class Hand2GripperProcessor:
    def __init__(self, hand2gripper_processor_config: Hand2GripperProcessorConfig, data_manager: DataManager):
        # config
        self.hand2gripper_processor_config = hand2gripper_processor_config
        self.data_manager = data_manager
        self.output_dir = self.hand2gripper_processor_config.processor_output_dir
        # model
        self.model = Hand2GripperModel(d_model=self.hand2gripper_processor_config.d_model, img_size=self.hand2gripper_processor_config.img_size)
        self.model._load_checkpoint(self.hand2gripper_processor_config.model_path)
        self.model.eval()
        # vis
        self.vis_gripper_base_left_right = self.hand2gripper_processor_config.vis_gripper_base_left_right
        # save
        self.vis_gripper_base_left_right_images_dir = self.hand2gripper_processor_config.vis_gripper_base_left_right_images_dir
        self.hand2gripper_processor_results_dir = self.hand2gripper_processor_config.hand2gripper_processor_results_dir

    def _process_single_sample(self, sample_id: int, color_image: Union[str, np.ndarray], depth_image: Union[str, np.ndarray]) -> np.ndarray:
        # read image
        color = read_color_image(color_image)
        depth = read_depth_image(depth_image)
        # read other results
        bboxes = self.data_manager._read_bbox(sample_id)
        is_right = self.data_manager._read_is_right(sample_id)
        joints = self.data_manager._read_joints(sample_id)
        contact_joint_out = self.data_manager._read_contact_joint_out(sample_id)
        joints_2d = self.data_manager._read_joints_2d(sample_id)
        # inference
        main_results = []
        for hand_id in range(len(bboxes)):
            out = self.inference(color, bboxes[hand_id], joints[hand_id], contact_joint_out[hand_id], is_right[hand_id])
            # vis
            if self.vis_gripper_base_left_right:
                self._vis_gripper_base_left_right_images(sample_id, hand_id, color, joints_2d[hand_id], out, self.vis_gripper_base_left_right_images_dir)
            # save
            infer_results = self._save_infer_results(sample_id, hand_id, out, self.hand2gripper_processor_results_dir)
            main_results.append(infer_results)
        return main_results
    
    def inference(self, color: np.ndarray, bbox: np.ndarray, joints: np.ndarray, contact_joint_out: np.ndarray, is_right: bool):
        # preprocess
        color_tensor = self.model._read_color(color)
        bbox_tensor = self.model._read_bbox(bbox)
        joints_tensor = self.model._read_keypoints_3d(joints)
        contact_joint_out_tensor = self.model._read_contact(contact_joint_out)
        is_right_tensor = self.model._read_is_right(is_right)
        crop_img = self.model._crop_and_resize(color_tensor, bbox_tensor)
        with torch.no_grad():
            out = self.model.forward(crop_img, joints_tensor, contact_joint_out_tensor, is_right_tensor)
        return out

    def _vis_gripper_base_left_right_images(self, sample_id: int, hand_id: int, color: np.ndarray, joints_2d: np.ndarray, out: dict, save_dir: str):
        vis_image = vis_choose_gripper(color, joints_2d, out['pred_triple'].cpu().numpy()[0])
        cv2.imwrite(os.path.join(save_dir, f"{sample_id}_{hand_id}.png"), vis_image)

    def _save_infer_results(self, sample_id: int, hand_id: int, out: dict, save_dir: str):
        data = {
            'pred_triple': _to_numpy(out['pred_triple']),
            'logits_base': _to_numpy(out['logits_base']),
            'logits_left': _to_numpy(out['logits_left']),
            'logits_right': _to_numpy(out['logits_right']),
            'S_bl': _to_numpy(out['S_bl']),
            'S_br': _to_numpy(out['S_br']),
            'S_lr': _to_numpy(out['S_lr']),
        }
        out_path = os.path.join(save_dir, f"{sample_id}_{hand_id}.npz")
        np.savez_compressed(out_path, **data)
        return data
if __name__ == "__main__":
    raw_samples_root_dir = "/home/yutian/projs/Hand2Gripper/hand2gripper/raw"
    for samples_id in os.listdir(raw_samples_root_dir):
        if os.path.isdir(os.path.join(raw_samples_root_dir, samples_id)):
            hand2gripper_processor_config = Hand2GripperProcessorConfig(samples_id)
            data_manager = DataManager(samples_id)
            hand2gripper_processor = Hand2GripperProcessor(hand2gripper_processor_config, data_manager)
            video_path = os.path.join(raw_samples_root_dir, samples_id, "video.mp4")
            depth_npy_path = os.path.join(raw_samples_root_dir, samples_id, "depth.npy")
            video = mediapy.read_video(video_path)
            depth_npy = np.load(depth_npy_path)  # meters
            assert len(video) == len(depth_npy), "Number of frames in video and depth image must be the same"
            for idx in tqdm.tqdm(range(len(video)), desc=f"Contact Processor: Processing samples {os.path.join(raw_samples_root_dir, samples_id)}"):
                if len(data_manager._read_hand2gripper_processor_results(idx)) > 0:
                    continue
                color_image = video[idx]
                depth_image = depth_npy[idx]  # meters
                hand2gripper_processor._process_single_sample(idx, color_image, depth_image)
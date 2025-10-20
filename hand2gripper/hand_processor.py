import os
import cv2
import numpy as np
import mediapy
import tqdm
from pathlib import Path
from typing import Union, List, Tuple

from hand2gripper_wilor import HandDetector, WiLoRModel, HandRenderer
from processor_config import HandProcessorConfig
from utils.common import read_color_image, read_depth_image, _to_numpy
from utils.visualize import vis_hand_mesh, vis_hand_2D_skeleton

class HandProcessor:
    def __init__(self, hand_processor_config: HandProcessorConfig):
        # config
        self.hand_processor_config = hand_processor_config
        self.output_dir = self.hand_processor_config.processor_output_dir
        # hand detector
        self.hand_detector = HandDetector(model_path=self.hand_processor_config.hand_detector_model_path, device=self.hand_processor_config.device)
        self.hand_detector_conf_threshold = self.hand_processor_config.hand_detector_conf_threshold
        self.hand_detector_iou_threshold = self.hand_processor_config.hand_detector_iou_threshold
        # wilor model
        self.wilor_model = WiLoRModel(model_path=self.hand_processor_config.wilor_model_path, config_path=self.hand_processor_config.wilor_model_config, device=self.hand_processor_config.device)
        self.wilor_model_rescale_factor = self.hand_processor_config.wilor_model_rescale_factor
        # hand renderer
        self.hand_renderer = HandRenderer(model_cfg=self.wilor_model.model_cfg, faces=self.wilor_model.model.mano.faces)
        # vis
        self.vis_hand_2D_skeleton = self.hand_processor_config.vis_hand_2D_skeleton
        self.vis_hand_mesh = self.hand_processor_config.vis_hand_mesh
        # save dirs
        self.vis_hand_2D_skeleton_images_dir = self.hand_processor_config.vis_hand_2D_skeleton_images_dir
        self.vis_hand_mesh_images_dir = self.hand_processor_config.vis_hand_mesh_images_dir
        self.hand_processor_results_dir = self.hand_processor_config.hand_processor_results_dir

    def _process_single_sample(self, sample_id: int, color_image: Union[str, np.ndarray], depth_image: Union[str, np.ndarray]) -> np.ndarray:
        # read image
        color = read_color_image(color_image)
        depth = read_depth_image(depth_image)
        
        # detect hand
        ## bboxes: List of bounding boxes [x1, y1, x2, y2]
        ## is_right: List of boolean values indicating right hand (True) or left hand (False)
        bboxes, is_right = self.hand_detector.detect_hands(image=color, conf_threshold=self.hand_detector_conf_threshold, iou_threshold=self.hand_detector_iou_threshold)
        if len(bboxes) == 0:
            return None
        
        # reconstruct hand
        ## reconstruction_results: Dictionary containing reconstruction results {'vertices': np.ndarray, 'joints': np.ndarray, 'cam_t': np.ndarray, 'is_right': np.ndarray, 'kpts_2d': np.ndarray}
        ## vertices: (778, 3), joints: (21, 3), cam_t: (3,), is_right: (1,), kpts_2d: Tensor(778, 2)
        ## joints_2d: (21, 2)
        reconstruction_results = self.wilor_model.reconstruct_hands(image=color, bboxes=bboxes, is_right=is_right, rescale_factor=self.wilor_model_rescale_factor)
        if len(reconstruction_results['vertices']) == 0:
            return None
        assert len(bboxes) == len(reconstruction_results['vertices'])
        reconstruction_results['joints_2d'] = self._generate_joints_2D(reconstruction_results)

        # render hand mesh
        reconstruction_results['hand_mesh'] = self._render_hand_mesh(reconstruction_results)

        # align hand mesh with depth image
        ## TODO: Implement ICP alignment
        reconstruction_results['vertices_aligned'] = self._depth_icp_alignment(depth, reconstruction_results)

        # visualize
        if any([self.vis_hand_2D_skeleton, self.vis_hand_mesh]):
            if self.vis_hand_2D_skeleton:
                self._vis_hand_2D_skeleton_images(sample_id, color, bboxes, is_right, reconstruction_results, self.vis_hand_2D_skeleton_images_dir)
            if self.vis_hand_mesh:
                self._vis_hand_mesh_images(sample_id, color, bboxes, is_right, reconstruction_results, self.vis_hand_mesh_images_dir)
        
        # save results
        self._save_results(sample_id, bboxes, is_right, reconstruction_results, self.hand_processor_results_dir)

    def _vis_hand_2D_skeleton_images(self, sample_id: int, color: np.ndarray, bboxes: List[List[int]], is_right: List[bool], reconstruction_results: dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for idx in range(len(reconstruction_results['vertices'])):
            joints_2d = reconstruction_results['joints_2d'][idx]
            bbox = bboxes[idx]
            is_right_hand = is_right[idx]
            vis_image = vis_hand_2D_skeleton(color, joints_2d, bbox, is_right_hand)
            cv2.imwrite(os.path.join(save_dir, f"{sample_id}_{idx}.png"), vis_image)
    
    def _vis_hand_mesh_images(self, sample_id: int, color: np.ndarray, bboxes: List[List[int]], is_right: List[bool], reconstruction_results: dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for idx in range(len(reconstruction_results['vertices'])):
            hand_mesh = reconstruction_results['hand_mesh'][idx]
            bbox = bboxes[idx]
            is_right_hand = is_right[idx]
            vis_image = vis_hand_mesh(color, hand_mesh, bbox, is_right_hand)
            cv2.imwrite(os.path.join(save_dir, f"{sample_id}_{idx}.png"), vis_image)
    
    def _generate_joints_2D(self, reconstruction_results: dict) -> List[np.ndarray]:
        hand_2D_keypoints = []
        for idx in range(len(reconstruction_results['vertices'])):
            joints_3d = reconstruction_results['joints'][idx]  # (21, 3)
            cam_t = reconstruction_results['cam_t'][idx]  # (3,)
            focal_length = reconstruction_results['focal_length']
            img_size = reconstruction_results['img_size']
            joints_2d = self.wilor_model._project_full_img(
                points=joints_3d, 
                cam_trans=cam_t, 
                focal_length=focal_length, 
                img_res=img_size
            )  # (21, 2)
            hand_2D_keypoints.append(joints_2d)
        return hand_2D_keypoints

    def _render_hand_mesh(self, reconstruction_results: dict) -> List[np.ndarray]:
        rendered_meshes = []
        
        for idx in range(len(reconstruction_results['vertices'])):
            vertices = reconstruction_results['vertices'][idx]  # (778, 3)
            cam_t = reconstruction_results['cam_t'][idx]  # (3,)
            is_right = reconstruction_results['is_right'][idx]
            focal_length = reconstruction_results['focal_length']
            img_size = reconstruction_results['img_size']
            
            cam_view = self.hand_renderer.render_hands(
                vertices_list=[vertices],
                cam_t_list=[cam_t],
                is_right_list=[is_right],
                img_size=img_size,
                focal_length=focal_length
            )
            
            rendered_meshes.append(cam_view)
        
        return rendered_meshes
    
    def _depth_icp_alignment(self, depth: np.ndarray, reconstruction_results: dict) -> List[np.ndarray]:
        return reconstruction_results['vertices']

    def _save_results(self, sample_id: int, bboxes: List[List[int]], is_right: List[bool], reconstruction_results: dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        assert len(bboxes) == len(is_right) == len(reconstruction_results['vertices']), "Number of bboxes, is_right, and vertices must be the same"
        for idx in range(len(reconstruction_results['vertices'])):
            data = {
                'bbox': _to_numpy(bboxes[idx]).astype(np.int32),                                # (4,)
                'is_right': is_right[idx],                                                      # bool
                'vertices': _to_numpy(reconstruction_results['vertices'][idx]),                 # (778, 3)
                'vertices_aligned': _to_numpy(reconstruction_results['vertices_aligned'][idx]), # (778, 3)
                'joints': _to_numpy(reconstruction_results['joints'][idx]),                     # (21, 3)
                'cam_t': _to_numpy(reconstruction_results['cam_t'][idx]),                       # (3,)
                'kpts_2d': _to_numpy(reconstruction_results['kpts_2d'][idx]),                   # (778, 2)
                'joints_2d': _to_numpy(reconstruction_results['joints_2d'][idx]),               # (21, 2)
                'img_size': _to_numpy(reconstruction_results.get('img_size'))                   # (W,H)
            }
            out_path = os.path.join(save_dir, f"{sample_id}_{idx}.npz")
            np.savez_compressed(out_path, **data)


if __name__ == '__main__':
    hand_processor_config = HandProcessorConfig()
    hand_processor = HandProcessor(hand_processor_config)
    video_path = "/home/yutian/projs/Hand2Gripper/hand2gripper/raw/0/video.mp4"
    depth_npy_path = "/home/yutian/projs/Hand2Gripper/hand2gripper/raw/0/depth.npy"
    video = mediapy.read_video(video_path)
    depth_npy = np.load(depth_npy_path)
    assert len(video) == len(depth_npy), "Number of frames in video and depth image must be the same"
    for idx in tqdm.tqdm(range(len(video)), desc="Hand Processor: Processing samples"):
        color_image = video[idx]
        depth_image = depth_npy[idx]  # meters
        hand_processor._process_single_sample(idx, color_image, depth_image)
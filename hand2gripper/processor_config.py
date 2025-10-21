import os
from typing import List
import numpy as np

# Base Config
ROOT_DIR = os.path.dirname(__file__)
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
class BaseConfig:
  def __init__(self):
    self.base_output_dir = BASE_OUTPUT_DIR




# Hand Processor Config
HAND_PROCESSOR_MODEL_DIR = os.path.join(ROOT_DIR, 'submodules', 'Hand2Gripper_WiLoR', 'hand2gripper_wilor', 'pretrained_models')
HAND_PROCESSOR_MANO_DATA_DIR = os.path.join(ROOT_DIR, 'submodules', 'Hand2Gripper_WiLoR', 'hand2gripper_wilor', 'mano_data')
HAND_PROCESSOR_HAND_DETECTOR_IOU_THRESHOLD = 0.5
HAND_PROCESSOR_HAND_DETECTOR_CONF_THRESHOLD = 0.3
HAND_PROCESSOR_WILOR_MODEL_RESCALE_FACTOR = 2.0
HAND_PROCESSOR_DEVICE = 'auto'
HAND_PROCESSOR_VIS_HAND_2D_SKELETON = False
HAND_PROCESSOR_VIS_HAND_MESH = False

class HandProcessorConfig(BaseConfig):
  def __init__(self):
    super().__init__()
    # Base Config
    self.processor_output_dir = os.path.join(self.base_output_dir, 'hand_processor')
    self.model_dir = HAND_PROCESSOR_MODEL_DIR
    self.device = HAND_PROCESSOR_DEVICE
    # hand detector
    self.hand_detector_model_path = os.path.join(self.model_dir, 'detector.pt')
    self.hand_detector_conf_threshold = HAND_PROCESSOR_HAND_DETECTOR_CONF_THRESHOLD
    self.hand_detector_iou_threshold = HAND_PROCESSOR_HAND_DETECTOR_IOU_THRESHOLD
    # wilor model
    self.wilor_model_path = os.path.join(self.model_dir, 'wilor_final.ckpt')
    self.wilor_model_config = os.path.join(self.model_dir, 'model_config.yaml')
    self.wilor_model_rescale_factor = HAND_PROCESSOR_WILOR_MODEL_RESCALE_FACTOR
    # hand renderer
    self.mano_data_dir = HAND_PROCESSOR_MANO_DATA_DIR
    # vis
    self.vis_hand_2D_skeleton = HAND_PROCESSOR_VIS_HAND_2D_SKELETON
    self.vis_hand_mesh = HAND_PROCESSOR_VIS_HAND_MESH
    # save
    self.vis_hand_2D_skeleton_images_dir = os.path.join(self.processor_output_dir, "vis_hand_2D_skeleton_images")
    self.vis_hand_mesh_images_dir = os.path.join(self.processor_output_dir, "vis_hand_mesh_images")
    self.hand_processor_results_dir = os.path.join(self.processor_output_dir, "hand_processor_results")







# Contact Processor Config
CONTACT_PROCESSOR_BACKBONE = 'hamer'
CONTACT_PROCESSOR_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'submodules', 'Hand2Gripper_HACO', 'base_data', 'release_checkpoint', 'haco_final_hamer_checkpoint.ckpt')
CONTACT_PROCESSOR_VIS_CONTACT_RENDERED = True
CONTACT_PROCESSOR_VIS_CROP_IMG = True

class ContactProcessorConfig(BaseConfig):
  def __init__(self):
    super().__init__()
    # Base Config
    self.processor_output_dir = os.path.join(self.base_output_dir, 'contact_processor')
    # contact estimator
    self.backbone = CONTACT_PROCESSOR_BACKBONE
    self.checkpoint_path = CONTACT_PROCESSOR_CHECKPOINT_PATH
    self.log_dir = self.processor_output_dir
    # vis
    self.vis_contact_rendered = CONTACT_PROCESSOR_VIS_CONTACT_RENDERED
    self.vis_crop_img = CONTACT_PROCESSOR_VIS_CROP_IMG
    # save
    self.vis_contact_rendered_images_dir = os.path.join(self.processor_output_dir, "vis_contact_rendered_images")
    self.vis_crop_img_images_dir = os.path.join(self.processor_output_dir, "vis_crop_img_images")
    self.contact_processor_results_dir = os.path.join(self.processor_output_dir, "contact_processor_results")





class DataManager:
    def __init__(self):
        pass
    
    # read hand processor results
    def _read_bbox(self, sample_id: int) -> List[np.ndarray]:
        hand_processor_config = HandProcessorConfig()
        hand_processor_results_dir = hand_processor_config.hand_processor_results_dir
        bboxes = []
        for file in os.listdir(hand_processor_results_dir):
            if file.startswith(f"{sample_id}_"):
                bbox = np.load(os.path.join(hand_processor_results_dir, file))['bbox']
                bboxes.append(bbox)
        return bboxes
    
    def _read_is_right(self, sample_id: int) -> List[bool]:
        hand_processor_config = HandProcessorConfig()
        hand_processor_results_dir = hand_processor_config.hand_processor_results_dir
        is_right = []
        for file in os.listdir(hand_processor_results_dir):
            if file.startswith(f"{sample_id}_"):
                is_right.append(np.load(os.path.join(hand_processor_results_dir, file))['is_right'])
        return is_right
    
    def _read_img_size(self, sample_id: int) -> np.ndarray:
        hand_processor_config = HandProcessorConfig()
        hand_processor_results_dir = hand_processor_config.hand_processor_results_dir
        for file in os.listdir(hand_processor_results_dir):
            if file.startswith(f"{sample_id}_"):
                img_size = np.load(os.path.join(hand_processor_results_dir, file))['img_size']
                return img_size
        return None
    
    def _read_joints(self, sample_id: int) -> List[np.ndarray]:
        hand_processor_config = HandProcessorConfig()
        hand_processor_results_dir = hand_processor_config.hand_processor_results_dir
        joints = []
        for file in os.listdir(hand_processor_results_dir):
            if file.startswith(f"{sample_id}_"):
                joints.append(np.load(os.path.join(hand_processor_results_dir, file))['joints'])
        return joints
    
    def _read_joints_2d(self, sample_id: int) -> List[np.ndarray]:
        hand_processor_config = HandProcessorConfig()
        hand_processor_results_dir = hand_processor_config.hand_processor_results_dir
        joints_2d = []
        for file in os.listdir(hand_processor_results_dir):
            if file.startswith(f"{sample_id}_"):
                joints_2d.append(np.load(os.path.join(hand_processor_results_dir, file))['joints_2d'])
        return joints_2d
    
    def _read_vertices(self, sample_id: int) -> List[np.ndarray]:
        hand_processor_config = HandProcessorConfig()
        hand_processor_results_dir = hand_processor_config.hand_processor_results_dir
        vertices = []
        for file in os.listdir(hand_processor_results_dir):
            if file.startswith(f"{sample_id}_"):
                vertices.append(np.load(os.path.join(hand_processor_results_dir, file))['vertices'])
        return vertices
    
    def _read_vertices_aligned(self, sample_id: int) -> List[np.ndarray]:
        hand_processor_config = HandProcessorConfig()
        hand_processor_results_dir = hand_processor_config.hand_processor_results_dir
        vertices_aligned = []
        for file in os.listdir(hand_processor_results_dir):
            if file.startswith(f"{sample_id}_"):
                vertices_aligned.append(np.load(os.path.join(hand_processor_results_dir, file))['vertices_aligned'])
        return vertices_aligned
    
    # read contact processor results
    def _read_contact_joint_out(self, sample_id: int) -> List[np.ndarray]:
        contact_processor_config = ContactProcessorConfig()
        contact_processor_results_dir = contact_processor_config.contact_processor_results_dir
        contact_joint_out = []
        for file in os.listdir(contact_processor_results_dir):
            if file.startswith(f"{sample_id}_"):
                contact_joint_out.append(np.load(os.path.join(contact_processor_results_dir, file))['contact_joint_out'])
        return contact_joint_out
    
    def _read_contact_out(self, sample_id: int) -> List[np.ndarray]:
        contact_processor_config = ContactProcessorConfig()
        contact_processor_results_dir = contact_processor_config.contact_processor_results_dir
        contact_out = []
        for file in os.listdir(contact_processor_results_dir):
            if file.startswith(f"{sample_id}_"):
                contact_out.append(np.load(os.path.join(contact_processor_results_dir, file))['contact_out'])
        return contact_out
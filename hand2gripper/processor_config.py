import os

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
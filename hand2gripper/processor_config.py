import os

# Base Config
ROOT_DIR = os.path.dirname(__file__)
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
# Hand Processor Config
HAND_PROCESSOR_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'hand_processor')
HAND_PROCESSOR_MODEL_DIR = os.path.join(ROOT_DIR, 'submodules', 'Hand2Gripper_WiLoR', 'hand2gripper_wilor', 'pretrained_models')
HAND_PROCESSOR_MANO_DATA_DIR = os.path.join(ROOT_DIR, 'submodules', 'Hand2Gripper_WiLoR', 'hand2gripper_wilor', 'mano_data')
HAND_PROCESSOR_HAND_DETECTOR_IOU_THRESHOLD = 0.5
HAND_PROCESSOR_HAND_DETECTOR_CONF_THRESHOLD = 0.3
HAND_PROCESSOR_WILOR_MODEL_RESCALE_FACTOR = 2.0
HAND_PROCESSOR_DEVICE = 'auto'
HAND_PROCESSOR_VIS_HAND_2D_SKELETON = True
HAND_PROCESSOR_VIS_HAND_MESH = True

class BaseConfig:
  def __init__(self):
    self.base_output_dir = BASE_OUTPUT_DIR
  
class HandProcessorConfig(BaseConfig):
  def __init__(self):
    super().__init__()
    # Base Config
    self.processor_output_dir = HAND_PROCESSOR_OUTPUT_DIR
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
    self.reconstruction_results_dir = os.path.join(self.processor_output_dir, "reconstruction_results")

CONTACT_PROCESSOR_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'contact_processor')


class ContactProcessorConfig(BaseConfig):
  def __init__(self):
    super().__init__()
    # Base Config
    self.processor_output_dir = CONTACT_PROCESSOR_OUTPUT_DIR
    
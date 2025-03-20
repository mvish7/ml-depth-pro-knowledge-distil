"""
Connects trainer and losses by applying the appropriate loss functions
to the outputs of the teacher and student models.
"""
from typing import Dict

from src.core.losses.losses import DepthSupervision, FoVSupervision, FeatureDistillation
import torch


class LossCalculator:
    # todo: make sure loss configs are being used properly -- i.e. active/inactive losses, loss scaling factors
    def __init__(self, depth_supervisor: DepthSupervision, fov_supervisor: FoVSupervision,
                 kd_supervisor: FeatureDistillation):
        self.depth_supervision = depth_supervisor
        self.fov_supervision = fov_supervisor
        self.knowledge_distil = kd_supervisor

    def calculate_losses(self, student_output: Dict,
                         teacher_output: Dict, valid_mask: torch.Tensor = None) -> Dict:
        """
        Calculate the losses for depth and field of view (FoV).

        Args:
            student_output: Output from the student model.
            teacher_output: Output from the teacher model.
            valid_mask:

        Returns:
            dict: A dictionary containing the calculated losses.
        """
        losses = {}

        # Calculate depth loss
        depth_loss = self.depth_supervision( student_output['depth'], teacher_output['depth'], mask=valid_mask)
        losses['depth_loss'] = depth_loss

        # Calculate FoV loss
        fov_loss = self.fov_supervision(teacher_output['fov'], student_output['fov'])
        losses['fov_loss'] = fov_loss

        # Calculate Knowledge distillation loss
        losses["kd_encoder_x0"] = self.knowledge_distil(student_output["projected_features"]["x0"],
                                              teacher_output["intermediate_features"]["x0"])
        losses["kd_encoder_x1"] = self.knowledge_distil(student_output["projected_features"]["x0"],
                                              teacher_output["intermediate_features"]["x0"])
        losses["kd_encoder_xglobal"] = self.knowledge_distil(student_output["projected_features"]["x_global"],
                                              teacher_output["intermediate_features"]["x_global"])
        losses["kd_decoder_features"] = self.knowledge_distil(student_output["projected_features"]["decoder_features"],
                                              teacher_output["intermediate_features"]["decoder_features"])
        losses["kd_decoder_lowres"] = self.knowledge_distil(student_output["projected_features"]["decoder_lowres"],
                                              teacher_output["intermediate_features"]["decoder_lowres"], grad_loss_only = True)
        losses["kd_head"] = self.knowledge_distil(student_output["projected_features"]["head_intermediate"],
                                              teacher_output["intermediate_features"]["head_intermediate"])
        losses["kd_fov"] = self.knowledge_distil(student_output["projected_features"]["head_intermediate"],
                                              teacher_output["intermediate_features"]["head_intermediate"], cs_loss_only=True)

        return losses


# Example usage
if __name__ == "__main__":
    # Dummy data for testing
    student_op = {
        'depth': torch.rand(4, 1, 64, 64),
        'fov': torch.rand(4),
        'intermediate_features': {
            'fov_intermediate': torch.rand(4, 128, 16, 16)
        }
    }
    teacher_op = {
        'depth': torch.rand(4, 1, 64, 64),
        'fov': torch.rand(4),
        'intermediate_features': {
            'fov_intermediate': torch.rand(4, 128, 16, 16)
        }
    }

    loss_calculator = LossCalculator()
    losses = loss_calculator.calculate_losses(student_op, teacher_op)
    print("Calculated Losses:", losses)

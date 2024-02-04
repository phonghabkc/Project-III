import os
import torch
import numpy as np
from Actionsrecognition.Utils import Graph
from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
from pose_utils import normalize_points_with_size, scale_pose


class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 weight_file='./Models/TSSTG/tsstg-model.pth',
                 device='cpu'):
        self.graph_args = {'strategy': 'spatial'}
        self.class_names = ['Jumping', 'Standing', 'Walking', 'Lying Down', 'Fall Down',
                            'Stand up', 'Sit down']
        self.num_class = len(self.class_names)
        self.device = device

        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)
        self.model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        # print(self.model)
        self.model.eval()

        # Fix the size of running_mean in Batch Normalization layers
        self.adjust_bn_running_mean_size()

    def adjust_bn_running_mean_size(self):
    # Iterate through layers
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                # Get the current size of running_mean
                current_size = layer.running_mean.size(0)

                # Resize running_mean and running_var
                if (current_size == 197): current_size = 69
                elif(current_size == 42): current_size = 69
                elif(current_size == 28): current_size = 46
                layer.running_mean = torch.nn.Parameter(layer.running_mean.new_empty(current_size))
                layer.running_var = torch.nn.Parameter(layer.running_var.new_empty(current_size))
                layer.weight = torch.nn.Parameter(layer.weight.new_empty(current_size))
                layer.bias = torch.nn.Parameter(layer.bias.new_empty(current_size))

                # Set requires_grad to False for running_mean and running_var
                layer.running_mean.requires_grad = False
                layer.running_var.requires_grad = False


                
    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (numpy array) Probability of each class actions.
        """
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]

        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        mot = mot.to(self.device)
        pts = pts.to(self.device)

        out = self.model((pts, mot))

        return out.detach().cpu().numpy()

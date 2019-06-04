# -*-coding: utf-8-*-
import torch

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
        representation for comparison to point form ground truth data.
        Args:
            boxes: (tensor) center-size default boxes from priorbox layers.
        Return:
            boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,            #xmin,ymin
                      boxes[:, :2] + boxes[:, 2:]/2), 1)        #xmax,ymax

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,            #cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)        #


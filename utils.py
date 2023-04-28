import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_IoU(cm):
    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)

    # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives
    
    iou = true_positives / denominator
    
    return iou, np.nanmean(iou) 

def eval_net_loader(net, val_loader, n_classes, device='cpu'):
    
    net.eval()
    labels = np.arange(n_classes)
    cm = np.zeros((n_classes,n_classes))
      
    for i, sample_batch in enumerate(val_loader):
            imgs = sample_batch['image']
            true_masks = sample_batch['mask']
            
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            outputs = net(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            for j in range(len(true_masks)): 
                true = true_masks[j].cpu().detach().numpy().flatten()
                pred = preds[j].cpu().detach().numpy().flatten()
                cm += confusion_matrix(true, pred, labels=labels)
    
    class_iou, mean_iou = compute_IoU(cm)
    
    return class_iou, mean_iou

def IoU(mask_true, mask_pred, n_classes=2):
        
        labels = np.arange(n_classes)
        cm = confusion_matrix(mask_true.flatten(), mask_pred.flatten(), labels=labels)
        
        return compute_IoU(cm)


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
#             print(preds.shape)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()



class get_dice_score(nn.Module):
    def __init__(self, smooth=1):
        super(get_dice_score, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, target):
        # flatten label and prediction tensors
        outputs_flat = outputs.view(-1)
        target_flat = target.view(-1)

        # calculate intersection and union
        intersection = (outputs_flat * target_flat).sum()
        union = outputs_flat.sum() + target_flat.sum()

        # calculate dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # calculate dice loss
        dice_loss = 1 - dice

        return dice_loss

def get_dice_score(output, target, smooth = 1):
    # # flatten label and prediction tensors
    # outputs_flat = outputs.view(-1)
    # target_flat = target.view(-1)

    # # calculate intersection and union
    # intersection = (outputs_flat * target_flat).sum()
    # union = outputs_flat.sum() + target_flat.sum()

    # # calculate dice coefficient
    # dice = (2. * intersection + smooth) / (union + smooth)

    # # calculate dice loss
    # dice_loss = 1 - dice

    # return dice_loss
        # smooth = 1.
    output = torch.sigmoid(output)
    output = output.flatten()
    target = target.flatten()
    intersection = (output * target).sum()
    union = output.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    """
    Calculate the Dice Loss for a single class.
    Args:
        pred: predicted mask of shape (batch_size, H, W)
        target: ground truth mask of shape (batch_size, H, W)
        smooth: smoothing value to avoid division by zero
    Returns:
        Dice Loss for the given class.
    """
    intersection = (pred * target).sum(dim=(1, 2))
    # if intersection.sum() < 0:
    #     print(f'intersection: {intersection}')
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    # if union.sum() < 0:
    #     print(f'union: {union}')
    dice = (2 * intersection + smooth) / (union + smooth)
    # if dice.mean() <0 or dice.mean() > 1:
    #     print(f'dice.mean(): {dice.mean()}')
        # return torch.tensor(1.0, requires_grad=True).to(device)
    # if dice.mean() <0:
        # print(f'dice.mean(): {dice.mean()}')
        # return torch.tensor(1.0, requires_grad=True).to(device)
    return 1 - dice.mean()



def dice_score(pred, target, smooth=1):
    """
    Calculate the Dice Loss for a single class.
    Args:
        pred: predicted mask of shape (batch_size, H, W)
        target: ground truth mask of shape (batch_size, H, W)
        smooth: smoothing value to avoid division by zero
    Returns:
        Dice Loss for the given class.
    """
    intersection = (pred * target).sum(dim=(1, 2))
    # if intersection.sum() < 0:
    #     print(f'intersection: {intersection}')
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    # if union.sum() < 0:
    #     print(f'union: {union}')
    dice = (2 * intersection + smooth) / (union + smooth)
    # if dice.mean() <0 or dice.mean() > 1:
    #     print(f'dice.mean(): {dice.mean()}')
        # return torch.tensor(1.0, requires_grad=True).to(device)
    # if dice.mean() <0:
        # print(f'dice.mean(): {dice.mean()}')
        # return torch.tensor(1.0, requires_grad=True).to(device)
    return 1 - dice.mean()



def multi_class_dice_loss(preds, targets, smooth=1):
    """
    Calculate the multi-class Dice Loss.
    Args:
        preds: predicted masks of shape (batch_size, num_classes, H, W)
        target: ground truth mask of shape (batch_size, H, W)
        smooth: smoothing value to avoid division by zero
    Returns:
        Average Dice Loss across all classes.
    """
    num_classes = preds.shape[1]
    total_loss = 0
    for i in range(num_classes):
        pred = preds[:, i, :, :]
        # target_class = (target == i+1).float() # i+1 for class 1 and 2 (0 is not a class)
        target_class = targets[:,i,:,:] # i for class 1 and 2 (0 is not a class)
        dice_l = dice_loss(pred, target_class, smooth=smooth)
        # if dice_l < 0:
            # print(f'dice_l: {dice_l}')
        total_loss += dice_l
    # num_classes = -num_classes
    # if total_loss < 0 or total_loss / num_classes < 0 or total_loss / num_classes > 1:
        # print(f'total_loss: {total_loss / num_classes}')
    # if total_loss / num_classes < 0:
    #     print("Loss is less than 0", total_loss / num_classes)
    #     return torch.tensor(0.0, requires_grad=True).to(device)
        
    # if total_loss / num_classes > 1:
    #     print("Loss is greater than 1", total_loss / num_classes)
    #     return torch.tensor(1.0, requires_grad=True).to(device)

    return total_loss / num_classes


def multi_class_dice_score(preds, targets, smooth=1):
    """
    Calculate the multi-class Dice Loss.
    Args:
        preds: predicted masks of shape (batch_size, num_classes, H, W)
        target: ground truth mask of shape (batch_size, H, W)
        smooth: smoothing value to avoid division by zero
    Returns:
        Average Dice Loss across all classes.
    """
    num_classes = preds.shape[1]
    total_loss = 0
    for i in range(num_classes):
        pred = preds[:, i, :, :]
        # target_class = (target == i+1).float() # i+1 for class 1 and 2 (0 is not a class)
        target_class = targets[:,i,:,:] # i for class 1 and 2 (0 is not a class)
        dice_l = dice_score(pred, target_class, smooth=smooth)
        # if dice_l < 0:
            # print(f'dice_l: {dice_l}')
        total_loss += dice_l
    # num_classes = -num_classes
    # if total_loss < 0 or total_loss / num_classes < 0 or total_loss / num_classes > 1:
        # print(f'total_loss: {total_loss / num_classes}')
    # if total_loss / num_classes < 0:
    #     print("Loss is less than 0", total_loss / num_classes)
    #     return torch.tensor(0.0, requires_grad=True).to(device)
        
    # if total_loss / num_classes > 1:
    #     print("Loss is greater than 1", total_loss / num_classes)
    #     return torch.tensor(1.0, requires_grad=True).to(device)

    return total_loss / num_classes
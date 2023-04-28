import torch
import torch.nn as nn
from dataset import get_transforms, CT_Dataset
from models import UNet
from tqdm import tqdm
from utils import get_dice_score, eval_net_loader, multi_class_dice_loss, multi_class_dice_score
import argparse
import tensorboardX
import os
import monai
from monai.losses import DiceFocalLoss
import segmentation_models_pytorch as smp

# 10.127.30.128  hca3  focal loss + dice loss
# 10.127.30.125  hca3  dice loss only

def main(args):
    # Load the dataset
    train_transforms, val_transforms = get_transforms()
    include_background = True if args.num_classes == 3 else False
    train_dataset = CT_Dataset(csv_dir = args.csv_dir, image_set="train", transforms= train_transforms, include_background=include_background)
    val_dataset = CT_Dataset(csv_dir = args.csv_dir, image_set="val", transforms= val_transforms, include_background=include_background)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size =args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size =args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Tensorboard
    writer = tensorboardX.SummaryWriter(os.path.join(args.log_dir,  args.model_name))

    # model = UNet(n_channels=3, n_classes=args.num_classes) # inlcluding background
    if args.pretrained:
        model = smp.Unet(
            encoder_name=args.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,                      # model output channels (number of classes in your dataset)
        )
    else:
        model = smp.Unet(
            encoder_name=args.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,                      # model output channels (number of classes in your dataset)
        )
    model = model.to(args.device)

    # Define the loss function and optimizer
    # criterion = multi_class_dice_loss
    # criterion_bce = nn.BCEWithLogitsLoss()
    # criterion_l2 = nn.MSELoss()
    # criterion_dice = monai.losses.DiceFocalLoss()
    # criterion_ce = nn.CrossEntropyLoss()

    if args.loss == "dice":
        criterion = monai.losses.DiceLoss() # 1 - dice_score
    elif args.loss == "focal":
        criterion = monai.losses.FocalLoss()
    elif args.loss == "GeneralizedDiceFocalLoss":
        criterion = monai.losses.GeneralizedDiceFocalLoss()
    elif args.loss == "BCE":
        criterion = nn.BCELoss()

    elif args.loss == "dice_score": # returns 1 - dice_score with for non-empty values only
        def criterion(y_pred, y_true):
            if len(y_pred.shape) == 3: # for single class
                return monai.metrics.compute_dice(y_pred.reshape(y_pred.shape[0], 1, args.image_size, args.image_size), 
                                            y_true.reshape(y_true.shape[0], 1, args.image_size, args.image_size), 
                                            include_background=False, ignore_empty=False).mean().item()
            # for multi class
            return monai.metrics.compute_dice(y_pred, y_true, include_background=False, ignore_empty=False).mean().item()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_eval_dice = 0
    sig = nn.Sigmoid()

    # Train the model
    for epoch in range(args.epochs):
        total_train_loss_cache = [] # used for tensorboard logs
        cls1_train_loss_cache = []
        cls2_train_loss_cache = []

        # l2_train_loss_cache = []
        # bce_train_loss_cache = []
        # dicefocal_train_loss_cache = []

        model.train()
        for i, (images, true_masks) in tqdm(enumerate(train_loader)):
            images = images.to(args.device)
            true_masks = true_masks.to(args.device)

            optimizer.zero_grad()

            pred_masks = sig(model(images))
            pred_cls1 = pred_masks[:, 0, :, :] # cls 1: liver 
            pred_cls2 = pred_masks[:, 1, :, :] # cls 2: tumor

            target_cls1 = true_masks[:, 0, :, :] # cls 1: liver
            target_cls2 = true_masks[:, 1, :, :] # cls 2: tumor

            loss_cl1 = criterion(pred_cls1, target_cls1)
            loss_cl2 = criterion(pred_cls2, target_cls2)
            
            # weighted loss
            loss = loss_cl1*args.loss_weight_cls1 + loss_cl2*args.loss_weight_cls2 

            # loss = criterion(pred_masks, true_masks)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            total_train_loss_cache.append(loss.item())
            cls1_train_loss_cache.append(loss_cl1.item())
            cls2_train_loss_cache.append(loss_cl2.item())
            if (i + 1) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Total Loss: {:.4f}, Class1 Loss: {:.4f}, Class2 Loss: {:.4f}'
                      .format(epoch + 1, args.epochs, i + 1, len(train_loader), loss.item(), loss_cl1.item(), loss_cl2.item()))
                
        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_dice_score = 0
            val_dice_score_cls1 = 0
            val_dice_score_cls2 = 0
            val_jaccard_score = 0
            val_jaccard_score_cls1 = 0
            val_jaccard_score_cls2 = 0

            val_loss_cls1 = 0
            val_loss_cls2 = 0
            

            for images, true_masks in val_loader:
                images = images.to(args.device)
                true_masks = true_masks.to(args.device)

                pred_masks = sig(model(images))
                
                pred_cls1= pred_masks[:, 0, :, :]
                pred_cls2 = pred_masks[:, 1, :, :]

                true_cls1 = true_masks[:, 0, :, :]
                true_cls2 = true_masks[:, 1, :, :]                

                val_loss_cls1 += criterion(pred_cls1, true_cls1).item()
                val_loss_cls2 += criterion(pred_cls2, true_cls2).item()
                val_dice_score += monai.metrics.compute_dice(pred_masks, true_masks, include_background=False, ignore_empty=False).mean() # mean dice score
                val_dice_score_cls1 += monai.metrics.compute_dice(pred_cls1.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls1.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False, ignore_empty=False).mean().item() 
                val_dice_score_cls2 += monai.metrics.compute_dice(pred_cls2.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls2.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False, ignore_empty=False).mean().item()

                val_jaccard_score += monai.metrics.compute_iou(pred_masks, true_masks, include_background=False, ignore_empty=False).mean() # mean jaccard score            
                val_jaccard_score_cls1 += monai.metrics.compute_iou(pred_cls1.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls1.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False, ignore_empty=False).mean().item()
                val_jaccard_score_cls2 += monai.metrics.compute_iou(pred_cls2.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls2.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False, ignore_empty=False).mean().item()

                # print(monai.metrics.compute_dice(pred_masks, true_masks, include_background=False).mean())
                # print(monai.metrics.compute_dice(pred_cls1.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls1.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False).mean())
                # print(monai.metrics.compute_dice(pred_cls2.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls2.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False).mean())

                # print(monai.metrics.compute_iou(pred_masks, true_masks, include_background=False).mean())
                # print(monai.metrics.compute_iou(pred_cls1.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls1.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False).mean())
                # print(monai.metrics.compute_iou(pred_cls2.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls2.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False).mean())

            print(len(val_dataset))
            val_loss_cls1 /= len(val_dataset)
            val_loss_cls2 /= len(val_dataset)

            # val_dice_score = val_dice_score   
            val_dice_score /= len(val_dataset)
            val_dice_score_cls1 /= len(val_dataset)
            val_dice_score_cls2 /= len(val_dataset)

            val_jaccard_score /= len(val_dataset)
            val_jaccard_score_cls1 /= len(val_dataset)
            val_jaccard_score_cls2 /= len(val_dataset)

            # val_loss_bce /= len(val_dataset)

            #  Log to tensorboard
            # writer.add_scalar('val_loss_dice', val_dice_score, epoch)
            writer.add_scalar('vall_loss_cls1', val_loss_cls1, epoch)
            writer.add_scalar('vall_loss_cls2', val_loss_cls2, epoch)
            # log dice score
            writer.add_scalar('val_dice_score', val_dice_score, epoch)
            writer.add_scalar('val_dice_score_cls1', val_dice_score_cls1, epoch)
            writer.add_scalar('val_dice_score_cls2', val_dice_score_cls2, epoch)
            # log jaccard score
            writer.add_scalar('val_jaccard_score', val_jaccard_score, epoch)
            writer.add_scalar('val_jaccard_score_cls1', val_jaccard_score_cls1, epoch)
            writer.add_scalar('val_jaccard_score_cls2', val_jaccard_score_cls2, epoch)

            # log train loss
            writer.add_scalar('train_loss', sum(total_train_loss_cache)/len(total_train_loss_cache), epoch)
            writer.add_scalar('train_loss_cls1', sum(cls1_train_loss_cache)/len(cls1_train_loss_cache), epoch)
            writer.add_scalar('train_loss_cls2', sum(cls2_train_loss_cache)/len(cls2_train_loss_cache), epoch)

            # print train dice score
            print('')
            print("------------------------------------")
            print(f"val loss cls1: {val_loss_cls1} , val loss cls2: {val_loss_cls2} ")
            print(f"val dice score: {val_dice_score}, val dice score cls1: {val_dice_score_cls1}, val dice score cls2: {val_dice_score_cls2}")
            print(f"val jaccard score: {val_jaccard_score},val jaccard score cls1: {val_jaccard_score_cls1} , val jaccard score cls2: {val_jaccard_score_cls2}")
            print("------------------------------------")
            print('')


            if val_dice_score > best_eval_dice:
                best_eval_dice = val_dice_score
                # checkpoint_dir = os.path.join(args.model_dir, f'{args.model_name}_{args.backbone}_{args.loss_weight_cls1}_{args.loss_weight_cls2}')
                checkpoint_dir = os.path.join(args.model_dir, args.model_name)
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch}_val_dice_{val_dice_score}.pth'))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # data args
    args.add_argument('--csv_dir', type=str, default='/home/anees.hashmi/Desktop/labs/hc701_assignment_3/csv')
    args.add_argument('--model_dir', type=str, default='/home/anees.hashmi/Desktop/labs/hc701_assignment_3/models')
    args.add_argument('--log_dir', type=str, default='/home/anees.hashmi/Desktop/labs/hc701_assignment_3/logs')

    # model args
    args.add_argument ('--model_name', type=str, default='unet')
    args.add_argument('--backbone', type=str, default='mobilenet_v2') # resnet101 or mobilenet_v2
    args.add_argument('--pretrained', type=bool, default=False)

    # training args
    args.add_argument('--batch_size', type=int, default=2)
    args.add_argument('--num_classes', type=int, default=2)
    args.add_argument('--epochs', type=int, default=1)
    args.add_argument('--lr', type=float, default=1e-5)
    args.add_argument('--log_step', type=int, default=10)
    args.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args.add_argument('--num_workers', type=int, default=24)
    args.add_argument('--image_size', type=int, default=256)
    
    # loss args
    args.add_argument('--loss', type=str, default='dice')
    args.add_argument("--loss_weight_cls1", type=float, default=1.0)
    args.add_argument("--loss_weight_cls2", type=float, default=5.0)

    config = args.parse_args()
    config.model_name = f'{config.model_name}_{config.loss}_loss_{config.backbone}_{config.loss_weight_cls1}_{config.loss_weight_cls2}'

    os.makedirs(os.path.join(config.model_dir, config.model_name), exist_ok=True)
    os.makedirs(os.path.join(config.log_dir,  config.model_name), exist_ok=True)

    print(config)
    main(config)

    # experiment 1: unet_dice_loss_mobilenet_v2_1.0_5.0 (not good enough)
    # experiment 2: unet_dice_loss_mobilenet_v2_1.0_1.0 


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
# import simplespace 
import pandas as pd

from torchmetrics.classification import MulticlassJaccardIndex

def main(args):
    # Load the dataset
    _, test_transforms = get_transforms()
    include_background = True if args.num_classes == 3 else False

    # train_dataset = CT_Dataset(csv_dir = args.csv_dir, image_set="train", transforms= train_transforms)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataset = CT_Dataset(csv_dir = args.csv_dir, image_set="test", transforms= test_transforms, include_background=include_background)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Tensorboard
    # uploade 10 sample images to tensorboard


    # Load the model

    # model = UNet(n_channels=3, n_classes=args.num_classes) # inlcluding background
    model = smp.Unet(
        encoder_name=args.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
    )
    model = model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name, args.checkpoint)))

    # criterion_dice = monai.losses.DiceLoss()
    # jaccard_metric = MulticlassJaccardIndex(num_classes=args.num_classes, average="macro").to(args.device)
    # get_iou = monai.metrics.compute_meaniou

    sig = nn.Sigmoid()

    model.eval()
    with torch.no_grad():
        test_dice_score = 0
        test_dice_score_cls1 = 0
        test_dice_score_cls2 = 0
        
        test_jaccard_score = 0
        test_jaccard_score_cls1 = 0
        test_jaccard_score_cls2 = 0

        
        for images, true_masks in test_loader:
            images = images.to(args.device)
            true_masks = true_masks.to(args.device)

            pred_masks = sig(model(images))
            
            pred_cls1= pred_masks[:, 0, :, :]
            pred_cls2 = pred_masks[:, 1, :, :]

            true_cls1 = true_masks[:, 0, :, :]
            true_cls2 = true_masks[:, 1, :, :]                

            test_dice_score += monai.metrics.compute_dice(pred_masks, true_masks, include_background=False, ignore_empty=False).mean() # mean dice score
            test_dice_score_cls1 += monai.metrics.compute_dice(pred_cls1.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls1.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False, ignore_empty=False).mean().item() 
            test_dice_score_cls2 += monai.metrics.compute_dice(pred_cls2.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls2.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False, ignore_empty=False).mean().item()

            test_jaccard_score += monai.metrics.compute_iou(pred_masks, true_masks, include_background=False, ignore_empty=False).mean() # mean jaccard score            
            test_jaccard_score_cls1 += monai.metrics.compute_iou(pred_cls1.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls1.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False, ignore_empty=False).mean().item()
            test_jaccard_score_cls2 += monai.metrics.compute_iou(pred_cls2.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls2.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False, ignore_empty=False).mean().item()

            # print(monai.metrics.compute_dice(pred_masks, true_masks, include_background=False).mean())
            # print(monai.metrics.compute_dice(pred_cls1.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls1.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False).mean())
            # print(monai.metrics.compute_dice(pred_cls2.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls2.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False).mean())

            # print(monai.metrics.compute_iou(pred_masks, true_masks, include_background=False).mean())
            # print(monai.metrics.compute_iou(pred_cls1.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls1.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False).mean())
            # print(monai.metrics.compute_iou(pred_cls2.reshape(pred_cls1.shape[0], 1, 256, 256), true_cls2.reshape(true_cls1.shape[0], 1, 256, 256), include_background=False).mean())

        print(len(test_loader))

        # val_dice_score = val_dice_score   
        test_dice_score /= len(test_loader)
        test_dice_score_cls1 /= len(test_loader)
        test_dice_score_cls2 /= len(test_loader)

        test_jaccard_score /= len(test_loader)
        test_jaccard_score_cls1 /= len(test_loader)
        test_jaccard_score_cls2 /= len(test_loader)




    # with torch.no_grad():
    #     test_loss_dice = 0
    #     test_loss_dice_cls1 = 0
    #     test_loss_dice_cls2 = 0

    #     test_jaccard = 0
    #     test_jaccard_cls1 = 0
    #     test_jaccard_cls2 = 0

    #     num_correct = 0
    #     num_correct_cls1 = 0
    #     num_correct_cls2 = 0
    #     num_pixels = 0
    #     num_pixels_cls1 = 0
    #     num_pixels_cls2 = 0
        
    #     for images, true_masks in tqdm(test_loader):
    #         images = images.to(args.device)
    #         true_masks = true_masks.to(args.device)
    #         pred_masks = sig(model(images))

    #         pred_cls1 = pred_masks[:, 0, :, :]
    #         pred_cls2 = pred_masks[:, 1, :, :]

    #         target_cls1 = true_masks[:, 0, :, :]
    #         target_cls2 = true_masks[:, 1, :, :]

    #         y = (pred_masks > 0.5).float()
    #         y1= (pred_cls1 > 0.5).float()
    #         y2= (pred_cls2 > 0.5).float()

    #         # pred_masks = torch.stack([pred_cls1, pred_cls2], dim=1)

    #         test_loss_dice_cls1 = criterion_dice(pred_cls1, target_cls1).item() * images.size(0)
    #         test_loss_dice_cls2 = criterion_dice(pred_cls2, target_cls2).item() * images.size(0)
    #         test_loss_dice += criterion_dice(pred_masks, true_masks).item() * images.size(0)

    #         # test_jaccard += jaccard_metric(pred_masks, true_masks).item() * images.size(0)
    #         # test_jaccard_cls1 += jaccard_metric(pred_cls1, target_cls1).item() * images.size(0)
    #         # test_jaccard_cls2 += jaccard_metric(pred_cls2, target_cls2).item() * images.size(0)

    #         test_jaccard += jaccard_metric(y, true_masks).item() * images.size(0)
    #         test_jaccard_cls1 += jaccard_metric(y1, target_cls1).item() * images.size(0)
    #         test_jaccard_cls2 += jaccard_metric(y2, target_cls2).item() * images.size(0)

    #         num_correct += (y == true_masks).sum()
    #         num_correct_cls1 += (y1 == target_cls1).sum()
    #         num_correct_cls2 += (y2 == target_cls2).sum()
                        
    #         num_pixels += torch.numel(true_masks)
    #         num_pixels_cls1 += torch.numel(target_cls1)
    #         num_pixels_cls2 += torch.numel(target_cls2)

    #         # img_iou = get_iou(pred_masks, true_masks).mean()
    #         # if img_iou.isnan():
    #         #     img_iou = torch.tensor(0.0, device=args.device, dtype=torch.float)
    #         # test_iou += img_iou.item()


    #     test_loss_dice /= len(test_dataset)
    #     test_loss_dice_cls1 /= len(test_dataset)
    #     test_loss_dice_cls2 /= len(test_dataset)
    #     test_dice_score = 1 - test_loss_dice
    #     test_dice_score_cls1 = 1 - test_loss_dice_cls1
    #     test_dice_score_cls2 = 1 - test_loss_dice_cls2

    #     test_jaccard /= len(test_dataset)
    #     test_jaccard_cls1 /= len(test_dataset)
    #     test_jaccard_cls2 /= len(test_dataset)

    #     test_accuracy = num_correct / num_pixels
    #     test_accuracy = test_accuracy.cpu().numpy()
    #     test_accuracy_cls1 = num_correct_cls1 / num_pixels_cls1
    #     test_accuracy_cls1 = test_accuracy_cls1.cpu().numpy()
    #     test_accuracy_cls2 = num_correct_cls2 / num_pixels_cls2
    #     test_accuracy_cls2 = test_accuracy_cls2.cpu().numpy()


        print(f'Test Dice Score: {test_dice_score:.4f}')
        print(f'Test Dice Score cls1: {test_dice_score_cls1:.4f}')
        print(f'Test Dice Score cls2: {test_dice_score_cls2:.4f}')

        print("")
        print('----------------------------- ')
        print("")

        print(f'Test Jackard: {test_jaccard_score:.4f}')
        print(f'Test Jackard cls1: {test_jaccard_score_cls1:.4f}')
        print(f'Test Jackard cls2: {test_jaccard_score_cls2:.4f}')

        print("")
        print('----------------------------- ')
        print("")

        # print(f'Test Accuracy: {test_accuracy:.4f}')
        # print(f'Test Accuracy cls1: {test_accuracy_cls1:.4f}')
        # print(f'Test Accuracy cls2: {test_accuracy_cls2:.4f}')


        df_dict = {'model_path': [os.path.join(args.model_path, args.checkpoint)], 
                   'Dice Score': [test_dice_score], 
                   "Dice Score cls0":[test_dice_score_cls1], 
                   "Dice Score cls1":[test_dice_score_cls2], 
                   'Jackard': [test_jaccard_score], 
                   'Test Jackard cls1': [test_jaccard_score_cls1], 
                   'Test Jackard cls2': [test_jaccard_score_cls2]}

        try:
            df = pd.read_csv(f"{args.result_dir}{args.model_name}.csv")
            df = df.concat(df, pd.DataFrame(df_dict))
            df.to_csv(f"{args.result_dir}{args.model_name}.csv", index=False)
        except:
            df = pd.DataFrame(df_dict)
            df.to_csv(f"{args.result_dir}{args.model_name}.csv", index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--csv_dir', type=str, default='/home/anees.hashmi/Desktop/labs/hc701_assignment_3/csv')
    args.add_argument('--model_dir', type=str, default='/home/anees.hashmi/Desktop/labs/hc701_assignment_3/models')

    args.add_argument ('--model_name', type=str, default='unet_dice_loss_mobilenet_v2_1.0_5.0')
    args.add_argument('--backbone', type=str, default='mobilenet_v2') # resnet101 or mobilenet_v2

    args.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args.add_argument('--num_workers', type=int, default=24)
    args.add_argument('--image_size', type=int, default=256)
    args.add_argument('--batch_size', type=int, default=2)
    args.add_argument('--num_classes', type=int, default=2)
    args.add_argument("--checkpoint", type=str, default=None)

    args.add_argument("--model_path", type=str, default="/home/anees.hashmi/Desktop/labs/hc701_assignment_3/models/")
    args.add_argument("--result_dir", type=str, default="/home/anees.hashmi/Desktop/labs/hc701_assignment_3/results/")
    config = args.parse_args()

    # if not args['checkpoint']:
    #     all_chckpoints = [i[:-5] for i in sorted(os.listdir(f"./models/{args['model_name']}"))]
    #     args["checkpoint"] = ["-".join(i) for i in sorted([i.split('-') for i in all_chckpoints], key=lambda x: x[2])][-1]
    # if not config.checkpoint:
    #     all_chckpoints = [i[:-4] for i in sorted(os.listdir(f"{config.model_path}{config.model_name}"))]
    #     config.checkpoint = f"""{["_".join(i) for i in sorted([i.split('_') for i in all_chckpoints], key=lambda x: int(x[-4]))][-1]}.ckpt"""

    config.checkpoint = "best_model_unet_dice_loss_mobilenet_v2_1.0_5.0_epoch_48_val_dice_5.162607408237818e-08.pth"

    os.makedirs(config.result_dir, exist_ok=True)

    # config.update(vars(args)
    print(config)
    main(config)
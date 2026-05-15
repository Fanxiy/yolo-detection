import argparse
import os
import time
from ultralytics import YOLO
import wandb


# ===================== 统一记录训练 & 验证指标 =====================
def log_all_metrics(trainer):
    """
    每个epoch（训练+验证）结束后统一上传：训练loss、验证loss、mAP
    """
    if not wandb.run:
        return

    log_data = {}

    # 1. 获取训练集 loss
    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
        box_loss = float(trainer.loss_items[0])
        cls_loss = float(trainer.loss_items[1])
        dfl_loss = float(trainer.loss_items[2])

        train_total_loss = box_loss + cls_loss + dfl_loss

        log_data["train/box_loss"] = box_loss
        log_data["train/cls_loss"] = cls_loss
        log_data["train/dfl_loss"] = dfl_loss
        log_data["train/total_loss"] = train_total_loss

    # 2. 获取验证集 loss + mAP
    val_box, val_cls, val_dfl = 0., 0., 0.
    if hasattr(trainer, 'metrics') and trainer.metrics is not None:
        for k, v in trainer.metrics.items():
            if isinstance(v, (int, float)):
                log_data[f"{k}"] = float(v)
                if k == "val/box_loss": val_box = float(v)
                if k == "val/cls_loss": val_cls = float(v)
                if k == "val/dfl_loss": val_dfl = float(v)
        val_total_loss = val_box + val_cls + val_dfl
        log_data["val/total_loss"] = val_total_loss

    # 3. wandb自动画曲线
    wandb.log(log_data, step=trainer.epoch)
# ================================================================


def train_model(args):
    """
    模型训练主函数，集成wandb可视化
    """
    model = YOLO(args.model)

    # 训练参数配置
    train_params = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'lr0': args.lr,
        'optimizer': args.optimizer,
        'device': args.device,
        'patience': args.patience,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        'pretrained': args.pretrained,
        'verbose': True,
        'workers': args.workers,
        'close_mosaic': 10,
        'cache': True,  
    }

    # 初始化wandb
    if args.wandb:
        os.environ['WANDB_PROJECT'] = args.wandb_project
        wandb.init(
            project=args.wandb_project,
            name=args.name,
            config=vars(args)  # vars() 将 args 转换为字典，方便存储
        )
        model.add_callback("on_fit_epoch_end", log_all_metrics)

    # 开始训练
    model.train(**train_params)
    
    if hasattr(model, 'trainer') and model.trainer is not None:
        result_dir = model.trainer.save_dir
        validator = model.trainer.validator
        if validator and hasattr(validator, 'metrics'):
            map50 = validator.metrics.box.map50
            map95 = validator.metrics.box.map
            print(f"\nTraining Finished! Best Validation mAP50: {map50:.4f}, mAP50-95: {map95:.4f}")
            
            if args.wandb and wandb.run:
                wandb.log({
                    "final/mAP50": map50,
                    "final/mAP50-95": map95
                })
    else:
        result_dir = args.project

    if args.wandb and wandb.run:
        wandb.finish()

    # 导出模型
    if args.export:
        print("Exporting model to ONNX...")
        model.export(format='onnx')

    return model, result_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO Training with Wandb Visualization')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset config path')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], help='optimizer type')
    parser.add_argument('--device', type=str, default='0', help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers (8 is best for 16vCPU)')
    parser.add_argument('--project', type=str, default='runs/train', help='save directory')
    
    # 时间戳记录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument('--name', type=str, default=f'exp_{timestamp}', help='experiment name')
    
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='do not use pretrained weights')
    parser.add_argument('--export', action='store_true', help='export onnx model')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false', help='disable wandb visualization')
    parser.add_argument('--wandb_project', type=str, default='yolo-detection', help='wandb project name')

    args = parser.parse_args()
    train_model(args)
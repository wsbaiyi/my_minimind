import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from models.model import MiniMindLM
from models.LMConfig import LMConfig
from models.dataset import PretrainDataset

warnings.filterwarnings('ignore')



def Logger(content):
    # 分布式训练ddp（如PyTorch的DistributedDataParallel
    # dist.get_rank() == 0：在分布式环境中（ddp=True），仅rank 0（主进程）执行日志操作，避免多进程写文件冲突。

    """
    日志记录函数，用于打印日志并保存到文件。
    在分布式训练中，仅由主进程（rank 0）执行日志操作，避免多进程写文件冲突。
    """
    if not ddp or dist.get_rank() == 0:
        print(content)
        with open('output.txt', 'a', encoding='utf-8') as file:
            print(content, file=file)


def get_lr(current_step, total_steps, lr):
    """
    计算当前学习率，使用余弦退火策略。
    学习率从初始值逐渐减小，最终不会完全降为0，而是保留初始值的1/10。
    """
    # 0.5 * (1 + math.cos(math.pi * current_step / total_steps))的范围是[1--->0]，即逐步减小lr
    #  lr / 10 防止最后的lr为0
    return lr / 10 +  lr *0.5 * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, wandb):
    """
    训练一个epoch的函数。
    - epoch: 当前epoch的序号
    - wandb: wandb对象，用于记录训练过程（如果启用）
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 定义损失函数，不自动求平均
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)  # 将输入数据移动到指定设备
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 计算当前学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 更新优化器的学习率

        with ctx:  # 使用自动混合精度（如果启用）
            res = model(X)  # 前向传播
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()  # 计算掩码损失
            loss += res.aux_loss  # 添加辅助损失（如果有）
            loss = loss / args.accumulation_steps  # 梯度累积

        scaler.scale(loss).backward()  # 反向传播（带梯度缩放）

        # 梯度累积完成后更新模型参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新梯度缩放器
            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        # 定期记录日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复实际损失值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 记录到wandb（如果启用）
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期保存模型
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            # 获取模型状态字典（处理DDP情况）
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)  # 保存模型
            model.train()

def init_model(lm_config):
    """初始化模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')  # 加载分词器

    # 初始化预训练模型
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer



def init_distributed_mode():
    """初始化分布式训练环境"""
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")  # 使用NCCL后端初始化进程组
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)  # 设置当前设备

# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    # 添加命令行参数
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./datasets/pretrain_hq.jsonl")

    # 命令行输入的参数保存在args中
    args = parser.parse_args() # 解析参数


    # 初始化模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = './train_res'  # 模型保存目录
    os.makedirs(args.save_dir, exist_ok=True)  # 创建目录
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len  # 每次迭代处理的token数
    torch.manual_seed(1337)  # 设置随机种子
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置wandb运行名称
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置自动混合精度上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 检查是否分布式训练
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"

    if ddp:
        init_distributed_mode()  # 初始化分布式训练
        args.device = torch.device(DEVICE)

    # 初始化wandb（如果启用）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)

    # 初始化数据集和数据加载器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 分布式采样器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 初始化梯度缩放器（用于混合精度训练）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # 优化器

    # 如果是分布式训练，包装模型
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)  # 每个epoch的迭代次数
    for epoch in range(args.epochs):  # 开始训练
        train_epoch(epoch, wandb)
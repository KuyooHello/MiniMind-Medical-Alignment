import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.dpo_diagnostics import MetricWindow, append_record, batch_integrity, evaluate_probe, probe_indices, token_kl_metrics
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, record_training_run

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels, return_full=False):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits.float(), dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return (log_probs_per_token, log_probs) if return_full else log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta, nll_weight=0.1,
             length_normalize=True, return_metrics=False):
    """Factorial ablation: sequence sum/mean preference scores; per-token chosen NLL."""
    seq_lengths = mask.sum(dim=1).clamp(min=1.0)
    ref_sum = (ref_log_probs * mask).sum(dim=1)
    policy_sum = (policy_log_probs * mask).sum(dim=1)
    policy_mean = policy_sum / seq_lengths
    ref_mean = ref_sum / seq_lengths
    ref_scores = ref_mean if length_normalize else ref_sum
    policy_scores = policy_mean if length_normalize else policy_sum
    half = ref_scores.shape[0] // 2
    pi_logratios = policy_scores[:half] - policy_scores[half:]
    ref_logratios = ref_scores[:half] - ref_scores[half:]
    margin = beta * (pi_logratios - ref_logratios)
    preference_loss = -F.logsigmoid(margin).mean()
    # Keep the NLL scale identical across both preference-score aggregations.
    chosen_nll = -policy_mean[:half].mean()
    loss = preference_loss + nll_weight * chosen_nll
    if return_metrics:
        return loss, {"preference_loss": preference_loss, "chosen_nll": chosen_nll,
                      "weighted_chosen_nll": nll_weight * chosen_nll,
                      "rejected_nll": -policy_mean[half:].mean(),
                      "ref_chosen_nll": -ref_mean[:half].mean(), "ref_rejected_nll": -ref_mean[half:].mean(),
                      "chosen_logp": policy_mean[:half].mean(), "rejected_logp": policy_mean[half:].mean(),
                      "chosen_logp_delta": (policy_mean[:half]-ref_mean[:half]).mean(),
                      "rejected_logp_delta": (policy_mean[half:]-ref_mean[half:]).mean(),
                      "chosen_seq_logp": policy_sum[:half].mean(), "rejected_seq_logp": policy_sum[half:].mean(),
                      "chosen_seq_logp_delta": (policy_sum[:half]-ref_sum[:half]).mean(),
                      "rejected_seq_logp_delta": (policy_sum[half:]-ref_sum[half:]).mean(),
                      "chosen_reward": (beta*(policy_scores[:half]-ref_scores[:half])).mean(),
                      "rejected_reward": (beta*(policy_scores[half:]-ref_scores[half:])).mean(),
                      "chosen_probability_decrease_rate": (policy_mean[:half] < ref_mean[:half]).float().mean(),
                      "rejected_probability_decrease_rate": (policy_mean[half:] < ref_mean[half:]).float().mean(),
                      "reward_margin": margin.mean(), "reward_margin_min": margin.min(), "reward_margin_max": margin.max(),
                      "preference_accuracy": (margin > 0).float().mean(),
                      "preference_tie_rate": (margin == 0).float().mean()}
    return loss


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()
    window = MetricWindow()
    epoch_metrics = MetricWindow()

    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        integrity = batch_integrity(x, y, mask, tokenizer.eos_token_id, tokenizer.pad_token_id)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs, ref_full_log_probs = logits_to_log_probs(ref_logits, y, return_full=True)
            
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs, policy_full_log_probs = logits_to_log_probs(logits, y, return_full=True)
            
            dpo_loss_val, loss_metrics = dpo_loss(
                ref_log_probs, policy_log_probs, mask, beta=beta,
                nll_weight=args.nll_weight, length_normalize=bool(args.length_normalize), return_metrics=True)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps

        if not torch.isfinite(loss):
            if is_main_process():
                append_record(args.diagnostic_path, 'failure', epoch=epoch+1, step=epoch*iters+step,
                              reason='nonfinite_loss')
            raise FloatingPointError('DPO loss 非有限，停止训练')

        scaler.scale(loss).backward()

        gradient_metrics = {}
        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            except RuntimeError:
                if is_main_process():
                    append_record(args.diagnostic_path, 'failure', epoch=epoch+1, step=epoch*iters+step,
                                  reason='gradient_norm_failed')
                raise
            post_clip_norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(parameter.grad.detach().float())
                for parameter in model.parameters() if parameter.grad is not None]))
            gradient_metrics = {'grad_norm_before_clip': grad_norm, 'grad_norm_before_clip_max': grad_norm,
                                'grad_norm_after_clip': post_clip_norm,
                                'grad_clipped': (grad_norm > args.grad_clip).float(),
                                'grad_scaler_scale': scaler.get_scale()}
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        diagnostic_metrics = {**loss_metrics, **integrity, **gradient_metrics,
                              'objective_loss': dpo_loss_val, 'nonfinite_loss_count': 0}
        if args.diagnostic_kl:
            diagnostic_metrics.update(token_kl_metrics(policy_full_log_probs, ref_full_log_probs, mask))
        pairs = x_chosen.shape[0]
        window.update(diagnostic_metrics, pairs)
        epoch_metrics.update(diagnostic_metrics, pairs)

        if step == start_step+1 or step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            metric_values = window.snapshot()
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min, metrics: {metric_values}')
            if is_main_process():
                append_record(args.diagnostic_path, 'train_window', epoch=epoch+1, step=epoch*iters+step,
                              batches=window.batches, pairs=window.pairs, learning_rate=current_lr,
                              elapsed_seconds=time.time()-start_time, metrics=metric_values)
            window = MetricWindow()
            
            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min, **metric_values})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        if is_main_process() and fixed_probe_indices and (
                step == iters or (args.probe_interval and (epoch*iters+step) % args.probe_interval == 0)):
            probe = evaluate_probe(model, ref_model, probe_ds, fixed_probe_indices, tokenizer, args.device,
                                   args.batch_size, args.beta, args.nll_weight, bool(args.length_normalize),
                                   dpo_loss, logits_to_log_probs, bool(args.diagnostic_kl))
            append_record(args.diagnostic_path, 'probe', epoch=epoch+1, step=epoch*iters+step, **probe)
            Logger(f"PROBE step={epoch*iters+step}: chosen_nll={probe['metrics']['chosen_nll']:.4f}, "
                   f"rejected_nll={probe['metrics']['rejected_nll']:.4f}")

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, ref_full_log_probs
        del outputs, logits, policy_log_probs, policy_full_log_probs, loss, loss_metrics, diagnostic_metrics

    if is_main_process():
        append_record(args.diagnostic_path, 'epoch_end', epoch=epoch+1, step=(epoch+1)*iters,
                      pairs=epoch_metrics.pairs, batches=epoch_metrics.batches, metrics=epoch_metrics.snapshot())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率；应结合求和/均值目标在验证集检查")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument('--nll_weight', default=0.1, type=float, help="chosen token 平均 NLL 权重；0 表示关闭")
    parser.add_argument('--length_normalize', default=1, type=int, choices=[0, 1], help="偏好分数使用 token 均值（1）或序列求和（0）")
    parser.add_argument('--seed', default=42, type=int, help="训练随机种子")
    parser.add_argument('--diagnostic_path', help="诊断 JSONL；默认保存于权重旁")
    parser.add_argument('--diagnostic_kl', type=int, choices=[0, 1], default=1, help="记录 teacher-forced token KL")
    parser.add_argument('--probe_samples', type=int, default=32, help="固定训练探针对数；0 关闭")
    parser.add_argument('--probe_interval', type=int, default=100, help="固定探针的训练步间隔；0 仅在每轮末评测")
    parser.add_argument('--probe_seed', type=int, default=73)
    parser.add_argument('--probe_keys_file', help="固定探针问题标识 JSON，供不同数据条件共享同一批样本")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
    if args.log_interval < 1 or args.probe_samples < 0 or args.probe_interval < 0:
        parser.error('日志间隔必须为正，探针数量与间隔必须非负')
    if args.diagnostic_path is None:
        suffix = '_moe' if args.use_moe else ''
        args.diagnostic_path = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}{suffix}.diagnostics.jsonl'
    if int(os.environ.get('RANK', -1)) <= 0 and os.path.exists(args.diagnostic_path) and not args.from_resume:
        parser.error(f'诊断记录已存在，请使用新的权重名称或路径: {args.diagnostic_path}')

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型（ref_model冻结）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    record_training_run(args, lm_config)
    fixed_probe_indices = []
    if is_main_process():
        if args.probe_samples:
            probe_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len, remove_empty_think=True)
            fixed_probe_indices = probe_indices(probe_ds.samples, args.probe_samples,
                                                args.probe_seed, args.probe_keys_file)
        append_record(args.diagnostic_path, 'header', settings=vars(args), probe_indices=fixed_probe_indices,
                      metric_scope='Sequence means over supervised response tokens; gradient means per optimizer update.',
                      train_scope='Forward metrics before update; gradient norms before and after clipping; probes after update.',
                      kl_scope='KL(policy || ref) over vocabulary at teacher-forced data contexts; not sequence-distribution KL.',
                      probe_scope='Fixed shared training pairs, empty think removed, live FP32 parameters, eval mode; no optimizer updates.')
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    if is_main_process() and fixed_probe_indices:
        probe = evaluate_probe(model, ref_model, probe_ds, fixed_probe_indices, tokenizer, args.device,
                               args.batch_size, args.beta, args.nll_weight, bool(args.length_normalize),
                               dpo_loss, logits_to_log_probs, bool(args.diagnostic_kl))
        epoch_batches = math.ceil((len(train_sampler) if train_sampler is not None else len(train_ds)) / args.batch_size)
        append_record(args.diagnostic_path, 'probe', epoch=start_epoch, step=start_epoch*epoch_batches+start_step,
                      initial=True, **probe)
        if not args.from_resume and probe['metrics']['policy_reference_logit_difference_max'] != 0:
            raise ValueError('初始化策略模型与参考模型不一致')
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
    
    record_training_run(args, lm_config, completed=True)
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

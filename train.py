import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import json
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dataloaders.dataloader_vctk import VCTKDemandDataset
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from models.loss import pesq_score, phase_losses
from models.discriminator import MetricDiscriminator, batch_pesq
from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed, 
    print_gpu_info, log_model_info, initialize_process_group,
)

torch.backends.cudnn.benchmark = True

def setup_optimizers(models, cfg):
    """Set up optimizers for the models."""
    generator, discriminator = models
    learning_rate = cfg['training_cfg']['learning_rate']
    betas = (cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2'])

    optim_g = optim.AdamW(generator.parameters(), lr=learning_rate, betas=betas)
    optim_d = optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=betas)

    return optim_g, optim_d

def setup_schedulers(optimizers, cfg, last_epoch):
    """Set up learning rate schedulers."""
    optim_g, optim_d = optimizers
    lr_decay = cfg['training_cfg']['lr_decay']

    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=last_epoch)

    return scheduler_g, scheduler_d

def normalize_feature_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a feature map per-sample to stabilize feature matching losses."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)

def load_teacher_model(kd_cfg, device):
    """Load a frozen teacher model for KD using its own config and checkpoint."""
    teacher_config = kd_cfg.get('teacher_config')
    teacher_checkpoint = kd_cfg.get('teacher_checkpoint')
    if not teacher_config or not teacher_checkpoint:
        raise ValueError("KD is enabled but teacher_config/teacher_checkpoint is missing.")

    teacher_cfg = load_config(teacher_config)
    teacher = SEMamba(teacher_cfg).to(device)

    print(f"Loading teacher checkpoint from {teacher_checkpoint}")
    teacher_state = torch.load(teacher_checkpoint, map_location=device)
    teacher_state = teacher_state.get('generator', teacher_state)
    teacher.load_state_dict(teacher_state, strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher

def get_model_module(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module (handles DDP-wrapped models)."""
    return model.module if isinstance(model, DistributedDataParallel) else model

def init_student_from_teacher(student, teacher, kd_cfg, rank: int) -> None:
    """Initialize the student from overlapping teacher weights when requested."""
    if not kd_cfg.get('student_init_from_teacher', False):
        return
    student_module = get_model_module(student)
    teacher_module = get_model_module(teacher)
    student_state = student_module.state_dict()
    teacher_state = teacher_module.state_dict()

    overlap = {}
    for key, teacher_value in teacher_state.items():
        if key in student_state and student_state[key].shape == teacher_value.shape:
            overlap[key] = teacher_value
    if not overlap:
        if rank == 0:
            print("KD student init requested but no overlapping parameters were found.")
        return

    student_state.update(overlap)
    student_module.load_state_dict(student_state, strict=False)
    if rank == 0:
        total = len(student_state)
        copied = len(overlap)
        print(f"Initialized student from teacher: copied {copied}/{total} parameter tensors.")

def set_named_components_requires_grad(model, component_names, requires_grad: bool, rank: int) -> int:
    """Enable/disable gradients for a list of top-level components by name."""
    module = get_model_module(model)
    changed_params = 0
    missing = []
    for name in component_names:
        submodule = getattr(module, name, None)
        if submodule is None:
            missing.append(name)
            continue
        for param in submodule.parameters():
            param.requires_grad = requires_grad
            changed_params += param.numel()
    if missing and rank == 0:
        missing_str = ", ".join(missing)
        print(f"Warning: KD freeze requested unknown components: {missing_str}")
    return changed_params

def create_dataset(cfg, train=True, split=True, device='cuda:0'):
    """Create dataset based on cfguration."""
    clean_json = cfg['data_cfg']['train_clean_json'] if train else cfg['data_cfg']['valid_clean_json']
    noisy_json = cfg['data_cfg']['train_noisy_json'] if train else cfg['data_cfg']['valid_noisy_json']
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if train else False
    pcs = cfg['training_cfg']['use_PCS400'] if train else False
    
    return VCTKDemandDataset(
        clean_json=clean_json,
        noisy_json=noisy_json,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        device=device,
        pcs=pcs
    )

def create_dataloader(dataset, cfg, train=True):
    """Create dataloader based on dataset and configuration."""
    if cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg['training_cfg']['training_epochs'])
        batch_size = (cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']) if train else 1
    else:
        sampler = None
        batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers'] if train else 1

    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True if train else False
    )

def cleanup_checkpoints(checkpoint_history, most_recent_step, exp_path, keep_best_n=11):
    """
    Clean up old checkpoints, keeping only:
    - Best N checkpoints by PESQ score
    - Most recent checkpoint (regardless of PESQ)
    
    Args:
        checkpoint_history: dict with {step: {'pesq': score, 'validated': bool}}
        most_recent_step: int, the most recent checkpoint step
        exp_path: str, path to experiment directory
        keep_best_n: int, number of best checkpoints to keep (default: 11)
    
    Returns:
        Updated checkpoint_history dict with only kept checkpoints
    """
    # Only consider validated checkpoints for ranking
    validated_ckpts = {step: info for step, info in checkpoint_history.items() 
                       if info.get('validated', False)}
    
    if not validated_ckpts:
        return checkpoint_history
    
    # Sort by PESQ score (descending)
    sorted_ckpts = sorted(validated_ckpts.items(), key=lambda x: x[1]['pesq'], reverse=True)
    
    # Identify checkpoints to keep: best N + most recent
    steps_to_keep = set()
    
    # Add top N by PESQ
    for i, (step, info) in enumerate(sorted_ckpts):
        if i < keep_best_n:
            steps_to_keep.add(step)
    
    # Always keep the most recent checkpoint
    if most_recent_step is not None:
        steps_to_keep.add(most_recent_step)
    
    # Delete checkpoints not in keep list
    for step in list(checkpoint_history.keys()):
        if step not in steps_to_keep:
            # Delete both g_* and do_* files
            g_path = os.path.join(exp_path, f"g_{step:08d}.pth")
            do_path = os.path.join(exp_path, f"do_{step:08d}.pth")
            
            try:
                if os.path.exists(g_path):
                    os.remove(g_path)
                    print(f"Deleted checkpoint: {g_path}")
                if os.path.exists(do_path):
                    os.remove(do_path)
                    print(f"Deleted checkpoint: {do_path}")
                
                # Remove from history
                del checkpoint_history[step]
            except Exception as e:
                print(f"Warning: Failed to delete checkpoint at step {step}: {e}")
    
    return checkpoint_history


def train(rank, args, cfg):
    num_gpus = cfg['env_setting']['num_gpus']
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    batch_size = cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']
    if num_gpus >= 1:
        initialize_process_group(cfg, rank)
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        raise RuntimeError("Mamba needs GPU acceleration")

    generator = SEMamba(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)
    kd_cfg = cfg['training_cfg'].get('kd', {})
    use_kd = bool(kd_cfg.get('enabled', False))
    teacher = load_teacher_model(kd_cfg, device) if use_kd else None

    if rank == 0:
        log_model_info(rank, generator, args.exp_path)
        if use_kd:
            print("Knowledge distillation is ENABLED.")

    state_dict_g, state_dict_do, steps, last_epoch = load_ckpts(args, device)
    if state_dict_g is not None:
        generator.load_state_dict(state_dict_g['generator'], strict=False)
        discriminator.load_state_dict(state_dict_do['discriminator'], strict=False)
    elif use_kd:
        init_student_from_teacher(generator, teacher, kd_cfg, rank)

    if num_gpus > 1 and torch.cuda.is_available():
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    if cfg['training_cfg'].get('use_pretrainedD', False):
        discriminator.load_state_dict( torch.load('ckpts/pretrained_discriminator.pth') )
        print("Loaded pretrained weight from ckpts/pretrained_discriminator.pth.")

    # Create optimizer and schedulers
    optimizers = setup_optimizers((generator, discriminator), cfg)
    load_optimizer_states(optimizers, state_dict_do)
    optim_g, optim_d = optimizers
    scheduler_g, scheduler_d = setup_schedulers(optimizers, cfg, last_epoch)

    # Create trainset and train_loader
    trainset = create_dataset(cfg, train=True, split=True, device=device)
    train_loader = create_dataloader(trainset, cfg, train=True)
    total_train_steps = cfg['training_cfg']['training_epochs'] * max(len(train_loader), 1)
    kd_ramp_ratio = float(kd_cfg.get('ramp_ratio', 0.0)) if use_kd else 0.0
    kd_ramp_steps = int(total_train_steps * kd_ramp_ratio) if kd_ramp_ratio > 0 else 0
    kd_lambda_out = float(kd_cfg.get('lambda_out', 0.0)) if use_kd else 0.0
    kd_lambda_feat = float(kd_cfg.get('lambda_feat', 0.0)) if use_kd else 0.0
    kd_out_weights = kd_cfg.get('out_weights', {}) if use_kd else {}
    kd_w_mag = float(kd_out_weights.get('mag', 1.0))
    kd_w_pha = float(kd_out_weights.get('pha', 0.3))
    kd_w_com = float(kd_out_weights.get('com', 0.5))
    kd_feature_cfg = kd_cfg.get('feature', {}) if use_kd else {}
    kd_feature_normalize = bool(kd_feature_cfg.get('normalize', True))
    kd_teacher_agg = kd_feature_cfg.get('teacher_agg', 'mean_all')
    kd_freeze_cfg = kd_cfg.get('freeze', {}) if use_kd else {}
    kd_freeze_enabled = bool(kd_freeze_cfg.get('enabled', False))
    kd_freeze_components = kd_freeze_cfg.get('components', [])
    kd_freeze_steps_cfg = kd_freeze_cfg.get('freeze_steps')
    kd_freeze_ratio = float(kd_freeze_cfg.get('freeze_ratio', 0.0)) if kd_freeze_steps_cfg is None else 0.0
    kd_freeze_steps = int(kd_freeze_steps_cfg) if kd_freeze_steps_cfg is not None else int(total_train_steps * kd_freeze_ratio)
    kd_freeze_active = None

    # Create validset and validation_loader if rank is 0
    if rank == 0:
        validset = create_dataset(cfg, train=False, split=False, device=device)
        validation_loader = create_dataloader(validset, cfg, train=False)
        sw = SummaryWriter(os.path.join(args.exp_path, 'logs'))
        if use_kd and kd_freeze_enabled:
            comp_str = ", ".join(kd_freeze_components) if kd_freeze_components else "(none)"
            print(f"KD freeze schedule: components={comp_str}, freeze_steps={kd_freeze_steps}")

    generator.train()
    discriminator.train()
    if teacher is not None:
        teacher.eval()

    best_pesq, best_pesq_step = 0.0, 0
    # Checkpoint management: track checkpoints with their PESQ scores
    checkpoint_history = {}  # {step: {'pesq': score, 'validated': bool}}
    most_recent_ckpt_step = None
    
    for epoch in range(max(0, last_epoch), cfg['training_cfg']['training_epochs']):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
            noisy_mag = torch.autograd.Variable(noisy_mag.to(device, non_blocking=True))
            noisy_pha = torch.autograd.Variable(noisy_pha.to(device, non_blocking=True))
            one_labels = torch.ones(batch_size).to(device, non_blocking=True)

            if use_kd and kd_freeze_enabled and kd_freeze_steps > 0 and kd_freeze_components:
                should_freeze = steps < kd_freeze_steps
                if kd_freeze_active is None or should_freeze != kd_freeze_active:
                    set_named_components_requires_grad(generator, kd_freeze_components, not should_freeze, rank)
                    kd_freeze_active = should_freeze
                    if rank == 0:
                        state = "FROZEN" if should_freeze else "UNFROZEN"
                        print(f"KD freeze update at step {steps}: components are now {state}.")

            if use_kd:
                mag_g, pha_g, com_g, student_features = generator(noisy_mag, noisy_pha, return_features=True)
                with torch.no_grad():
                    mag_t, pha_t, com_t, teacher_features = teacher(noisy_mag, noisy_pha, return_features=True)
            else:
                mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
                student_features = None
                mag_t = pha_t = com_t = None
                teacher_features = None

            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_list_r, audio_list_g = list(clean_audio.cpu().numpy()), list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g, cfg)

            # Discriminator
            # ------------------------------------------------------- #
            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                loss_disc_g = 0
            
            loss_disc_all = loss_disc_r + loss_disc_g
            
            loss_disc_all.backward()
            optim_d.step()
            # ------------------------------------------------------- #
            
            # Generator
            # ------------------------------------------------------- #
            optim_g.zero_grad()

            # Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
            # L2 Magnitude Loss
            loss_mag = F.mse_loss(clean_mag, mag_g)
            # Anti-wrapping Phase Loss
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf
            # L2 Complex Loss
            loss_com = F.mse_loss(clean_com, com_g) * 2
            # Time Loss
            loss_time = F.l1_loss(clean_audio, audio_g)
            # Metric Loss
            metric_g = discriminator(clean_mag, mag_g)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
            # Consistancy Loss
            _, _, rec_com = mag_phase_stft(audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True)
            loss_con = F.mse_loss(com_g, rec_com) * 2

            loss_gen_all = (
                loss_metric * cfg['training_cfg']['loss']['metric'] +
                loss_mag * cfg['training_cfg']['loss']['magnitude'] +
                loss_pha * cfg['training_cfg']['loss']['phase'] +
                loss_com * cfg['training_cfg']['loss']['complex'] +
                loss_time * cfg['training_cfg']['loss']['time'] + 
                loss_con * cfg['training_cfg']['loss']['consistancy']
            )

            loss_kd_out = torch.tensor(0.0, device=device)
            loss_kd_feat = torch.tensor(0.0, device=device)
            kd_scale = 1.0
            if use_kd:
                if kd_ramp_steps > 0:
                    kd_scale = min(float(steps) / float(kd_ramp_steps), 1.0)

                loss_kd_mag = F.mse_loss(mag_g, mag_t)
                loss_kd_pha = F.mse_loss(pha_g, pha_t)
                loss_kd_com = F.mse_loss(com_g, com_t)
                loss_kd_out = kd_w_mag * loss_kd_mag + kd_w_pha * loss_kd_pha + kd_w_com * loss_kd_com

                student_block = student_features['tfmamba_blocks'][-1]
                teacher_blocks = teacher_features['tfmamba_blocks']
                if kd_teacher_agg == 'last2' and len(teacher_blocks) >= 2:
                    teacher_target = torch.stack(teacher_blocks[-2:], dim=0).mean(dim=0)
                else:
                    teacher_target = torch.stack(teacher_blocks, dim=0).mean(dim=0)

                if kd_feature_normalize:
                    student_block = normalize_feature_map(student_block)
                    teacher_target = normalize_feature_map(teacher_target)
                loss_kd_feat = F.mse_loss(student_block, teacher_target)

                loss_gen_all = loss_gen_all + kd_scale * (
                    kd_lambda_out * loss_kd_out + kd_lambda_feat * loss_kd_feat
                )

            loss_gen_all.backward()
            optim_g.step()
            # ------------------------------------------------------- #

            metric_error = loss_metric.item()
            mag_error = loss_mag.item()
            pha_error = loss_pha.item()
            com_error = loss_com.item()
            time_error = loss_time.item()
            con_error = loss_con.item()
            kd_out_error = loss_kd_out.item() if use_kd else 0.0
            kd_feat_error = loss_kd_feat.item() if use_kd else 0.0
            kd_scale_value = kd_scale if use_kd else 0.0

            if rank == 0:
                # STDOUT logging
                if steps % cfg['env_setting']['stdout_interval'] == 0:
                    print(
                        'Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric Loss: {:4.3f}, '
                        'Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, Cons Loss: {:4.3f}, '
                        'KD_out: {:4.3f}, KD_feat: {:4.3f}, KD_scale: {:4.3f}, s/b : {:4.3f}'.format(
                            steps, loss_gen_all, loss_disc_all, metric_error, mag_error, pha_error, com_error, time_error, con_error,
                            kd_out_error, kd_feat_error, kd_scale_value, time.time() - start_b
                        )
                    )

                # Checkpointing
                if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                    exp_name = f"{args.exp_path}/g_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            'generator': (generator.module if num_gpus > 1 else generator).state_dict()
                        }
                    )
                    exp_name = f"{args.exp_path}/do_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            'discriminator': (discriminator.module if num_gpus > 1 else discriminator).state_dict(),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch
                        }
                    )
                    
                    # Track this checkpoint (not yet validated)
                    checkpoint_history[steps] = {'pesq': None, 'validated': False}
                    most_recent_ckpt_step = steps

                # Tensorboard summary logging
                if steps % cfg['env_setting']['summary_interval'] == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistancy Loss", con_error, steps)
                    if use_kd:
                        sw.add_scalar("Training/KD Output Loss", loss_kd_out.item(), steps)
                        sw.add_scalar("Training/KD Feature Loss", loss_kd_feat.item(), steps)
                        sw.add_scalar("Training/KD Ramp Scale", kd_scale, steps)

                # If NaN happend in training period, RaiseError
                if torch.isnan(loss_gen_all).any():
                    raise ValueError("NaN values found in loss_gen_all")

                # Validation
                if steps % cfg['env_setting']['validation_interval'] == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
                            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
                            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
                            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))

                            mag_g, pha_g, com_g = generator(noisy_mag.to(device), noisy_pha.to(device))

                            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
                            audios_r += torch.split(clean_audio, 1, dim=0) # [1, T] * B
                            audios_g += torch.split(audio_g, 1, dim=0)

                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_g, cfg)
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g).item()

                        val_mag_err = val_mag_err_tot / (j+1)
                        val_pha_err = val_pha_err_tot / (j+1)
                        val_com_err = val_com_err_tot / (j+1)
                        val_pesq_score = pesq_score(audios_r, audios_g, cfg).item()
                        print('Steps : {:d}, PESQ Score: {:4.3f}, s/b : {:4.3f}'.
                                format(steps, val_pesq_score, time.time() - start_b))
                        sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)
                        sw.add_scalar("Validation/Phase Loss", val_pha_err, steps)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, steps)

                    generator.train()

                    # Print best validation PESQ score in terminal
                    if val_pesq_score >= best_pesq:
                        best_pesq = val_pesq_score
                        best_pesq_step = steps
                    print(f"valid: PESQ {val_pesq_score}, Mag_loss {val_mag_err}, Phase_loss {val_pha_err}. Best_PESQ: {best_pesq} at step {best_pesq_step}")
                    
                    # Update checkpoint history with PESQ scores for unvalidated checkpoints
                    for ckpt_step in list(checkpoint_history.keys()):
                        if not checkpoint_history[ckpt_step]['validated']:
                            # Assign the current validation PESQ score to recent unvalidated checkpoints
                            checkpoint_history[ckpt_step]['pesq'] = val_pesq_score
                            checkpoint_history[ckpt_step]['validated'] = True
                    
                    # Clean up old checkpoints, keeping best 11 + most recent
                    checkpoint_history = cleanup_checkpoints(
                        checkpoint_history, 
                        most_recent_ckpt_step, 
                        args.exp_path, 
                        keep_best_n=11
                    )

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='RT-SEMamba_KD1')
    parser.add_argument('--config', default='recipes/KD1/KD1.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg['env_setting']['seed']
    num_gpus = cfg['env_setting']['num_gpus']
    available_gpus = torch.cuda.device_count()

    if num_gpus > available_gpus:
        warnings.warn(
            f"Warning: The actual number of available GPUs ({available_gpus}) is less than the .yaml config ({num_gpus}). Auto reset to num_gpu = {available_gpus}",
            UserWarning
        )
        cfg['env_setting']['num_gpus'] = available_gpus
        num_gpus = available_gpus
        time.sleep(5)
        

    initialize_seed(seed)
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, 'config.yaml', args.exp_path)

    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_available_gpus}")
        print_gpu_info(num_available_gpus, cfg)
    else:
        print("CUDA is not available.")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, cfg))
    else:
        train(0, args, cfg)

if __name__ == '__main__':
    main()

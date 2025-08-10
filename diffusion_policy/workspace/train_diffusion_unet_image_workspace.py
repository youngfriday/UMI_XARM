if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.common.lr_decay import param_groups_lrd
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())
        if cfg.training.layer_decay < 1.0:
            assert not cfg.policy.obs_encoder.use_lora
            assert not cfg.policy.obs_encoder.share_rgb_model
            obs_encorder_param_groups = param_groups_lrd(self.model.obs_encoder,
                                                         shape_meta=cfg.shape_meta,
                                                         weight_decay=cfg.optimizer.encoder_weight_decay,
                                                         no_weight_decay_list=self.model.obs_encoder.no_weight_decay(),
                                                         layer_decay=cfg.training.layer_decay)
            count = 0
            for group in obs_encorder_param_groups:
                count += len(group['params'])
            if cfg.policy.obs_encoder.feature_aggregation == 'map':
                obs_encorder_param_groups.extend([{'params': self.model.obs_encoder.attn_pool.parameters()}])
                for _ in self.model.obs_encoder.attn_pool.parameters():
                    count += 1
            print(f'obs_encorder params: {count}')
            param_groups = [{'params': self.model.model.parameters()}]
            param_groups.extend(obs_encorder_param_groups)
        else:
            obs_encorder_lr = cfg.optimizer.lr
            if cfg.policy.obs_encoder.pretrained and not cfg.policy.obs_encoder.use_lora:
                obs_encorder_lr *= cfg.training.encoder_lr_coefficient
                print('==> reduce pretrained obs_encorder\'s lr')
            obs_encorder_params = list()
            for param in self.model.obs_encoder.parameters():
                if param.requires_grad:
                    obs_encorder_params.append(param)
            print(f'obs_encorder params: {len(obs_encorder_params)}')
            param_groups = [
                {'params': self.model.model.parameters()},
                {'params': obs_encorder_params, 'lr': obs_encorder_lr}
            ]
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        if 'encoder_weight_decay' in optimizer_cfg.keys():
            optimizer_cfg.pop('encoder_weight_decay')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with='wandb')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            with open(normalizer_path, 'wb') as f:
                pickle.dump(normalizer, f)

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        
        if cfg.training.use_in_the_wild_val:
            in_the_wild_val_dataloader_dict = {}
            for in_the_wild_type in cfg.training.in_the_wild_type:
                in_the_wild_val_dataloader_dict[in_the_wild_type] = []
                for i, wild_dataset_index in enumerate(os.listdir(os.path.join(cfg.task.in_the_wild_dataset.dataset_path, in_the_wild_type))):
                    wild_dataset_config = copy.deepcopy(cfg.task.in_the_wild_dataset)
                    wild_dataset_config['dataset_path'] = os.path.join(wild_dataset_config['dataset_path'], in_the_wild_type, wild_dataset_index, 'dataset.zarr.zip')
                    in_the_wild_val_dataset = hydra.utils.instantiate(wild_dataset_config)
                    in_the_wild_val_dataloader = DataLoader(in_the_wild_val_dataset, **cfg.val_dataloader)
                    in_the_wild_val_dataloader_dict[in_the_wild_type].append(in_the_wild_val_dataloader)
                    if accelerator.is_main_process:
                        print(f'{i} th {in_the_wild_type} dataset:', len(in_the_wild_val_dataset), f'{i} th {in_the_wild_type} dataloader:', len(in_the_wild_val_dataloader))
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        if cfg.checkpoint.only_save_recent:
            cfg.checkpoint.topk.monitor_key = 'epoch'
        elif cfg.training.use_in_the_wild_val:
            if "unseen_object-unseen_env" in cfg.training.in_the_wild_type:
                assert len(cfg.training.in_the_wild_type) == 3
                cfg.checkpoint.topk.monitor_key = 'wild_val_unseen_object-unseen_env_action_mse_error'
            else:
                assert len(cfg.training.in_the_wild_type) == 1
                cfg.checkpoint.topk.monitor_key = 'wild_val_' + cfg.training.in_the_wild_type[0] + '_action_mse_error'
            assert cfg.training.checkpoint_every >= cfg.training.wild_sample_every and cfg.training.checkpoint_every % cfg.training.wild_sample_every == 0
            assert cfg.training.num_epochs // cfg.training.wild_sample_every <= 30
        else:
            cfg.checkpoint.topk.monitor_key = 'val_action_mse_error'
            assert cfg.training.checkpoint_every >= cfg.training.sample_every and cfg.training.checkpoint_every % cfg.training.sample_every == 0
        cfg.checkpoint.topk.format_str = 'epoch={epoch:04d}-' + cfg.checkpoint.topk.monitor_key + '={' + cfg.checkpoint.topk.monitor_key + ':.5f}.ckpt'
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        # device = torch.device(cfg.training.device)
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        if cfg.training.use_in_the_wild_val:
            for key in in_the_wild_val_dataloader_dict.keys():
                in_the_wild_val_dataloader_list = in_the_wild_val_dataloader_dict[key]
                for i, in_the_wild_val_dataloader in enumerate(in_the_wild_val_dataloader_list):
                    in_the_wild_val_dataloader_list[i] = accelerator.prepare(in_the_wild_val_dataloader)

        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train/loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'train/epoch': self.epoch,
                            'train/lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train/loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                # if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                #     with torch.no_grad():
                #         val_losses = list()
                #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                #                 loss = self.model(batch)
                #                 val_losses.append(loss)
                #                 if (cfg.training.max_val_steps is not None) \
                #                     and batch_idx >= (cfg.training.max_val_steps-1):
                #                     break
                #         if len(val_losses) > 0:
                #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                #             # log epoch average validation loss
                #             step_log['val_loss'] = val_loss
                
                def log_action_mse(step_log, category, pred_action, gt_action):
                    B, T, _ = pred_action.shape
                    pred_action = pred_action.view(B, T, -1, 10)
                    gt_action = gt_action.view(B, T, -1, 10)
                    step_log[f'{category}/action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log[f'{category}_action_mse_error_pos'] = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                    step_log[f'{category}_action_mse_error_rot'] = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
                    step_log[f'{category}_action_mse_error_width'] = torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9])
                
                def cal_action_mse(pred_action, gt_action):
                    B, T, _ = pred_action.shape
                    pred_action = pred_action.view(B, T, -1, 10)
                    gt_action = gt_action.view(B, T, -1, 10)
                    action_mse_error = torch.nn.functional.mse_loss(pred_action, gt_action)
                    action_mse_error_pos = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                    action_mse_error_rot = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
                    action_mse_error_width = torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9])
                    return action_mse_error, action_mse_error_pos, action_mse_error_rot, action_mse_error_width

                if ((self.epoch % cfg.training.wild_sample_every) == 0 or self.epoch == cfg.training.num_epochs - 1) and cfg.training.use_in_the_wild_val:
                    with torch.no_grad():
                        for key in in_the_wild_val_dataloader_dict.keys():
                            in_the_wild_loss_list = []
                            in_the_wild_val_dataloader_list = in_the_wild_val_dataloader_dict[key]
                            for i, in_the_wild_val_dataloader in enumerate(in_the_wild_val_dataloader_list):
                                action_mse_error_list, action_mse_error_pos_list, action_mse_error_rot_list, action_mse_error_width_list = [], [], [], []
                                for val_sampling_batch in tqdm.tqdm(in_the_wild_val_dataloader):
                                    batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                                    gt_action = batch['action']
                                    pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                                    all_preds, all_gt = accelerator.gather_for_metrics((pred_action, gt_action))
                                    action_mse_error, action_mse_error_pos, action_mse_error_rot, action_mse_error_width = \
                                        cal_action_mse(all_preds, all_gt)
                                    action_mse_error_list.append(action_mse_error)
                                    action_mse_error_pos_list.append(action_mse_error_pos)
                                    action_mse_error_rot_list.append(action_mse_error_rot)
                                    action_mse_error_width_list.append(action_mse_error_width)
                                val_action_mse_error = torch.mean(torch.stack(action_mse_error_list)).item()
                                val_action_mse_error_pos = torch.mean(torch.stack(action_mse_error_pos_list)).item()
                                val_action_mse_error_rot = torch.mean(torch.stack(action_mse_error_rot_list)).item()
                                val_action_mse_error_width = torch.mean(torch.stack(action_mse_error_width_list)).item()
                                in_the_wild_loss_list.append({
                                    'val_action_mse_error': val_action_mse_error,
                                    'val_action_mse_error_pos': val_action_mse_error_pos,
                                    'val_action_mse_error_rot': val_action_mse_error_rot,
                                    'val_action_mse_error_width': val_action_mse_error_width
                                })
                                step_log[f'{i}_{key}_val_action_mse_error'] = val_action_mse_error
                                step_log[f'{i}_{key}_val_action_mse_error_pos'] = val_action_mse_error_pos
                                step_log[f'{i}_{key}_val_action_mse_error_rot'] = val_action_mse_error_rot
                                step_log[f'{i}_{key}_val_action_mse_error_width'] = val_action_mse_error_width
                            step_log[f'wild_val/{key}_action_mse_error'] = np.mean([x['val_action_mse_error'] for x in in_the_wild_loss_list])
                            step_log[f'{key}_val_action_mse_error_pos'] = np.mean([x['val_action_mse_error_pos'] for x in in_the_wild_loss_list])
                            step_log[f'{key}_val_action_mse_error_rot'] = np.mean([x['val_action_mse_error_rot'] for x in in_the_wild_loss_list])
                            step_log[f'{key}_val_action_mse_error_width'] = np.mean([x['val_action_mse_error_width'] for x in in_the_wild_loss_list])
                
                # run diffusion sampling on a training batch
                if ((self.epoch % cfg.training.sample_every) == 0 or self.epoch == cfg.training.num_epochs - 1):
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        gt_action = batch['action']
                        pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                        all_preds, all_gt = accelerator.gather_for_metrics((pred_action, gt_action))
                        log_action_mse(step_log, 'train', all_preds, all_gt)

                        if len(val_dataloader) > 0:
                            action_mse_error_list, action_mse_error_pos_list, action_mse_error_rot_list, action_mse_error_width_list = [], [], [], []
                            for val_sampling_batch in tqdm.tqdm(val_dataloader):
                                batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                                gt_action = batch['action']
                                pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                                all_preds, all_gt = accelerator.gather_for_metrics((pred_action, gt_action))
                                action_mse_error, action_mse_error_pos, action_mse_error_rot, action_mse_error_width = \
                                    cal_action_mse(all_preds, all_gt)
                                action_mse_error_list.append(action_mse_error)
                                action_mse_error_pos_list.append(action_mse_error_pos)
                                action_mse_error_rot_list.append(action_mse_error_rot)
                                action_mse_error_width_list.append(action_mse_error_width)
                            step_log['val/action_mse_error'] = torch.mean(torch.stack(action_mse_error_list)).item()
                            step_log['val_action_mse_error_pos'] = torch.mean(torch.stack(action_mse_error_pos_list)).item()
                            step_log['val_action_mse_error_rot'] = torch.mean(torch.stack(action_mse_error_rot_list)).item()
                            step_log['val_action_mse_error_width'] = torch.mean(torch.stack(action_mse_error_width_list)).item()
                            
                        del batch
                        del gt_action
                        del pred_action
                accelerator.wait_for_everyone()
                    
                # checkpoint
                if cfg.checkpoint.only_save_recent:
                    if ((self.epoch % cfg.training.checkpoint_every) == 0 or \
                         self.epoch == cfg.training.num_epochs - 1 or \
                         (cfg.training.num_epochs - self.epoch) in [10, 20, 30, 40, cfg.training.num_epochs // 2]) and accelerator.is_main_process:
                        if_save_ckpt = True
                    else:
                        if_save_ckpt = False
                elif ((self.epoch % cfg.training.checkpoint_every) == 0 or self.epoch == cfg.training.num_epochs - 1) and accelerator.is_main_process:
                    if_save_ckpt = True
                else:
                    if_save_ckpt = False
                if if_save_ckpt:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        if new_key == 'train_epoch':
                            new_key = 'epoch'
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    if not cfg.checkpoint.only_save_recent or (cfg.training.num_epochs - self.epoch) in [10, 20, 30, 40, cfg.training.num_epochs // 2]:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

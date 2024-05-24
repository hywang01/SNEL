import time
import numpy as np
import os
import os.path as osp
import datetime
import shutil
import pickle
from functools import partial
from collections import OrderedDict
import torch
import torch.autograd as autograd
from torch.optim.swa_utils import SWALR
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import logging

from snel.utils import (prepare_device, AverageMeter, MetricMeter, 
                          mkdir_if_missing, load_pretrained_weights, 
                          count_num_param)
from snel.data_loader import (build_dataloader_train, build_dataloader_test)
from snel.modeling import (build_model_training, build_evaluator, 
                             build_optimizer, build_lr_scheduler, build_loss)


def wandb_init(cfg: dict):
    wandb.init(
        project='my_proj',
        group=cfg.exp_group,
        name=cfg.exp_name,
        notes=cfg.exp_desc_notes,
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    

class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        self.mnt_mode = self.cfg.trainers.monitor_mode
        self.mnt_metric = self.cfg.trainers.monitor_metric
        # save logged informations into log dict
        self.train_log = {'epoch': None,
                          self.mnt_metric: None}

        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg)
        self.best_result = -np.inf
        self.early_stop_indicator = False
        self.not_improved_count = 0
        
        self.debugging = cfg.debugging
        if not self.debugging:
            wandb_init(self.cfg)
        
    def build_data_loader(self):
        """Create essential data-related attributes.
        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        if self.cfg.trainers.train_mode:
            if self.cfg.models.peer_training:
                self.train_dataloader, self.train_dataloader_peer, self.val_dataloader, self.val_dataloader_peer = build_dataloader_train(
                    self.cfg, train_mode=True)
            else:
                self.train_dataloader, self.val_dataloader = build_dataloader_train(
                    self.cfg, train_mode=True)
        else:
            self.test_dataloader = build_dataloader_test(
                self.cfg, train_mode=False)
            
    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = build_model_training(cfg)
        if cfg.models.backbone_pretrained:
            load_pretrained_weights(self.model, cfg.models.pretrained_weights_path)
        self.device, device_ids = prepare_device(cfg.n_gpu_use)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg)
        self.sched = build_lr_scheduler(self.optim, cfg)

        
        self.loss = build_loss(cfg.models.loss_name, cfg)
        if self.cfg.models.optim_warmup:
            self.warmup_loss = build_loss(cfg.models.optim_warmup_loss, cfg)
        
        self.register_model("model", self.model, self.optim, self.sched)

        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Detected {device_count} GPUs (use nn.DataParallel)")
        #     self.model = nn.DataParallel(self.model)

    def train(self):
        """Generic training loops"""
        self.iter = 0
        self.start_epoch = 0
        self.max_epoch = self.cfg.models.optim_max_epoch
        early_stop_enabled = self.cfg.trainers.early_stop
        
        #wandb_init(self.cfg)

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            current_result = self.after_epoch()
            self.train_log['epoch'] = self.epoch
            self.train_log[self.mnt_metric] = current_result
            if self.early_stop_indicator:
                break
            if early_stop_enabled:
                self.early_stop_indicator = self.early_stop()
                if self.early_stop_indicator:
                    break
        self.after_train()

    def early_stop(self):
        # evaluate model performance according to configured metric, 
        # save best checkpoint as model_best
        best_checkpoint = False
        early_stop_indicator = False

        try:
            # check whether model performance improved or not, 
            # according to specified metric(mnt_metric)
            improved = (self.mnt_mode == 'min' and self.train_log[self.mnt_metric] <= self.best_result) or \
                (self.mnt_mode == 'max' and self.train_log[self.mnt_metric] >= self.best_result)
        except KeyError:
            self.logger.warning(
                "Warning: Metric '{}' is not found. "
                "Model performance monitoring is disabled.".format(self.mnt_metric))
            self.mnt_mode = 'off'
            improved = False

        if improved:
            self.best_result = self.train_log[self.mnt_metric]
            self.not_improved_count = 0
            best_checkpoint = True
            if self.cfg.trainers.save_best_model:
                self.save_model(epoch=self.epoch,
                                directory=self.cfg.output_dir,
                                is_best=best_checkpoint)
        else:
            self.not_improved_count += 1

        if self.not_improved_count > self.cfg.trainers.early_stop:
            self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                "Training stops.".format(self.early_stop))
            early_stop_indicator = True

        if early_stop_indicator:
            self.early_stop_indicator = early_stop_indicator
            self.save_model(epoch=self.epoch, 
                            directory=self.cfg.output_dir,
                            is_best=best_checkpoint)
        
        return early_stop_indicator
        
    def register_model(self, name="model", model=None, optim=None, sched=None):
        # * Certain models that have complex archs may reqiure this func
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        # * get a string name of the model
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
        self, epoch, directory, 
        is_best=False, val_result=None, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            self.save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                is_early_stop=self.early_stop_indicator,
                model_name=model_name,
            )
            
        print('trained model saved!')

    @staticmethod
    def save_checkpoint(
        state, save_dir,
        is_best=False,
        is_early_stop=False,
        remove_module_from_keys=False,
        model_name=""):
        """Save checkpoint.

        Args:
            state (dict): dictionary.
            save_dir (str): directory to save checkpoint.
            is_best (bool, optional): if True, this checkpoint will be copied and named
                ``model-best.pth.tar``. Default is False.
            remove_module_from_keys (bool, optional): whether to remove "module."
                from layer names. Default is True.
            model_name (str, optional): model name to save.
        """
        mkdir_if_missing(save_dir)

        if remove_module_from_keys:
            # remove 'module.' in state_dict's keys
            state_dict = state["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            state["state_dict"] = new_state_dict

        # save model
        epoch = state["epoch"]
        if not model_name:
            model_name = "model_epoch" + str(epoch) + ".pth.tar" 
        fpath = osp.join(save_dir, model_name)
        torch.save(state, fpath)
        print(f"Checkpoint saved to {fpath}")

        # save current model name
        checkpoint_file = osp.join(save_dir, "checkpoint")
        checkpoint = open(checkpoint_file, "w+")
        checkpoint.write("{}\n".format(osp.basename(fpath)))
        checkpoint.close()

        if is_best:
            best_fpath = osp.join(osp.dirname(fpath), "model_best.pth.tar")
            shutil.copy(fpath, best_fpath)
            print('Best checkpoint saved to "{}"'.format(best_fpath))
            
        if is_best and is_early_stop:
            best_fpath = osp.join(osp.dirname(fpath), "model_best_earlystop.pth.tar")
            shutil.copy(fpath, best_fpath)
            print('Best checkpoint by early stop saved to "{}"'.format(best_fpath))

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = self.resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch
    
    @staticmethod
    def resume_from_checkpoint(self, fdir, model, optimizer=None, scheduler=None):
        r"""Resume training from a checkpoint.

        This will load (1) model weights and (2) ``state_dict``
        of optimizer if ``optimizer`` is not None.

        Args:
            fdir (str): directory where the model was saved.
            model (nn.Module): model.
            optimizer (Optimizer, optional): an Optimizer.
            scheduler (Scheduler, optional): an Scheduler.

        Returns:
            int: start_epoch.

        Examples::
            >>> fdir = 'log/my_model'
            >>> start_epoch = resume_from_checkpoint(fdir, model, optimizer, scheduler)
        """
        with open(osp.join(fdir, "checkpoint"), "r") as checkpoint:
            model_name = checkpoint.readlines()[0].strip("\n")
            fpath = osp.join(fdir, model_name)

        print('Loading checkpoint from "{}"'.format(fpath))
        checkpoint = self.load_checkpoint(fpath)
        model.load_state_dict(checkpoint["state_dict"])
        print("Loaded model weights")

        if optimizer is not None and "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded optimizer")

        if scheduler is not None and "scheduler" in checkpoint.keys():
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("Loaded scheduler")

        start_epoch = checkpoint["epoch"]
        print("Previous epoch: {}".format(start_epoch))

        return start_epoch

    @staticmethod
    def load_checkpoint(fpath):
        r"""Load checkpoint.

        ``UnicodeDecodeError`` can be well handled, which means
        python2-saved files can be read from python3.

        Args:
            fpath (str): path to checkpoint.

        Returns:
            dict

        Examples::
            >>> fpath = 'log/my_model/model.pth.tar-10'
            >>> checkpoint = load_checkpoint(fpath)
        """
        if fpath is None:
            raise ValueError("File path is None")

        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))

        map_location = None if torch.cuda.is_available() else "cpu"

        try:
            checkpoint = torch.load(fpath, map_location=map_location)

        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )

        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise

        return checkpoint

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model_best.pth.tar"

        if epoch is not None:
            # model_file = "model.pth.tar-" + str(epoch)
            # model_epoch50.pth.tar
            model_file = "model_epoch" + str(epoch) + ".pth.tar"  

        for name in names:
            model_path = osp.join(directory, name, model_file)

            # if not osp.exists(model_path):
            #     raise FileNotFoundError(f"No model at {model_path}")
            if not os.path.exists(model_path):
                try:
                    file_list = os.listdir(osp.join(directory, name))
                    for f_i in file_list:
                        if os.path.splitext(f_i)[-1]  == '.tar':
                            model_path = osp.join(directory, name, f_i)
                            print(f"Find model at {model_path}")
                except FileNotFoundError:
                    print(f"No model at {model_path}")

            checkpoint = self.load_checkpoint(model_path)
            #state_dict = checkpoint["state_dict"]
            
            state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                if k != 'n_averaged':
                    new_key = k.split('module.')[-1]
                    state_dict[new_key] = v
            
            epoch = checkpoint["epoch"]
            # val_result = checkpoint["val_result"]
            # print(
            #     f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            # )
            print(
                f"Load {model_path} to {name} (epoch={epoch})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        '''
        models.train():
        #* enable batch normalization and dropout
        models.eval():
        #* disable batch normalization and dropout
        '''
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval", 'valid']:
                self._models[name].eval()
            else:
                raise KeyError

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]
    
    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss): 
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

         
    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            #self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def before_train(self):
        directory = self.cfg.output_dir
        if self.cfg.resume:
            directory = self.cfg.resume
        self.start_epoch = self.resume_model_if_exist(directory)

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.trainers.test_no_test
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.trainers.train_checkpoint_freq == 0
            if self.cfg.trainers.train_checkpoint_freq > 0 else False
        )
        
        if self.cfg.trainers.early_stop > 0:
            curr_result = self.test(split="val")
        else:
            curr_result = 0

        if do_test and self.cfg.trainers.test_final_model == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.cfg.output_dir,
                    model_name="model-best.pth.tar"
                )
            
        if last_epoch:
            self.save_model(self.epoch, self.cfg.output_dir)
            
        return curr_result

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_dataloader)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_dataloader):
            self.batch = batch
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.trainers.train_print_freq == 0
            only_few_batches = self.num_batches < self.cfg.trainers.train_print_freq
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                #print(" ".join(info))
                self.logger.info(" ".join(info))

            end = time.time()
        if not self.debugging:    
            wandb.log(step=self.epoch, 
                    data={'epoch': self.epoch,
                            'loss': loss_summary["loss"],
                            'lr': self.get_current_lr()
                            })

    @torch.no_grad()
    def test(self, split=None, verbose=False, save_preds=False):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.trainers.test_split

        if split == "val" and self.val_dataloader is not None:
            data_loader = self.val_dataloader
        else:
            data_loader = self.test_dataloader

        print(f"Evaluate on the *{split}* set")

        val_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            batch_loss = self.loss(output, label)
            val_loss += batch_loss.item()
            self.evaluator.process(output, label)
            
        val_loss /= batch_idx+1
        
        results = self.evaluator.evaluate(save_preds=save_preds)
        eval_data_info = self.evaluator.call_label_pred()
        
        # recording val loss
        results['val_loss'] = val_loss
        
        info = []
        info += ['=> per-class result']
        info += ['number of labels: ', str(eval_data_info['number of labels'])]
        info += ['number of predictions: ', str(eval_data_info['number of predictions'])]
        info += ['val loss: ', '{:.4f}'.format(val_loss)]
        
        for class_i, auc_i in results.items():
            info += [class_i + ': ', '{:.4f}'.format(auc_i)]
        
        self.logger.info(" ".join(info))
        
        if not self.debugging:
            wandb.log(results)
        
        if verbose:
            return results
        else:
            return results['average_auc']

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        img = batch["img"]
        label = batch["lab"]

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        return self.model(input)

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def data_anomaly(self, batch):
        image = batch["img"]
        label = batch["lab"]

        image = image.to(self.device)
        label = label.to(self.device)

        return image, label

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        batch_i_img, batch_i_label = self.data_anomaly(self.batch)
        batch_i_output = torch.sigmoid(self.model(batch_i_img))
        batch_i_img_id = self.batch['idx'].numpy()
        
        with autograd.detect_anomaly():
            loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)


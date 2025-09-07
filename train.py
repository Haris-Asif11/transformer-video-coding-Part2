# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from pathlib import Path
from typing import Dict

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed
import wandb
from datamodules.video_data_api import VideoData, VideoDataset
from model_lightning import VCTModule
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor
from torchvision.utils import make_grid
from utils.hydra_tools import OmegaConf


class WandbImageCallback(pl.Callback):
    """
    Log images at end of each validatin step
    """

    def __init__(self, train_batch: VideoData, eval_batch: VideoData):
        super().__init__()
        self.train_batch = train_batch
        self.eval_batch = eval_batch
        self.key_set = ("image", "image_compressed")

    def log_images(
            self, trainer, base_key: str, image_dict: Dict[str, Tensor], global_step: int
    ) -> None:
        for key in self.key_set:
            if image_dict.get(key) is not None:
                log_dict = {
                    f"{base_key}/{key}": wandb.Image(image_dict[key], caption=f"{key}"),
                    "global_step": global_step,
                }
                trainer.logger.experiment.log(log_dict)

    def _compute_input_output_batch(
            self, pl_module: pl.LightningModule, batch: VideoData
    ):
        batch = VideoDataset(
            video_tensor=(batch.video_tensor).to(device=pl_module.device)
        )
        outputs = torch.clip(pl_module(batch)[0], min=0, max=1.0)

        return {
            "image": make_grid(batch.video_tensor[0], nrow=7),
            "image_compressed": make_grid(outputs[0], nrow=7),
        }

    def on_validation_end(self, trainer, pl_module: pl.LightningModule):
        self.log_images(
            trainer,
            "train_images",
            self._compute_input_output_batch(pl_module, self.train_batch),
            pl_module.global_step,
        )
        self.log_images(
            trainer,
            "eval_images",
            self._compute_input_output_batch(pl_module, self.eval_batch),
            pl_module.global_step,
        )


def build_image_logger(data_module: LightningDataModule):
    data_module.setup(stage=None)  # type: ignore

    train_sample = next(iter(data_module.train_dataloader()))
    val_sample = next(iter(data_module.val_dataloader()))
    return WandbImageCallback(train_sample, val_sample)  # type: ignore


@hydra.main(config_path="config", config_name="train_config")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    ######################################################################################
    # Check for saved checkpoints
    save_dir = Path.cwd().absolute() / "checkpoints"
    if (
            not cfg.checkpoint.overwrite
            and not cfg.checkpoint.resume_training
            and len(list(save_dir.glob("*.ckpt"))) > 0
    ):
        raise RuntimeError(
            "Checkpoints detected in save directory: set resume_training=True"
            " to restore trainer state from these checkpoints, or set overwrite=True"
            " to ignore them."
        )

    save_dir.mkdir(exist_ok=True, parents=True)
    last_checkpoint = save_dir / "last.ckpt"
    if last_checkpoint.exists() and cfg.checkpoint.resume_training:
        print(f"Resuming training from last checkpoint = {last_checkpoint}.")
    else:
        print(f"Initialising new model.")
        last_checkpoint = None

    # set up logger
    log_dir = Path.cwd().absolute() / "wandb_logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    # This will create an id based on the logging path
    name = "/".join([Path.cwd().parent.name, Path.cwd().name])
    sha = hashlib.sha256()
    sha.update(str(Path.cwd()).encode())
    wandb_id = sha.hexdigest()

    wandb_logger = WandbLogger(
        name=name,
        save_dir=str(log_dir),
        id=wandb_id,
        config=OmegaConf.to_container(cfg, resolve=True),  # ! resolve=True to load later
        **cfg.logger,
    )
    ######################################################################################

    ### Instantiate dataloader module, model module and set up a wandb watch ###
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, pin_memory=cfg.ngpu != 0
    )
    # isntantiate model outside the PLModule for ease of debugging
    model = hydra.utils.instantiate(cfg.model)
    modelmodule = VCTModule(model, cfg=cfg)

    temp_state_dict = modelmodule.model.state_dict()


    detectron2_weights_complete_path = '/home/ac35anos/PycharmProjects/detectron2_new/output_test_29_integrated_model_final/model_0059999.pth'
    detectron2_weights_complete_checkpoint = torch.load(detectron2_weights_complete_path)


    # Define the prefixes that should match between the two models
    vct_synthesis_prefix = 'synthesis_transform'
    vct_analysis_prefix = 'analysis_transform'
    detectron2_synthesis_prefix = 'elic_synthesis'
    detectron2_analysis_prefix = 'elic_analysis'

    # Function to compare keys between the two state dicts
    def compare_and_replace_keys(vct_dict, detectron2_dict, vct_prefix, detectron2_prefix):
        vct_keys = [k for k in vct_dict.keys() if k.startswith(vct_prefix)]
        detectron2_keys = [k for k in detectron2_dict.keys() if k.startswith(detectron2_prefix)]

        # Create transformed VCT keys (replacing vct_prefix with detectron2_prefix)
        vct_keys_transformed = [k.replace(vct_prefix, detectron2_prefix) for k in vct_keys]

        # Check if all VCT keys have counterparts in Detectron2
        unmatched_vct_keys = set(vct_keys_transformed) - set(detectron2_keys)
        unmatched_detectron2_keys = set(detectron2_keys) - set(vct_keys_transformed)

        if unmatched_vct_keys or unmatched_detectron2_keys:
            print("Unmatched keys found:")
            if unmatched_vct_keys:
                print(f"VCT has unmatched keys (transformed): {unmatched_vct_keys}")
            if unmatched_detectron2_keys:
                print(f"Detectron2 has unmatched keys: {unmatched_detectron2_keys}")
        else:
            print("All keys match between the two models.")

            # Replace the values in VCT with the corresponding values from Detectron2
            for vct_key in vct_keys:
                corresponding_key = vct_key.replace(vct_prefix, detectron2_prefix)
                if corresponding_key in detectron2_dict:
                    vct_dict[vct_key] = detectron2_dict[corresponding_key]
            print("Keys replaced successfully.")
        return vct_dict

    temp_state_dict = compare_and_replace_keys(temp_state_dict, detectron2_weights_complete_checkpoint['model'], vct_synthesis_prefix, detectron2_synthesis_prefix)
    temp_state_dict = compare_and_replace_keys(temp_state_dict, detectron2_weights_complete_checkpoint['model'], vct_analysis_prefix,
                             detectron2_analysis_prefix)

    modified_state_dict = {'model.' + key: value for key, value in temp_state_dict.items()}

    modelmodule.load_state_dict(modified_state_dict)
    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    '''
    val_batch = next(iter(val_loader))
    # print('val_batch shape', val_batch.shape)
    # print(train_batch)
    #modelmodule.test_step(val_batch, 0)
    '''

    # wandb_logger.watch(model=modelmodule.model.bottleneck, log="all", log_freq=100)

    ### Set up trainer and fit the model ###
    image_logger = build_image_logger(datamodule)
    trainer = Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=[
            image_logger,  # LogPredictionsCallback()
            LearningRateMonitor(),
            ModelCheckpoint(**cfg.checkpoint.callback),
        ],
    )

    trainer.fit(
        model=modelmodule,
        datamodule=datamodule,
        ckpt_path=str(last_checkpoint) if last_checkpoint is not None else None,
    )
    if getattr(cfg, "test_datamodule", None) is not None:
        # https://github.com/Lightning-AI/lightning/issues/8375#issuecomment-878739663
        torch.distributed.destroy_process_group()
        print(trainer.global_rank)
        trainer = Trainer(**cfg.tester)
        test_datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg.test_datamodule, pin_memory=cfg.ngpu != 0
        )
        modelmodule.eval()
        trainer.test(modelmodule, datamodule=test_datamodule)


if __name__ == "__main__":
    main()








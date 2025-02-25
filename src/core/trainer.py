import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.hypersim import HypersimDataset
from src.depth_pro.depth_pro import create_model_and_transforms

from src.configs import model_configs as model_configs
from src.configs.training_configs import training_configs
from src.configs.data_configs import data_configs


class DepthEstimationTrainer:

    def __init__(self, model_config: dict, data_config: dict,
                 training_config: dict) -> None:
        """Initialize the depth estimation trainer with model, data, and training configurations.

        Args:
            model_config: Configuration dictionary for the model architecture
            data_config: Configuration dictionary for dataset and data loading
            training_config: Configuration dictionary for training parameters
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config

        # Initialize dataset and dataloader
        if self.data_config["dataset"] == "hypersim":
            dataset_cls = HypersimDataset
        self.train_loader, self.val_loader = self.create_dataloaders(
            dataset_cls)
        # instantiate teacher and student models
        # for teacher checkpoint is being loaded during initialization
        # behavior can be controlled from configs
        self.teacher, _ = create_model_and_transforms(
            model_config.DEFAULT_MONODEPTH_CONFIG_DICT,
            training_config["device"], training_config["precision"])
        self.student, _ = create_model_and_transforms(
            model_config.SMALL_MONODEPTH_CONFIG_DICT,
            training_config["device"], training_config["precision"])

        # Initialize optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler(self.train_loader.__len__())

    def create_dataloaders(self,
                           dataset_cls: type) -> tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders.

        Args:
            dataset_cls: Dataset class to instantiate (e.g., HypersimDataset)

        Returns:
            A tuple containing (train_loader, val_loader)
        """
        train_dataset = dataset_cls(
            root_dir=self.data_config["root_dir"],
            split="train",
            device=self.training_config["device"],
            precision=self.training_config["precision"])
        val_dataset = dataset_cls(root_dir=self.data_config["root_dir"],
                                  split="val",
                                  device=self.training_config["device"],
                                  precision=self.training_config["precision"])

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.data_config['batch_size'],
                                  shuffle=True,
                                  num_workers=self.data_config.get(
                                      'num_workers', 4),
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.data_config['batch_size'],
                                shuffle=False,
                                num_workers=self.data_config.get(
                                    'num_workers', 4),
                                pin_memory=True)

        return train_loader, val_loader

    def create_optimizer(self) -> optim.Adam:
        """Create Adam optimizer for the student model.

        Returns:
            Configured Adam optimizer instance
        """
        return optim.Adam(self.student.parameters(),
                          lr=self.training_config['lr'])

    def create_scheduler(self, total_steps: int) -> LambdaLR:
        """Create a learning rate scheduler with warmup and decay phases.

        Args:
            total_steps: Total number of training steps

        Returns:
            LambdaLR scheduler instance with custom learning rate adjustment
        """

        def lr_lambda(step: int) -> float:
            """Calculate learning rate multiplier based on current step.

            Args:
                step: Current training step

            Returns:
                Learning rate multiplier
            """
            warmup_steps = int(0.01 * total_steps)
            decay_start = int(0.81 * total_steps)

            if step < warmup_steps:
                return step / warmup_steps
            elif step < decay_start:
                return 1.0
            else:
                return 0.1

        return LambdaLR(self.optimizer, lr_lambda)

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model, optimizer, and scheduler states to a checkpoint file.

        Args:
            checkpoint_path: Path where the checkpoint will be saved
        """
        checkpoint = {
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model, optimizer, and scheduler states from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file to load
        """
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.student.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

    def pre_epoch(self, epoch: int) -> None:
        """Preparations before starting an epoch.
        Args:
            epoch: Current epoch number
        """
        # Load student checkpoint if specified and exists
        if self.training_config.get("resume_from_checkpoint"):
            if epoch == 0:  # Only load at the start of training
                self.load_checkpoint(
                    self.training_config["resume_from_checkpoint"])

        # Set models to appropriate mode
        self.teacher.eval()  # Teacher model always in eval mode
        self.student.train()

    def train(self, num_epochs: int) -> None:
        """Run the full training loop.

        Args:
            num_epochs: Number of epochs to train for
        """
        writer = torch.utils.tensorboard.SummaryWriter(
            self.training_config.get("tensorboard_dir",
                                     "runs/depth_distillation"))

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            self.pre_epoch(epoch)
            self.train_epoch(epoch, writer)
            self.after_epoch(epoch, writer)

        writer.close()

    def train_epoch(self, epoch: int,
                    writer: torch.utils.tensorboard.SummaryWriter) -> None:
        """Train for one epoch.

        Args:
            epoch: Current epoch number
            writer: Tensorboard writer instance
        """
        epoch_losses = {'total_loss': 0.0, 'distill_loss': 0.0, 'gt_loss': 0.0}

        for batch_idx, batch in enumerate(self.train_loader):
            batch_results = self.train_batch(batch)

            # Update running losses
            for k in epoch_losses.keys():
                epoch_losses[k] += batch_results[k]

            # Log batch results
            if batch_idx % self.training_config.get("log_interval", 10) == 0:
                step = epoch * len(self.train_loader) + batch_idx
                writer.add_scalar('train/total_loss',
                                  batch_results['total_loss'], step)
                writer.add_scalar('train/distill_loss',
                                  batch_results['distill_loss'], step)
                writer.add_scalar('train/gt_loss', batch_results['gt_loss'],
                                  step)

        # Compute epoch average losses
        num_batches = len(self.train_loader)
        for k in epoch_losses.keys():
            epoch_losses[k] /= num_batches
            writer.add_scalar(f'train/epoch_{k}', epoch_losses[k], epoch)

    def train_batch(self, batch: dict) -> dict:
        """Process a single training batch.

        Args:
            batch: Dictionary containing 'image', 'depth', and 'valid_mask'

        Returns:
            Dictionary containing loss values and predictions
        """
        self.optimizer.zero_grad()

        # Get inputs
        images = batch['image']
        target_depth = batch['depth']
        valid_mask = batch['valid_mask']

        # Forward passes
        with torch.no_grad():
            teacher_pred = self.teacher(images)
        student_pred = self.student(images)

        # Calculate losses
        # Distillation loss (student learning from teacher)
        distill_loss = torch.nn.functional.mse_loss(student_pred * valid_mask,
                                                    teacher_pred * valid_mask)

        # Ground truth loss (student learning from real data)
        gt_loss = torch.nn.functional.l1_loss(student_pred * valid_mask,
                                              target_depth * valid_mask)

        # Combined loss
        total_loss = (
            self.training_config.get("distill_weight", 0.5) * distill_loss +
            self.training_config.get("gt_weight", 0.5) * gt_loss)

        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {
            'total_loss': total_loss.item(),
            'distill_loss': distill_loss.item(),
            'gt_loss': gt_loss.item(),
            'predictions': student_pred.detach()
        }

    def after_epoch(self, epoch: int,
                    writer: torch.utils.tensorboard.SummaryWriter) -> None:
        """Post-epoch operations including validation and checkpointing.

        Args:
            epoch: Current epoch number
            writer: Tensorboard writer instance
        """
        val_loss = self.validate(writer, epoch)

        # Save checkpoint if specified
        if self.training_config.get("save_checkpoint", False):
            if (epoch + 1) % self.training_config.get("checkpoint_interval",
                                                      5) == 0:
                checkpoint_path = os.path.join(
                    self.training_config["checkpoint_dir"],
                    f"checkpoint_epoch_{epoch+1}.pth")
                self.save_checkpoint(checkpoint_path)

        # Save best model if specified
        if self.training_config.get("save_best", False):
            if not hasattr(self,
                           'best_val_loss') or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(
                    self.training_config["checkpoint_dir"], "best_model.pth")
                self.save_checkpoint(best_model_path)

    def validate(self, writer: torch.utils.tensorboard.SummaryWriter,
                 epoch: int) -> float:
        """Run validation and log metrics.

        Args:
            writer: Tensorboard writer instance
            epoch: Current epoch number

        Returns:
            Average validation loss
        """
        self.student.eval()
        metrics = HypersimDataset.get_validation_metrics()
        val_metrics = {k: 0.0 for k in metrics.keys()}
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image']
                target_depth = batch['depth']
                valid_mask = batch['valid_mask']

                predictions = self.student(images)

                # Calculate validation loss
                loss = torch.nn.functional.l1_loss(predictions * valid_mask,
                                                   target_depth * valid_mask)
                val_loss += loss.item()

                # Calculate other metrics
                for name, metric_fn in metrics.items():
                    val_metrics[name] += metric_fn(predictions * valid_mask,
                                                   target_depth *
                                                   valid_mask).item()

        # Average metrics
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        for k in val_metrics.keys():
            val_metrics[k] /= num_batches
            writer.add_scalar(f'val/{k}', val_metrics[k], epoch)

        writer.add_scalar('val/loss', val_loss, epoch)
        return val_loss


if __name__ == "__main__":
    kd_trainer = DepthEstimationTrainer(model_configs, data_configs,
                                        training_configs)

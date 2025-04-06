import os
import torch
import logging
from datetime import datetime

from src.core.trainer import DepthEstimationTrainer
from src.configs import model_configs, data_configs, training_configs


def setup_logging(log_dir: str) -> None:
    """Setup logging configuration.

    Args:
        log_dir: Directory to save log files
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file),
                  logging.StreamHandler()])


def setup_directories(config: dict) -> None:
    """Create necessary directories for training.

    Args:
        config: Training configuration dictionary
    """
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    if "tensorboard_dir" in config:
        os.makedirs(config["tensorboard_dir"], exist_ok=True)


def main():
    # Setup CUDA if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("No GPU available, using CPU")

    # Create directories
    setup_directories(training_configs.training_configs)

    # Setup logging
    setup_logging(training_configs.training_configs["checkpoint_dir"])

    # Log configurations
    logging.info("Training configurations:")
    logging.info(f"Model config: {model_configs}")
    logging.info(f"Data config: {data_configs.data_configs}")
    logging.info(f"Training config: {training_configs.training_configs}")

    try:
        # Initialize trainer
        trainer = DepthEstimationTrainer(model_config=model_configs,
                                         data_config=data_configs.data_configs,
                                         training_config=training_configs.training_configs)

        # Start training
        logging.info("Starting training...")
        trainer.train(num_epochs=training_configs.training_configs["num_epochs"])

        logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

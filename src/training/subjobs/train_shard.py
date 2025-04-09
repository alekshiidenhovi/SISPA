import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from common.logger import logger
from models.resnet import ResNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from datasets.mnist.sharded_dataloader import get_mnist_dataloader


def train_shard(
    accelerator: Accelerator,
    shard_idx: int,
    dataloader: DataLoader,
    num_blocks: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    model_save_dir: str,
):
    """
    Train a ResNet model on a specific shard of data.

    Parameters
    ----------
    shard_idx : int
        Index of the shard to train on
    dataloader : DataLoader
        DataLoader for the shard
    num_blocks : int
        Number of ResNet blocks to use
    hidden_dim : int
        Hidden dimension for the model
    learning_rate : float
        Learning rate for the optimizer
    weight_decay : float
        Weight decay for the optimizer
    epochs : int
        Number of epochs to train for
    device : torch.device
        Device to train on
    model_save_dir : str
        Directory to save the trained model
    """

    model = ResNet(num_blocks=num_blocks, hidden_dim=hidden_dim)
    classifier = nn.Linear(hidden_dim, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    model, classifier, optimizer, dataloader = accelerator.prepare(
        model, classifier, optimizer, dataloader
    )

    for epoch in range(epochs):
        model.train()
        classifier.train()
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in progress_bar:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    embeddings = model(images)
                    outputs = classifier(embeddings)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                    _, predicted = torch.max(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    progress_bar.set_postfix(
                        {
                            "loss": total_loss / (progress_bar.n + 1),
                            "acc": 100 * correct / total,
                        }
                    )

        logger.info(
            f"Shard {shard_idx}, Epoch {epoch + 1}: "
            f"Loss: {total_loss / len(dataloader):.4f}, "
            f"Accuracy: {100 * correct / total:.2f}%"
        )

    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(
        model.state_dict(), os.path.join(model_save_dir, f"shard_{shard_idx}_model.pt")
    )
    logger.info(
        f"Model for shard {shard_idx} saved to {os.path.join(model_save_dir, f'shard_{shard_idx}_model.pt')}"
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a ResNet model on a specific shard of data"
    )
    parser.add_argument(
        "--shard_idx", type=int, required=True, help="Index of the shard to train on"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing the dataset"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=3, help="Number of ResNet blocks"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension for the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--epochs", type=int, default=8, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() // 2,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="./models/shards/",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    torch.manual_seed(args.seed)

    dataloaders = get_mnist_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
    )

    train_shard(
        accelerator=accelerator,
        shard_idx=args.shard_idx,
        dataloader=dataloaders[args.shard_idx],
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        model_save_dir=args.model_save_dir,
    )

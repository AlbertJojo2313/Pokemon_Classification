import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import logging
import os
from dataset.dataset_creator import get_dataloaders, get_num_classes, BATCH_SIZE
from models.efficientnetb0 import PokemonClassifierEfficientNetB0

LOG_DIR = ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join(LOG_DIR, "training.log"),
)


class Trainer:
    def __init__(self, region="kanto", batch_size=BATCH_SIZE):
        self.region = region
        self.batch_size = batch_size
        self.train_loader, self.val_loader, self.test_loader, self.idx_to_name = (
            get_dataloaders()
        )
        self.num_classes = get_num_classes()
        self.model = PokemonClassifierEfficientNetB0(num_classes=self.num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )

        # Tracking metrics
        self.best_val_acc = 0.0
        self.best_model_path = ROOT / "outputs" / "checkpoints" / "best_model.pth"
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)

        if not self.logger.hasHandlers():
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

            fh = logging.FileHandler(LOG_DIR / "training.log")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.logger.info("Starting training epoch...")

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                batch_acc = 100 * correct / total
                self.logger.info(
                    f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] - "
                    f"Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%"
                )

        avg_loss = epoch_loss / len(self.train_loader)
        train_acc = 100 * correct / total

        self.logger.info(f"Training - Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")

        return avg_loss, train_acc

    def validate_epoch(self):
        """Validate for one epoch"""
        self.logger.info("Starting validation epoch...")
        self.model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = 100 * correct / total

        self.logger.info(
            f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%"
        )

        return avg_val_loss, val_acc

    def train(self, num_epochs):
        """Main training loop"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info("=" * 60)

        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            self.logger.info("-" * 60)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate_epoch()

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_acc)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(self.best_model_path)
                self.logger.info(
                    f"✓ New best model saved! Validation accuracy: {val_acc:.2f}%"
                )

            # Summary
            self.logger.info(
                f"Epoch {epoch + 1} Summary - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"Best Val Acc: {self.best_val_acc:.2f}%"
            )
            self.logger.info("=" * 60)

        self.logger.info("\n🎉 Training complete!")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        self.logger.info(f"Best model saved at: {self.best_model_path}")

        # Load best model for final evaluation
        self.load_checkpoint(self.best_model_path)
        self.logger.info("\n📊 Final evaluation on test set...")
        self.evaluate_test()

    def evaluate_test(self):
        """Evaluate on test set"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        self.logger.info(f"Test Accuracy: {test_acc:.2f}%")
        return test_acc

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "best_val_acc" in checkpoint:
            self.best_val_acc = checkpoint["best_val_acc"]

        self.logger.info(f"Checkpoint loaded from {path}")

    def plot_metrics(self):
        """Plot training metrics"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Loss plot
            epochs = range(1, len(self.train_losses) + 1)
            ax1.plot(epochs, self.train_losses, "b-", label="Train Loss")
            ax1.plot(epochs, self.val_losses, "r-", label="Val Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training and Validation Loss")
            ax1.legend()
            ax1.grid(True)

            # Accuracy plot
            ax2.plot(epochs, self.val_accuracies, "g-", label="Val Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy (%)")
            ax2.set_title("Validation Accuracy")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plot_path = ROOT / "outputs" / "plots" / "training_metrics.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path)
            self.logger.info(f"Training metrics plot saved to {plot_path}")
            plt.close()

        except ImportError:
            self.logger.warning("Matplotlib not available. Skipping plot generation.")


def main():
    trainer = Trainer(region="kanto", batch_size=BATCH_SIZE)
    trainer.train(num_epochs=10)

    # Plot metrics
    trainer.plot_metrics()

    # Save final model
    final_model_path = (
        ROOT / "outputs" / "checkpoints" / "kanto_efficientnetb0_final.pth"
    )
    trainer.save_checkpoint(final_model_path)


if __name__ == "__main__":
    main()

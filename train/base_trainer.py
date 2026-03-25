import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import logging
from dataset.dataset_creator import get_dataloaders, get_num_classes, BATCH_SIZE
from models.efficientnetb0 import PokemonClassifierEfficientNetB0
from models.efficientnetb2_allreg import PokemonClassifierEfficientNetB2_3reg

LOG_DIR = ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Model Selection ────────────────────────────────────


def _select_model(regions, num_classes):
    if len(regions) == 1:
        return PokemonClassifierEfficientNetB0(
            num_classes=num_classes
        ), "efficientnetb0"
    else:
        return PokemonClassifierEfficientNetB2_3reg(
            num_classes=num_classes
        ), "efficientnetb2"


# ── Trainer ────────────────────────────────────────────


class Trainer:
    def __init__(
        self,
        regions,
        num_epochs=60,
        batch_size=BATCH_SIZE,
        early_stopping_patience=10,
    ):
        self.regions = regions if isinstance(regions, list) else [regions]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience

        # ── Model selection ───────────────────────────────────
        self.num_classes = get_num_classes(self.regions)
        self.model, model_name = _select_model(self.regions, self.num_classes)

        # ── Region tag includes model name ────────────────────
        # e.g. "kanto_efficientnetb0", "kanto_johto_efficientnetb2"
        self.region_tag = f"{'_'.join(self.regions)}_{model_name}"

        # ── Logger ────────────────────────────────────────────
        self.logger = logging.getLogger(f"Trainer_{self.region_tag}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.hasHandlers():
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

            fh = logging.FileHandler(LOG_DIR / f"training_{self.region_tag}.log")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # ── Data ──────────────────────────────────────────────
        self.train_loader, self.val_loader, self.test_loader, self.idx_to_name = (
            get_dataloaders(region=self.regions, batch_size=batch_size)
        )

        # ── Device ────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # ── Loss, optimiser, scheduler ────────────────────────
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )

        # ── Checkpointing ─────────────────────────────────────
        self.best_val_acc = 0.0
        self.best_model_path = (
            ROOT / "outputs" / "checkpoints" / f"best_model_{self.region_tag}.pth"
        )
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Metric history ────────────────────────────────────
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    # ── Epoch methods ──────────────────────────────────────────

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.logger.info("Starting training epoch...")

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
        """Validate for one epoch."""
        self.logger.info("Starting validation epoch...")
        self.model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

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

    # ── Main loop ──────────────────────────────────────────────

    def train(self):
        """Main training loop with early stopping."""
        self.logger.info(f"Starting training for up to {self.num_epochs} epochs")
        self.logger.info(f"Region(s): {', '.join(self.regions)}")
        self.logger.info(f"Model: {self.region_tag.split('_')[-1]}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Early stopping patience: {self.early_stopping_patience}")
        self.logger.info("=" * 60)

        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            self.logger.info("-" * 60)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            self.scheduler.step(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                epochs_no_improve = 0
                self.save_checkpoint(self.best_model_path)
                self.logger.info(
                    f"✓ New best model saved! Validation accuracy: {val_acc:.2f}%"
                )
            else:
                epochs_no_improve += 1
                self.logger.info(
                    f"No improvement for {epochs_no_improve}/{self.early_stopping_patience} epochs"
                )
                if epochs_no_improve >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

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

        self.load_checkpoint(self.best_model_path)
        self.logger.info("\n📊 Final evaluation on test set...")
        self.evaluate_test()

    # ── Evaluation ─────────────────────────────────────────────

    def evaluate_test(self):
        """Evaluate on test set."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        self.logger.info(f"Test Accuracy: {test_acc:.2f}%")
        return test_acc

    # ── Checkpoint helpers ─────────────────────────────────────

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "best_val_acc" in checkpoint:
            self.best_val_acc = checkpoint["best_val_acc"]

        self.logger.info(f"Checkpoint loaded from {path}")

    # ── Plotting ───────────────────────────────────────────────

    def plot_metrics(self):
        """Plot training and validation metrics."""
        try:
            import matplotlib.pyplot as plt

            epochs = range(1, len(self.train_losses) + 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(epochs, self.train_losses, "b-", label="Train Loss")
            ax1.plot(epochs, self.val_losses, "r-", label="Val Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title(f"Loss — {self.region_tag}")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(epochs, self.train_accuracies, "b-", label="Train Accuracy")
            ax2.plot(epochs, self.val_accuracies, "g-", label="Val Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy (%)")
            ax2.set_title(f"Accuracy — {self.region_tag}")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plot_path = ROOT / "outputs" / "plots" / f"metrics_{self.region_tag}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path)
            self.logger.info(f"Plot saved to {plot_path}")
            plt.close()

        except ImportError:
            self.logger.warning("Matplotlib not available. Skipping plot generation.")

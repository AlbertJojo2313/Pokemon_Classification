from base_trainer import Trainer
from dataset.dataset_creator import BATCH_SIZE


def main():
    # --- Trainer
    trainer = Trainer(
        regions=["kanto","johto","hoenn"],
        num_epochs=75,
        batch_size=BATCH_SIZE,
        early_stopping_patience=10
    )
    trainer.train()
    trainer.plot_metrics()

if __name__ == "__main__":
    main()
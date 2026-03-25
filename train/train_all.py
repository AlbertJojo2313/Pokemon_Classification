# Trains the model on all the regions in the dataset (Kanto-Johto-Hoenn-Sinnoh)

from base_trainer import Trainer
from dataset.dataset_creator import BATCH_SIZE

def main():
    trainer = Trainer(
        regions=['kanto', 'johto', 'hoenn', 'sinnoh'],
        num_epochs=100,
        batch_size=BATCH_SIZE,
        early_stopping_patience=15
    )
    trainer.train()
    trainer.plot_metrics()

if __name__ == "__main__":
    main()

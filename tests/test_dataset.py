import pytest
import numpy as np
from PIL import Image
from dataset_creator import PokemonDataset, get_transform


@pytest.fixture
def dummy_dataset(tmp_path):
    classes = ["001_bulbasaur", "002_ivysaur", "003_venusaur"]
    for class_name in classes:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for j in range(6):
            img = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
            img.save(class_dir / f"img{j}.png")
    return PokemonDataset(base_dir=tmp_path)


def test_dataset_length(dummy_dataset):
    assert len(dummy_dataset) == 18  # 3 classes x 6 images


def test_labels_are_integers(dummy_dataset):
    for label in dummy_dataset.labels:
        assert isinstance(label, int)


def test_getitem_tensor_shape(dummy_dataset):
    image, label = dummy_dataset[0]
    assert image.shape == (3, 224, 224)


def test_getitem_returns_correct_types(dummy_dataset):
    image, label = dummy_dataset[0]
    assert isinstance(label, int)


def test_class_to_idx_mapping(dummy_dataset):
    assert "001_bulbasaur" in dummy_dataset.classes_to_idx


def test_train_transform_has_augmentations():
    transform = get_transform(train=True)
    transform_names = [type(t).__name__ for t in transform.transforms]
    assert "HorizontalFlip" in transform_names


def test_val_transform_has_no_augmentations():
    transform = get_transform(train=False)
    transform_names = [type(t).__name__ for t in transform.transforms]
    assert "HorizontalFlip" not in transform_names

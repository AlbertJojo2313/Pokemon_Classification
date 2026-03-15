import pytest
import numpy as np
from PIL import Image
import torch
from dataset.dataset_creator import (
    PokemonDataset,
    get_transform,
    _collect_samples,
    _split_by_class,
    _get_dex_id,
    _in_range,
    get_num_classes,
)

@pytest.fixture
def dummy_data_dir(tmp_path):
    """Create a temporary directory with dummy Pokémon image folders."""
    classes = ["001_Bulbasaur", "002_Ivysaur", "003_Venusaur"]
    for class_name in classes:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for j in range(10):
            img = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
            img.save(class_dir / f"img{j}.png")
    return tmp_path


@pytest.fixture
def dummy_samples(dummy_data_dir):
    """Collect samples from the dummy data directory"""
    samples, idx_to_name  = _collect_samples(dummy_data_dir, dex_range=(1,3))
    return samples, idx_to_name

@pytest.fixture
def dummy_dataset(dummy_samples):
    """Create a Pokemon Dataset from dummy samples"""
    samples, _ =  dummy_samples
    return PokemonDataset(samples, train=False)


#--- Helper functions
def test_get_dex_id_standard():
    assert _get_dex_id("001_Bulbasaur") == 1
 
 
def test_get_dex_id_large():
    assert _get_dex_id("152_Chikorita") == 152
 
 
def test_get_dex_id_invalid():
    with pytest.raises(ValueError):
        _get_dex_id("invalid_folder")
 
 
def test_in_range_true():
    assert _in_range("001_Bulbasaur", (1, 151)) is True
 
 
def test_in_range_false():
    assert _in_range("152_Chikorita", (1, 151)) is False


#--- Sample Collection Test

def test_collect_samples_count(dummy_data_dir):
    """Should collect 10 images per class, 3 classes = 30 Total"""
    samples, idx_to_name = _collect_samples(dummy_data_dir, dex_range=(1,3))
    assert len(samples) == 30

def test_collect_samples_idx_to_name(dummy_data_dir):
    """idx_to_name should map 0→001_Bulbasaur, 1→002_Ivysaur, 2→003_Venusaur."""
    _, idx_to_name = _collect_samples(dummy_data_dir, dex_range=(1, 3))
    assert idx_to_name[0] == "001_Bulbasaur"
    assert idx_to_name[1] == "002_Ivysaur"
    assert idx_to_name[2] == "003_Venusaur"

def test_collect_samples_labels_are_integers(dummy_data_dir):
    samples, _ = _collect_samples(dummy_data_dir, dex_range=(1,3))
    for _, sample in samples:
        assert isinstance(sample, int) # assert if labels are int

def test_collect_samples_no_folders_raises(tmp_path):
    """Empty dir should raise ValueError"""
    with pytest.raises(ValueError):
        _collect_samples(tmp_path,dex_range=(1,151))



def test_split_is_reproducible(dummy_samples):
    """Same seed should produce identical splits."""
    samples, _ = dummy_samples
    train_a, val_a, test_a = _split_by_class(samples, seed=42)
    train_b, val_b, test_b = _split_by_class(samples, seed=42)
    assert train_a == train_b
    assert val_a == val_b
    assert test_a == test_b
 
 
def test_split_different_seeds_differ(dummy_samples):
    """Different seeds should produce different splits."""
    samples, _ = dummy_samples
    train_a, _, _ = _split_by_class(samples, seed=42)
    train_b, _, _ = _split_by_class(samples, seed=99)
    assert train_a != train_b


# ── Dataset tests 
 
 
def test_dataset_length(dummy_dataset, dummy_samples):
    samples, _ = dummy_samples
    assert len(dummy_dataset) == len(samples)
 
 
def test_getitem_tensor_shape(dummy_dataset):
    image, label = dummy_dataset[0]
    assert image.shape == (3, 224, 224)
 
 
def test_getitem_returns_correct_types(dummy_dataset):
    image, label = dummy_dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)
 
 
def test_dataset_invalid_image(tmp_path):
    """Corrupt image file should raise RuntimeError."""
    bad_file = tmp_path / "bad.png"
    bad_file.write_bytes(b"not an image")
    samples = [(str(bad_file), 0)]
    dataset = PokemonDataset(samples=samples, train=False)
    with pytest.raises(RuntimeError):
        dataset[0]

# ── Transform tests 
 
 
def test_train_transform_has_augmentations():
    transform = get_transform(train=True)
    transform_names = [type(t).__name__ for t in transform.transforms]
    assert "HorizontalFlip" in transform_names
 
 
def test_val_transform_no_augmentations():
    transform = get_transform(train=False)
    transform_names = [type(t).__name__ for t in transform.transforms]
    assert "HorizontalFlip" not in transform_names
    assert "ColorJitter" not in transform_names
    assert "Affine" not in transform_names
 
 
def test_transform_output_shape():
    """Transform should resize any image to (3, 224, 224)."""
    transform = get_transform(train=False)
    img = np.uint8(np.random.rand(128, 256, 3) * 255)
    result = transform(image=img)["image"]
    assert result.shape == (3, 224, 224)

# ── get_num_classes tests 
def test_get_num_classes_kanto():
    assert get_num_classes("kanto") == 151
 
 
def test_get_num_classes_multi_region():
    assert get_num_classes(["kanto", "johto"]) == 251
 
 
def test_get_num_classes_all():
    assert get_num_classes("all") == 1025
 
 
def test_get_num_classes_invalid():
    with pytest.raises(ValueError):
        get_num_classes("invalid_region")
 
 
"""
Descarga Fashion-MNIST: 70K imágenes de ropa, 10 clases, 28x28 píxeles.
Hosteado en AWS S3 (confiable).

Clases: 0=camiseta, 1=pantalón, 2=pullover, 3=vestido, 4=campera,
        5=sandalia, 6=camisa, 7=zapatilla, 8=bolso, 9=bota

Uso: python download_fashion_mnist.py
"""

import numpy as np
import requests
import gzip
import struct
import os

DATA_DIR = "data"

# AWS S3 mirror (muy confiable)
BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

FILES = {
    "train_imgs": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_imgs": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_file(url, path):
    if os.path.exists(path):
        print(f"  Ya existe: {path}")
        return
    print(f"  Descargando: {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  OK ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


def parse_mnist_images(path):
    """Lee archivo de imágenes en formato idx3-ubyte.gz"""
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64) / 255.0


def parse_mnist_labels(path):
    """Lee archivo de labels en formato idx1-ubyte.gz"""
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Descargando Fashion-MNIST desde AWS...\n")
    paths = {}
    for key, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        download_file(BASE_URL + filename, path)
        paths[key] = path

    print("\nConvirtiendo a formato .npz...")

    for split, img_key, label_key in [
        ("fashion_train", "train_imgs", "train_labels"),
        ("fashion_test", "test_imgs", "test_labels"),
    ]:
        imgs = parse_mnist_images(paths[img_key])
        labels = parse_mnist_labels(paths[label_key])

        out_path = os.path.join(DATA_DIR, f"{split}.npz")
        np.savez(out_path, images=imgs, labels=labels)
        print(f"  {out_path}: {len(labels)} muestras, {imgs.shape[1]} features")

    # Limpiar .gz
    for path in paths.values():
        os.remove(path)

    print("\n¡Listo!")
    print("  data/fashion_train.npz (60,000 muestras)")
    print("  data/fashion_test.npz  (10,000 muestras)")
    print("\nClases: camiseta, pantalón, pullover, vestido, campera,")
    print("        sandalia, camisa, zapatilla, bolso, bota")


if __name__ == "__main__":
    main()

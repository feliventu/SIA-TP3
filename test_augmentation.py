import numpy as np
import matplotlib.pyplot as plt
import os
try:
    from src.cuda_ops import augment_images
except ImportError:
    import sys
    sys.path.append("src/cuda ops")
    try:
        import cuda_ops
    except ImportError:
        print("Error: compila cuda_ops primero (cd 'src/cuda ops' && make clean && make && make install)")
        exit(1)

def test_augmentation():
    # Cargar una imagen (usaremos np.load si existe fashion o digits en formato npz)
    data_path = "data/digits.csv"
    if not os.path.exists(data_path):
        print(f"No encuentro {data_path}")
        return

    import pandas as pd
    import ast
    df = pd.read_csv(data_path)
    # Tomar la primera imagen
    img_str = df.iloc[0]["image"]
    img_array = np.array(ast.literal_eval(img_str), dtype=np.float32).reshape(1, 784)
    
    # Crear batch de la misma imagen repetida para probar variaciones
    batch_size = 5
    images = np.repeat(img_array, batch_size, axis=0)
    
    # Parámetros fuertes para que se note
    alpha = 36.0 
    sigma = 5.0 
    rot = 15.0 
    scale = 0.15
    
    print(f"Aplicando deformación en GPU (alpha={alpha}, sigma={sigma}, rot={rot})...")
    aug_images = cuda_ops.augment_images(images, alpha, sigma, rot, scale)
    
    # Graficar
    fig, axes = plt.subplots(2, batch_size, figsize=(15, 6))
    
    for i in range(batch_size):
        # Original
        axes[0, i].imshow(images[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        # Deformada
        axes[1, i].imshow(aug_images[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f"Deformada {i+1}")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig("results/test_augmentation.png")
    print("Resultado guardado en results/test_augmentation.png")

if __name__ == "__main__":
    test_augmentation()

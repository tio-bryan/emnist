from matplotlib import pyplot as plt

def show_dataset(
    X, y, label_map, 
    n_rows=8, n_cols=8, start_idx=0, 
    img_shape=(28, 28), transpose=True
):
    
    """
    Plota uma grade de imagens a partir de um dataset.

    Parâmetros
    ----------
    X : array-like
        Dataset de imagens, onde cada linha é uma imagem achatada.
    y : array-like
        Vetor de rótulos correspondentes a X.
    label_map : dict ou list
        Mapeamento de rótulos para nomes legíveis (ex: {0: 'A', 1: 'B', ...}).
    start_idx : int, opcional
        Índice inicial para exibir as imagens. Default é 0.
    n_rows : int, opcional
        Número de linhas na grade. Default é 8.
    n_cols : int, opcional
        Número de colunas na grade. Default é 8.
    img_shape : tuple, opcional
        Formato (altura, largura) da imagem. Default é (28, 28).
    transpose : bool, opcional
        Se True, aplica .T na imagem (útil para EMNIST). Default é True.
    """

    assert start_idx >= 0
    assert start_idx < len(X) - (n_rows * n_cols)
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(14, 14))
    idx_forward = 0

    for i in range(n_rows):
        for j in range(n_cols):
            current_plot = start_idx + idx_forward
            img = X[current_plot].reshape(img_shape)
            if transpose:
                img = img.T
            
            ax[i][j].imshow(img, cmap='gray')
            ax[i][j].set_title(label_map[y[current_plot]])
            ax[i][j].set_axis_off()
            idx_forward += 1

    plt.tight_layout()
    plt.show()
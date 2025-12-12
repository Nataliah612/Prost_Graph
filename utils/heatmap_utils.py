from PIL import Image, ImageEnhance
import numpy as np
import os 

def build_heatmap(coords, scores, patch_size=512, scale_factor=16, save_path=None):
    """
    coords: Nx2 array (x, y)
    scores: N array con valores 0,1,2,3
    Produce heatmap en RGB con colores fijos por clase.
    """

    # Escalar coordenadas
    coords = coords // scale_factor
    patch_size = patch_size // scale_factor

    max_x, max_y = coords[:, 0].max(), coords[:, 1].max()
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()

    width  = max_x - min_x + 1
    height = max_y - min_y + 1

    # Ajustar coordenadas
    shifted_coords = coords.copy()
    shifted_coords[:, 0] -= min_x
    shifted_coords[:, 1] -= min_y

    # Heatmap RGB
    heatmap = np.zeros((height + patch_size, width + patch_size, 3), dtype=np.uint8)

    # Mapa de colores
    color_map = {
        3: (255,   0,   0),   # Rojo
        2: (255, 128,   0),   # Naranja
        1: (255, 255,   0),   # Amarillo
        0: (  0,   0, 255)    # Azul
    }

    # Dibujar los parches según su score en color sólido
    for (x, y), s in zip(shifted_coords, scores):
        color = color_map.get(int(s), (255, 255, 255))  # Blanco si algo raro
        heatmap[y:y+patch_size, x:x+patch_size] = color

    img = Image.fromarray(heatmap, mode="RGB")

    if save_path:
        img.save(save_path)

    return img

def overlay_heatmap_on_image(orig, heatmap, alpha=0.45):
    """Devuelve el overlay del heatmap sobre la imagen original."""
    if heatmap.mode != "RGB":
        heatmap = heatmap.convert("RGB")
    
    heatmap_resized = heatmap.resize(orig.size)
    # Asegurar que el heatmap sea RGB
    
    return Image.blend(orig, heatmap_resized, alpha)

def draw_gt_patches(orig_img, df_patch_class, file_name, mask_folder="/workspace/data/annotation_masks_SICAP", alpha=0.45):
    """
    Carga una máscara de grises, asigna un color por cada valor de gris y la superpone a la imagen original.

    Parámetros
    ----------
    orig_img : PIL.Image
        Imagen original (ya reescalada).
    df_patch_class : DataFrame (NO se usa más, pero lo dejo por compatibilidad)
    file_name : str
        Nombre del archivo correspondiente (para buscar la máscara)
    mask_folder : str
        Carpeta donde están las máscaras en escala de grises.
    alpha : float
        Nivel de transparencia de la máscara coloreada al superponerla.

    Retorna
    -------
    PIL.Image
        Imagen original con máscara coloreada superpuesta.
    """

    # Ruta de la máscara en grises
    mask_path = os.path.join(mask_folder, file_name)#.replace(".jpg", ".png"))

    if not os.path.exists(mask_path):
        print(f"[WARN] No existe la máscara: {mask_path}")
        return orig_img

    # Cargar máscara en modo 'L' (grayscale)
    mask_gray = Image.open(mask_path).convert("L")
    mask_gray = mask_gray.resize(orig_img.size, Image.NEAREST)

    mask_np = np.array(mask_gray)

    # --- DEFINIR COLORES POR VALOR DE GRIS ---
    # Ajusta aquí los colores para cada clase:
    color_map = {
        50:  (255, 255, 0),      # Verde     → Grado 3
        100: (255, 128, 0),    # Amarillo    → Grado 4
        150: (255, 128, 0),    # Naranja     → Grado 4 cribiforme
        200: (255, 0, 0),      # Rojo        → Grado 5
    }

    # Crear imagen RGB coloreada a partir de la máscara
    color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)

    for gray_val, color in color_map.items():
        color_mask[mask_np == gray_val] = color

    color_mask_img = Image.fromarray(color_mask, mode="RGB")

    # Ajustar opacidad (alpha blending)
    color_mask_img = ImageEnhance.Brightness(color_mask_img).enhance(alpha)

    # Superponer sobre imagen original
    overlay = Image.blend(orig_img, color_mask_img, alpha)

    return overlay
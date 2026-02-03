from PIL import Image, ImageEnhance

import numpy as np
import os 
import sys
import matplotlib.cm as cm
import cv2

current_dir  = os.path.abspath(os.getcwd())# Directorio actual
sys.path.append(os.path.dirname(current_dir))
from models.CNN import CNNEmbedder
from models.keep_model import get_keep_ia3_model

def get_model(path):
    if 'graph' in path:
        classification_mode = 'graph'
    elif 'mlp':
        classification_mode = 'mlp'
    elif 'mean':
        classification_mode = 'mean_aggre'
    else: 
        raise ValueError("No se encontró ningún tipo de agregacion válido en 'path'")

    if 'pretrain' in path: 
        pretrain=True
    else: 
        pretrain = False

    # SELECCION DEL MODELO 
    if 'CNN' in path:
        model = CNNEmbedder(classification_mode)
    elif 'KEEP' in path:
        model = get_keep_ia3_model(classification_mode, pretrain)
  
        raise ValueError("No se encontró ningún tipo de modelo válido en ckpt_path")

    return model

def gt_building(wsi_path, green_folder, file_name, factor= 16, dataset = 'SICAP' ): 

    orig = Image.open(wsi_path)
    w, h = orig.size
    new_size = (w // factor, h // factor)
    orig = orig.resize(new_size, Image.Resampling.BICUBIC)
    file_name = file_name.split("/")[-1]
    if dataset == 'SICAP':
        slide_id_png = file_name.replace('.jpg','') + '.png'
    else:
        slide_id_png = file_name
    green_path = os.path.join(green_folder,slide_id_png)
    green = Image.open(green_path)
    green = green.resize(new_size, Image.Resampling.BICUBIC)
    return orig, green, h, w

def build_heatmap_binary(coords, scores, weights,
                          height, width,
                          patch_size=512,
                          scale_factor=16):

    width  = int(width  / scale_factor)
    height = int(height / scale_factor)

    # tamaño real en 20x
    W20 = max(c[0] for c in coords) + patch_size
    H20 = max(c[1] for c in coords) + patch_size

    sx = width  / W20
    sy = height / H20

    masks = np.zeros((height, width), dtype=np.float32)
    m_weights = np.zeros((height, width), dtype=np.float32)

    for (x, y), s, w in zip(coords, scores, weights):

        # bbox en 20x
        x0_20, y0_20 = x, y
        x1_20 = x + patch_size
        y1_20 = y + patch_size

        # escalar a JPG
        x0 = int(x0_20 * sx)
        x1 = int(x1_20 * sx)
        y0 = int(y0_20 * sy)
        y1 = int(y1_20 * sy)

        # clip
        x0 = max(0, min(x0, width  - 1))
        x1 = max(0, min(x1, width))
        y0 = max(0, min(y0, height - 1))
        y1 = max(0, min(y1, height))

        if x0 < x1 and y0 < y1:
            masks[y0:y1, x0:x1] = s
            m_weights[y0:y1, x0:x1] += w.item()

    return masks, m_weights
from matplotlib import cm 

def heatmap_building(h,w, mask, w_mask, grades, slide_id, factor):
    heatmap = np.zeros((h//factor, w//factor, 3), dtype=np.float32)

    for index, pred_name in enumerate(grades):

        class_id = pred_name.item()

        # Máscara binaria
        mask_bin = (mask == class_id).astype(np.float32)

        # Máscara ponderada
        mask_pred = mask_bin * w_mask

        if mask_pred.max() == 0:
            continue

        # Normalizar a [0,1]
        mask_pred = mask_pred / mask_pred.max()

        # --- COLORES ---
        grade_colormaps = {
            0: cm.Blues,    # azul
            1: cm.YlOrBr,   #YlOrBr,   # amarillo
            2: cm.Oranges, # naranja
            3: cm.Reds     # rojo
        }

        color_map_fn = grade_colormaps.get(class_id, cm.Greys)
        color_map = color_map_fn(mask_pred)

        # Quitar alpha
        color_map = color_map[:, :, :3]

        # Fondo negro
        color_map[mask_pred == 0] = 0

        # Aplicar tissue mask
        #color_map[tissue == 0] = 0

        # Acumular (sumar heatmaps)
        heatmap += color_map


    # Clamp y convertir a uint8
    heatmap_join = np.clip(heatmap, 0, 1)
    heatmap_join = (heatmap_join * 255).astype(np.uint8)
    return heatmap_join

def overlay_inside_green_line(
    base_img,
    overlay_img,
    green_thresh=(150, 100, 100)
):
    """
    Sobrepone una imagen sobre otra haciendo transparente todo lo que
    queda fuera de una línea verde cerrada.

    Parámetros:
    - base_image_path: imagen que contiene la línea verde
    - overlay_image_path: imagen a sobreponer
    - output_path: ruta donde se guarda el resultado
    - green_thresh: (g_min, r_max, b_max) para detectar verde
    """

    # --- Cargar imágenes ---
    #base_img = Image.open(base_image_path).convert("RGBA")
    #overlay_img = Image.open(overlay_image_path).convert("RGBA")

    if base_img.size != overlay_img.size:
        raise ValueError("Las imágenes deben tener el mismo tamaño")
    overlay_img = overlay_img.convert("RGBA")
    base_np = np.array(base_img)
    overlay_np = np.array(overlay_img)
    

    # --- Detectar verde ---
    r = base_np[:, :, 0]
    g = base_np[:, :, 1]
    b = base_np[:, :, 2]

    g_min, r_max, b_max = green_thresh

    green_mask = (g > g_min) & (r < r_max) & (b < b_max)

    # --- Rellenar el interior del contorno ---
    mask_uint8 = (green_mask.astype(np.uint8)) * 255

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filled_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(
        filled_mask, contours, -1, 255, thickness=cv2.FILLED
    )

    # --- Aplicar máscara como alpha ---
    overlay_np[:, :, 3] = filled_mask

    # --- Componer imágenes ---
    overlay_masked = Image.fromarray(overlay_np, mode="RGBA")
    #result = Image.alpha_composite(base_img, overlay_masked)

    return overlay_masked

def overlay_heatmap_on_image(orig, heatmap, alpha=0.45):
    """Devuelve el overlay del heatmap sobre la imagen original."""

    # Si heatmap es numpy array → convertir a PIL
    if isinstance(heatmap, np.ndarray):
        heatmap = Image.fromarray(heatmap)

    if heatmap.mode != "RGB":
        heatmap = heatmap.convert("RGB")

    heatmap_resized = heatmap.resize(orig.size)

    return Image.blend(orig.convert("RGB"), heatmap_resized, alpha)


def draw_gt_patches(orig_img, file_name,mask_folder="/workspace/data/annotation_masks_SICAP", alpha=0.45):
    """
    Carga una máscara de grises, asigna un color por cada valor de gris y la superpone a la imagen original.

    Parámetros
    ----------
    orig_img : PIL.Image
        Imagen original (ya reescalada).
    
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
    mask_path = os.path.join(mask_folder, file_name) #.replace(".jpg", ".png"))

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
        50:  (255, 255, 0),    # Amarillo       → Grado 3
        100: (255, 128, 0),    # Naranja        → Grado 4
        150: (255, 128, 0),    # Naranja        → Grado 4 cribiforme
        200: (255, 0, 0),      # Rojo           → Grado 5
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
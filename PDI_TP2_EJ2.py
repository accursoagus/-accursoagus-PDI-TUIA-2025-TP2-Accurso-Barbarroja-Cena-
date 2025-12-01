import cv2
import numpy as np
import matplotlib.pyplot as plt

"""EJERCICIO 2: detección de patentes y caracteres"""

# ---------------------------------------------------------------------------------
# 1: Detección automática de la patente y recorte de la misma:
# ---------------------------------------------------------------------------------

# Preproocesamiento y funciones auxiliares
def preprocesar_imagen(img):
    """Convierte a gris, aplica Sobel y Binarización Otsu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detección de bordes verticales (Sobel) 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    sobelx_abs = cv2.convertScaleAbs(sobelx)
    
    # Binarización automática (Otsu)
    ret, binary = cv2.threshold(sobelx_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def aplicar_morfologia(binary):
    """Aplica Apertura y Clausura."""

    # Apertura: usamos kernel vertical (1, 5) para cortar conexiones finas arriba/abajo.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    binary_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Clausura:
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 3))
    closing = cv2.morphologyEx(binary_open, cv2.MORPH_CLOSE, kernel_close)
    return closing

def ordenar_puntos(pts):
    # Ordena los 4 puntos: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)] 
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)] 
    return rect

def transformar_perspectiva(imagen, pts):
    # Obtiene el rectángulo ordenado y aplica la homografía
    rect = ordenar_puntos(pts)
    (tl, tr, br, bl) = rect
    
    # Calcular ancho y alto del nuevo rectángulo plano
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Puntos destino (un rectángulo perfecto)
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
    
    # Calcular matriz de homografía y aplicar
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(imagen, M, (maxWidth, maxHeight))
    return warped

# Función de extracción:
def extraer_patentes_rectificadas(archivos):
    recortes_rectificados = []
    
    for path in archivos:
        img = cv2.imread(path)
        if img is None: continue
        
        # 1. Preprocesamiento (Mismo que ya tenías y funcionaba)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1))
        ret, binary = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 3))
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        # 2. Contornos y Rectificación
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # A) Filtros rápidos usando el bounding box normal
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(cnt)
            
            if (1.6 < aspect_ratio < 5.0) and (560 < area < 1750):
                roi_test = binary[y:y+h, x:x+w]
                density = cv2.countNonZero(roi_test) / (w * h)
                
                if 0.10 < density < 0.60:
                    # B) ¡CANDIDATO! Ahora usamos el rectángulo rotado para enderezar
                    rect_rotado = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect_rotado)
                    box = np.int32(box)
                    
                    try:
                        # Aplicamos Homografía (Unidad 4)
                        plate_warped = transformar_perspectiva(img, box.reshape(4, 2))
                        
                        # C) Un pequeño padding final por seguridad
                        #plate_warped = cv2.copyMakeBorder(plate_warped, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
                        
                        recortes_rectificados.append(plate_warped)
                    except:
                        pass # Si la geometría falla, ignoramos

    return recortes_rectificados

# Ejecución:
archivos = [f"TP2/Imagenes/img{i:02d}.png" for i in range(1, 13)]       # CAMBIAR SEGÚN RUTA LOCAL DE ARCHIVOS
candidatos_rectificados = extraer_patentes_rectificadas(archivos)

# Visualizar pasos intermedios:
if len(archivos) > 0:
    img_ejemplo = cv2.imread(archivos[0]) 
    # Llamamos a tus funciones individuales para graficar
    paso1 = preprocesar_imagen(img_ejemplo)
    paso2 = aplicar_morfologia(paso1)

    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Pasos Intermedios (Ejemplo: {archivos[0]})", fontsize=12)

    plt.subplot(1, 2, 1)
    plt.title("1. Preprocesamiento (Sobel + Otsu)")
    plt.imshow(paso1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("2. Morfología (Candidatos)")
    plt.imshow(paso2, cmap='gray')
    plt.axis('off')
    plt.show()

# Visualización candidatos obtenidos
if len(candidatos_rectificados) > 0:
    cols = 4
    rows = (len(candidatos_rectificados) // cols) + 1
    plt.figure(figsize=(12, 3 * rows))
    plt.suptitle("Patentes rectificadas (homografía)", fontsize=16)
    
    for i, recorte in enumerate(candidatos_rectificados):
        plt.subplot(rows, cols, i+1)
        # Mostramos en RGB
        plt.imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
else:
    print("No se encontraron candidatos.")

# Se extrayeron algunos recortes que no corresponden a patentes
# Vamos a eliminar falsas detecciones usando nuevamente un filtrado por tamaño y formato
def filtrar_falsos_positivos(lista_recortes):
    patentes_ok = []
    descartes = []
    
    for roi in lista_recortes:
        h, w = roi.shape[:2]
        
        # --- FILTRO 1: GEOMETRÍA FINAL ---
        # Como ya están rectificadas, el ratio w/h es muy confiable.
        ratio = float(w) / h
        
        es_alargada = (2.0 < ratio < 3.15)
        
        # Tamaño mínimo (por si la homografía generó algo diminuto)
        area = w * h
        es_grande = (h > 15 and w > 19) and (650 < area < 3700)
        
        if es_alargada and es_grande:
             # --- FILTRO 2: CONTENIDO (Opcional pero recomendado) ---
             # Convertimos a gris y chequeamos si está "vacía" o "muy llena"
             gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
             _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
             
             # Una patente tiene letras (bordes internos). Un faro suele ser liso o todo brillo.
             # Medimos la "densidad de bordes" en el centro.
             bordes = cv2.Canny(gray, 60, 150)
             densidad_bordes = cv2.countNonZero(bordes) / (w * h)
             
             # Una patente tiene texto -> alta densidad de bordes (> 0.05)
             if densidad_bordes > 0.05:
                 patentes_ok.append(roi)
             else:
                 descartes.append(roi) # Descartado por liso/vacío
        else:
            descartes.append(roi) # Descartado por forma (cuadrado o muy fino)
            
    return patentes_ok, descartes

patentes_finales, basura = filtrar_falsos_positivos(candidatos_rectificados)
print(f"Patentes confirmadas: {len(patentes_finales)}")

# Visualización
if len(patentes_finales) > 0:
    cols = 4
    rows = (len(patentes_finales) // cols) + 1
    plt.figure(figsize=(12, 3 * rows))
    plt.suptitle("Patentes finales", fontsize=16)
    
    for i, img in enumerate(patentes_finales):
        plt.subplot(rows, cols, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# Ver qué descartamos (para asegurarnos que no borramos ninguna buena)
if len(basura) > 0:
    plt.figure(figsize=(12, 2))
    plt.suptitle("Descartes", fontsize=14, color='red')
    for i, img in enumerate(basura[:10]): # Mostrar primeros 10
        plt.subplot(1, 10, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

# Aún quedan dos 'falsos positivos'. Vamos a utilizar el método de perfil de histograma para ver si logramos eliminarlos,
# ya que parecen tener mucha menor variación de color que las patentes.

def calcular_perfil_vertical(img):
    """Auxiliar: Retorna la imagen binaria y el perfil suavizado."""
    # Pasamos a gris y aplcicamos Otsu invertido)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Proyección Vertical
    perfil = np.sum(binary, axis=0) / 255
    
    # Suavizado (para que el conteo de picos no falle con ruido)
    perfil_suave = np.convolve(perfil, np.ones(5)/5, mode='same')
    
    return binary, perfil_suave

def analizar_perfiles(lista_recortes):
    """Visualización"""
    n = min(len(lista_recortes), 12)
    if n == 0: return

    fig, axes = plt.subplots(n, 2, figsize=(12, 2*n))

    for i in range(n):
        binary, perfil = calcular_perfil_vertical(lista_recortes[i])
        
        # Imagen binaria
        axes[i][0].imshow(binary, cmap='gray')
        axes[i][0].axis('off')
        
        # Histograma suavizado 
        perfil_norm = perfil / np.max(perfil) * 100 if np.max(perfil) > 0 else perfil
        axes[i][1].plot(perfil_norm, color='blue', linewidth=2)
        axes[i][1].fill_between(range(len(perfil_norm)), perfil_norm, color='blue', alpha=0.1)
        axes[i][1].set_ylim(0, 110)
        axes[i][1].grid(True, alpha=0.3)
        
        # Referencia visual del umbral de corte (60%)
        umbral_vis = np.max(perfil_norm) * 0.60
        axes[i][1].axhline(umbral_vis, color='red', linestyle='--', linewidth=1)
        
        axes[i][1].set_xticks([])
        axes[i][1].set_yticks([])

    plt.tight_layout()
    plt.show()

def filtrar_por_picos_suavizados(lista_imgs):
    aceptadas = []
    descartadas = []

    for img in lista_imgs:
        # Llamamos a la misma lógica base
        _, perfil_suave = calcular_perfil_vertical(img)

        # Lógica de picos 
        umbral_corte = np.max(perfil_suave) * 0.60
        picos = 0
        ancho_actual = 0
        
        for val in perfil_suave:
            if val > umbral_corte:
                ancho_actual += 1
            else:
                if ancho_actual > 2: 
                    picos += 1
                ancho_actual = 0
        
        if ancho_actual > 2: picos += 1

        if picos >= 2: # Mínimo 2 bloques para aceptarlo
            aceptadas.append(img)
        else:
            descartadas.append(img)

    return aceptadas, descartadas

# Ejecución y visualización:
analizar_perfiles(patentes_finales) 

patentes_ok, descartes_perfil = filtrar_por_picos_suavizados(patentes_finales)
print(f"\nResumen final:\nEntraron: {len(patentes_finales)}\nAceptadas: {len(patentes_ok)}\nDescartadas: {len(descartes_perfil)}")

if len(patentes_ok) > 0:
    cols = 4
    rows = (len(patentes_ok) // cols) + 1
    plt.figure(figsize=(12, 3 * rows))
    plt.suptitle("Patentes confirmadas", fontsize=16, color='green')
    
    for i, img in enumerate(patentes_ok):
        plt.subplot(rows, cols, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if len(descartes_perfil) > 0:
    plt.figure(figsize=(12, 2))
    plt.suptitle("Objetos descartados:", fontsize=16, color='red')
    for i, img in enumerate(descartes_perfil[:5]):
        plt.subplot(1, 5, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()


# ---------------------------------------------------------------------------------
# 2: Extracción automática de caracteres:
# ---------------------------------------------------------------------------------


# Mejoramos la calidad de la imagen, binarizamos, recortamos bordes (con dos métodos) y aplicamos morfología
def paso_preparacion_hd(lista_imgs):
    """Upscaling x2, conversión a gris y normalización."""
    procesadas = []
    for img in lista_imgs:
        # 1. Upscaling (Cúbica)
        hd = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # 2. Gris (Manejo robusto por si ya viene en gris)
        if len(hd.shape) == 3:
            gray = cv2.cvtColor(hd, cv2.COLOR_BGR2GRAY)
        else:
            gray = hd
        # 3. Normalización
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        procesadas.append(norm)
    return procesadas

def paso_binarizacion_hd(lista_hd, block_size=29, c=3):
    """Aplica Threshold Adaptativo con parámetros optimizados para imágenes x2."""
    binarias = []
    for img in lista_hd:
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, block_size, c)
        binarias.append(binary)
    return binarias

def paso_recorte_asimetrico(lista_imgs, p_top=0.10, p_bottom=0.10):
    """Recorta márgenes superior e inferior por porcentaje."""
    recortadas = []
    for img in lista_imgs:
        h = img.shape[0]
        pixel_top = int(h * p_top)
        pixel_bottom = int(h * p_bottom)
        
        # Validación para evitar recortes destructivos
        if pixel_top + pixel_bottom < h:
            recorte = img[pixel_top : h - pixel_bottom, :]
            recortadas.append(recorte)
        else:
            recortadas.append(img)
    return recortadas

def visualizar_grilla(lista_imgs, titulo):
    """Para graficar los resultados"""
    if not lista_imgs: return
    n = len(lista_imgs)
    cols = 4
    rows = (n // cols) + 1
    
    plt.figure(figsize=(12, 3 * rows))
    plt.suptitle(titulo, fontsize=16)
    
    for i, img in enumerate(lista_imgs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Img {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Ejecución y visualización:
if 'patentes_ok' in locals() and len(patentes_ok) > 0:
    
    # PASO 1: Preparación
    print("Ejecutando Paso 1...")
    patentes_hd = paso_preparacion_hd(patentes_ok)
    visualizar_grilla(patentes_hd, "Paso 1: Upscaling + Normalización")

    # PASO 2: Binarización
    print("Ejecutando Paso 2...")
    patentes_binarias = paso_binarizacion_hd(patentes_hd)
    visualizar_grilla(patentes_binarias, "Paso 2: Binarización Adaptativa")

    # PASO 3: Recorte
    print("Ejecutando Paso 3...")
    # Ajustá p_top y p_bottom según tus imágenes
    patentes_recortadas = paso_recorte_asimetrico(patentes_binarias, p_top=0.23, p_bottom=0.05)
    visualizar_grilla(patentes_recortadas, "Paso 3: Recorte Final")

else:
    print("Error: no se encontró la variable 'patentes_ok' o está vacía.")


def paso_recorte_adaptativo_proyeccion(lista_imgs, umbral_corte=4, padding=2):
    """
    Recorta la imagen verticalmente basándose en la proyección horizontal (densidad de píxeles).
    Usa lógica 'Center-Out': busca desde el centro hacia los bordes.
    """
    recortadas = []
    
    for img in lista_imgs:
        h, w = img.shape
        
        # 1. Proyección Horizontal
        # Invertimos para sumar blancos (texto)
        img_inv = cv2.bitwise_not(img)
        proj = np.sum(img_inv, axis=1)
        
        # Normalizamos 0-100%
        max_valor = w * 255
        proj_norm = (proj / max_valor) * 100
        
        # 2. Búsqueda de cortes (Desde el centro)
        mitad = h // 2
        y_min, y_max = 0, h
        
        # Hacia arriba
        for y in range(mitad, 0, -1):
            if proj_norm[y] < umbral_corte:
                y_min = y
                break
                
        # Hacia abajo
        for y in range(mitad, h):
            if proj_norm[y] < umbral_corte:
                y_max = y
                break
        
        # 3. Padding y Validación
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        # Si el recorte colapsa (imagen vacía), devolvemos la original
        if y_max - y_min < 5:
            recortadas.append(img)
        else:
            recortadas.append(img[y_min:y_max, :])
            
    return recortadas

def paso_morfologia_letras(lista_imgs, k_w=2, k_h=2, iters=1, operacion=cv2.MORPH_CLOSE):
    """
    Aplica morfología para mejorar la definición de las letras.
    - MORPH_CLOSE (Clausura): Rellena agujeros dentro de las letras y une roturas.
    - MORPH_OPEN (Apertura): Separa letras que están pegadas por líneas finas.
    """
    procesadas = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    
    for img in lista_imgs:
        # Aplicamos la operación seleccionada
        res = cv2.morphologyEx(img, operacion, kernel, iterations=iters)
        procesadas.append(res)
        
    return procesadas

def visualizar_grilla(lista_imgs, titulo):
    if not lista_imgs: return
    n = len(lista_imgs)
    cols = 4
    rows = (n // cols) + 1
    
    plt.figure(figsize=(12, 3 * rows))
    plt.suptitle(titulo, fontsize=16)
    
    for i, img in enumerate(lista_imgs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Img {i+1} ({img.shape[0]}x{img.shape[1]})") # Muestra tamaño
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Ejecución y visualización:
patentes_finales = paso_recorte_adaptativo_proyeccion(patentes_recortadas, umbral_corte=20, padding=2)
visualizar_grilla(patentes_finales, "Recorte adaptativo")

# Morfología: usamos MORPH_CLOSE para separar caracteres pegados, ya que el fondo es blanco y llos caracteres negros
patentes_separadas = paso_morfologia_letras(patentes_finales, k_w=2, k_h=3, iters=1, operacion=cv2.MORPH_CLOSE)
    
visualizar_grilla(patentes_separadas, "Morfología")

# Segmentación de caracteres:

def etapa_1_clasificacion_inicial(lista_binarias):
    """
    Segmentación inicial con filtros estrictos.
    Clasifica cada patente como 'OK' (6 caracteres) o 'REVISAR'.
    """
    resultados = [] 
    
    for i, img_in in enumerate(lista_binarias):
        # 1. Polaridad (Invertir para que letras sean blancas)
        binary = cv2.bitwise_not(img_in)
        h, w = binary.shape
        area_total = h * w

        # 2. Componentes Conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        cajas_validas = []

        # 3. Filtrado Estricto
        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w_box = stats[label, cv2.CC_STAT_WIDTH]
            h_box = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            # Criterios
            es_alto = (0.60 * h) < h_box < (0.98 * h)
            ratio = w_box / h_box
            es_proporcion = 0.2 < ratio < 1.1
            es_tamano = area > (area_total * 0.02)

            if es_alto and es_proporcion and es_tamano:
                cajas_validas.append((x, y, w_box, h_box))
        
        # Ordenamos de izquierda a derecha
        cajas_validas.sort(key=lambda c: c[0])
        
        # 4. Decisión
        cantidad = len(cajas_validas)
        estado = "OK" if 6 == cantidad else "REVISAR"
        
        resultados.append({
            "id": i,
            "imagen_binaria": binary, # Guardamos la invertida
            "cajas": cajas_validas,
            "estado": estado
        })
        
    return resultados



    """
    Intenta reparar SOLO las patentes marcadas como 'REVISAR'.
    Usa recorte de márgenes y umbrales más permisivos.
    """
    datos_actualizados = datos_etapa_1.copy() # No modificamos la lista original
    
    for item in datos_actualizados:
        if item['estado'] == "OK": continue # Saltamos las que ya están bien

        # 1. PREPARACIÓN (Recorte para romper marcos)
        binary = item['imagen_binaria']
        h, w = binary.shape
        margen_y = int(h * 0.06)
        
        # ROI: Region of Interest (Recorte central)
        roi = binary[margen_y : h-margen_y, :]
        
        # 2. RECOLECCIÓN NUEVA
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
        candidatos_finales = []
        
        # Mediana de anchos para detectar caracteres pegados
        anchos_todos = stats[1:, cv2.CC_STAT_WIDTH]
        mediana_ancho = np.median(anchos_todos) if len(anchos_todos) > 0 else w / 7

        for label in range(1, num):
            wb = stats[label, cv2.CC_STAT_WIDTH]
            hb = stats[label, cv2.CC_STAT_HEIGHT]
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            
            y_real = y + margen_y # Ajustamos coord Y al original
            curr_ratio = wb / hb
            curr_area = wb * hb

            # --- FILTROS PERMISIVOS ---
            
            # A. Borde Lateral (Eliminar marcos verticales)
            if x <= 2 or (x + wb) >= (w - 2): continue

            # B. Esqueletos (Ruido vertical fino)
            if curr_ratio < 0.15: continue

            # C. Área Mínima (Polvo)
            if curr_area < (h * w * 0.02): continue

            # D. Altura (Muy tolerante: 25% a 95%)
            if not ((h * 0.45) < hb < (h * 0.99)): continue

            # --- LÓGICA DE DIVISIÓN (Caracteres pegados) ---
            if (wb > mediana_ancho * 1.8) or (curr_ratio > 1.5):
                # Dividimos en dos mitades iguales
                mitad = wb // 2
                candidatos_finales.append((x, y_real, mitad, hb))
                candidatos_finales.append((x + mitad, y_real, mitad, hb))
            else:
                candidatos_finales.append((x, y_real, wb, hb))

        # Actualizamos el item
        candidatos_finales.sort(key=lambda c: c[0])
        item['cajas'] = candidatos_finales
        item['estado'] = "RECUPERADO" # Cambiamos estado para visualizar después
        item['debug_margen'] = margen_y # Guardamos para graficar la línea de corte

    return datos_actualizados

def etapa_2_recuperacion_estructural(datos_etapa_1):
    """
    Recuperación avanzada:
    1. Limpieza de bordes (Rompehuesos).
    2. División de caracteres pegados.
    3. Inferencia de huecos (Relleno inteligente).
    """
    datos_actualizados = datos_etapa_1.copy()
    
    for item in datos_actualizados:
        if item['estado'] == "OK": continue

        # PASO 1: LIMPIEZA DE MÁRGENES
        # En lugar de recortar la imagen, pintamos de negro los bordes para desconectar letras que tocan el marco (ID 10).
        binary = item['imagen_binaria']
        h, w = binary.shape
        roi_segura = binary.copy()
        
        # Márgenes a borrar
        mx = int(w * 0.05)
        my = int(h * 0.10)
        
        roi_segura[:, :mx] = 0        # Borrar izq
        roi_segura[:, w-mx:] = 0      # Borrar der
        roi_segura[:my, :] = 0        # Borrar techo
        roi_segura[h:, :] = 0         # Borrar piso (no borramos nada del piso finalmente

        # PASO 2: DETECCIÓN Y DIVISIÓN 
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_segura, connectivity=8)
        candidatos_validos = []
        anchos_letras = [] # Para calcular la mediana después
        
        for label in range(1, num):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            wb = stats[label, cv2.CC_STAT_WIDTH]
            hb = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            
            # Filtros mínimos de ruido
            if area < (h * w * 0.015): continue # Muy chico
            if hb < (h * 0.30): continue        # Muy petiso

            # Lógica de división para letras 'pegadas'
            ratio = wb / hb
            # Si es muy ancho (ratio > 0.9) o muy grande respecto a lo esperado
            if ratio > 0.90: 
                # Asumimos 2 caracteres pegados -> Dividimos a la mitad
                mitad = wb // 2
                candidatos_validos.append((x, y, mitad, hb))
                candidatos_validos.append((x + mitad, y, mitad, hb))
                anchos_letras.extend([mitad, mitad])
            else:
                candidatos_validos.append((x, y, wb, hb))
                anchos_letras.append(wb)
        
        candidatos_validos.sort(key=lambda c: c[0])

        # PASO 3: INFERENCIA ESTRUCTURAL (agregar bbox si 'faltan')
        # Solo inferimos si encontramos ALGO (entre 1 y 5 letras)
        if 0 < len(candidatos_validos) < 6:
            
            # Calculamos cuánto mide una letra promedio en esta patente
            ancho_ref = int(np.median(anchos_letras)) if anchos_letras else int(w/7)
            candidatos_finales = []
            
            # A. Relleno Izquierdo (Antes de la primera letra)
            prim_x = candidatos_validos[0][0]
            if prim_x > (ancho_ref * 1.5):
                cuantas_caben = int(prim_x / (ancho_ref * 1.1))
                for k in range(cuantas_caben):
                    # Insertamos caja simulada hacia atrás
                    nx = prim_x - ((k + 1) * int(ancho_ref * 1.1))
                    if nx > 0:
                        # Usamos la Y y H de la primera letra real
                        candidatos_finales.insert(0, (nx, candidatos_validos[0][1], ancho_ref, candidatos_validos[0][3]))

            # B. Relleno Intermedio (Entre letras detectadas)
            for i in range(len(candidatos_validos)):
                curr = candidatos_validos[i]
                candidatos_finales.append(curr)
                
                if i < len(candidatos_validos) - 1:
                    next_x = candidatos_validos[i+1][0]
                    curr_end = curr[0] + curr[2]
                    gap = next_x - curr_end
                    
                    # Si hay un hueco grande (> 80% de una letra)
                    if gap > (ancho_ref * 0.8):
                        cuantas = int(gap / (ancho_ref * 0.95))
                        step = gap / (cuantas + 1)
                        for k in range(cuantas):
                            nx = int(curr_end + (step * (k+1)) - (ancho_ref/2))
                            candidatos_finales.append((nx, curr[1], ancho_ref, curr[3]))

            # C. Relleno Derecho (Después de la última)
            ult_end = candidatos_finales[-1][0] + candidatos_finales[-1][2]
            espacio_der = w - ult_end
            if espacio_der > (ancho_ref * 0.8):
                cuantas = int(espacio_der / (ancho_ref * 1.1))
                for k in range(cuantas):
                    nx = ult_end + (k * int(ancho_ref * 1.1)) + 5
                    if (nx + ancho_ref) < w:
                        candidatos_finales.append((nx, candidatos_finales[-1][1], ancho_ref, candidatos_finales[-1][3]))

            # Actualizamos
            item['cajas'] = candidatos_finales
            item['estado'] = "RECUPERADO_IA"
        
        elif len(candidatos_validos) >= 6:
            # Si encontramos 6 o más sin inferir, genial
            item['cajas'] = candidatos_validos
            item['estado'] = "RECUPERADO"

    return datos_actualizados

def visualizar_etapa_1(resultados):
    """Muestra todas las patentes y su clasificación inicial (Verde=OK, Rojo=REVISAR)."""
    
    n = len(resultados)
    cols = 4
    rows = (n // cols) + 1
    
    plt.figure(figsize=(15, 3 * rows))
    plt.suptitle("ETAPA 1: Segmentación Estándar", fontsize=16)

    for i, item in enumerate(resultados):
        # Convertimos a BGR para dibujar en color
        vis = cv2.cvtColor(item['imagen_binaria'], cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]
        
        # Color del borde según estado
        color = (0, 255, 0) if item['estado'] == "OK" else (255, 0, 0)
        
        # Dibujar Cajas
        for (x, y, wb, hb) in item['cajas']:
            cv2.rectangle(vis, (x, y), (x+wb, y+hb), (0, 255, 0), 1)
            
        plt.subplot(rows, cols, i+1)
        plt.imshow(vis) 
        plt.title(f"ID {item['id']}: {len(item['cajas'])} chars")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualizar_etapa_2(resultados):
    """Muestra SOLO las patentes que pasaron por recuperación."""
    # Filtramos las recuperadas
    recuperadas = [r for r in resultados if r['estado'] == "RECUPERADO"]
    
    if not recuperadas:
        print("No hubo patentes para recuperar en la Etapa 2.")
        return

    n = len(recuperadas)
    cols = 4
    rows = (n // cols) + 1
    
    plt.figure(figsize=(15, 3 * rows))
    plt.suptitle("ETAPA 2: Recuperación Final (Casos Problemáticos)", fontsize=16)
    
    for i, item in enumerate(recuperadas):
        vis = cv2.cvtColor(item['imagen_binaria'], cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]
        margen = item.get('debug_margen', 0)
        
        # Dibujar líneas de corte usadas (Azul)
        if margen > 0:
            cv2.line(vis, (0, margen), (w, margen), (0, 0, 255), 1)
            cv2.line(vis, (0, h-margen), (w, h-margen), (0, 0, 255), 1)
        
        # Dibujar Nuevas Cajas
        for (x, y, wb, hb) in item['cajas']:
            cv2.rectangle(vis, (x, y), (x+wb, y+hb), (0, 255, 0), 1)

        plt.subplot(rows, cols, i+1)
        plt.imshow(vis)
        plt.title(f"ID {item['id']}: {len(item['cajas'])} (Recup)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualizar_y_guardar_letras(datos_etapa_2, guardar=False):
    """
    Toma los datos finales, busca patentes con 6 caracteres y muestra
    cada letra individualmente.
    """
    # Filtramos solo las que tienen 6 caracteres (Patentes Válidas)
    validas = [d for d in datos_etapa_2 if len(d['cajas']) == 6]
    
    if not validas:
        print("No se encontraron patentes con exactamente 6 caracteres.")
        return

    n_filas = len(validas)
    n_cols = 6
    
    plt.figure(figsize=(10, 1.5 * n_filas))
    plt.suptitle(f"Extracción Final: Caracteres Individuales ({len(validas)} patentes)", fontsize=16)

    for i, data in enumerate(validas):
        img_bin = data['imagen_binaria'] 
        # Invertimos color para visualizar mejor
        img_bin = cv2.bitwise_not(img_bin)
        
        for j, (x, y, w, h) in enumerate(data['cajas']):
            # Recorte de la letra
            roi = img_bin[y:y+h, x:x+w]
            
            # Plot
            idx = (i * n_cols) + j + 1
            plt.subplot(n_filas, n_cols, idx)
            plt.imshow(roi, cmap='gray')
            plt.axis('off')
            
            # Etiquetas
            if j == 0: plt.ylabel(f"ID {data['id']}", fontsize=10, rotation=0, labelpad=30, va='center')
            if i == 0: plt.title(f"Char {j+1}")

            # Guardado opcional
            if guardar:
                filename = f"letra_P{data['id']}_C{j}.png"
                cv2.imwrite(filename, roi)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

# Clasificación inicial
print("\n[Etapa 1] Clasificación Inicial...")
datos_1 = etapa_1_clasificacion_inicial(patentes_separadas)
visualizar_etapa_1(datos_1)

# Recuperación estructural
print("\n[Etapa 2] Recuperación Estructural (Rompehuesos + Inferencia)...")
datos_finales = etapa_2_recuperacion_estructural(datos_1)
visualizar_etapa_2(datos_finales)

# Extracción final
print("\n[Final] Extracción de Caracteres...")
visualizar_y_guardar_letras(datos_finales, guardar=False)

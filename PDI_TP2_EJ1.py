import cv2
import numpy as np
import matplotlib.pyplot as plt


# Segmentación de la imágen
def segmentar_objetos(imagen_color):
    # Preprocesamiento

    #Convertimos a escala de grises
    imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
    
    #Eliminamos lo mejor posible el granulado de la imagen
    img_desenfoque = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

    #Detección de Bordes (Los parámetros fueron encontrados por fuerza bruta, es decr, prueba y error)
    img_bordes = cv2.Canny(img_desenfoque, 30, 180)

    #Clausura Morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    #Dilatamos los bordes para intentar que se unan y cerrar los objetos
    img_dilatada = cv2.dilate(img_bordes, kernel, iterations=7)

    #Pasamos la imagen a binario
    _, mascara_binaria = cv2.threshold(img_dilatada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #Detectamos los contornos, solo los exteriores, ignorando huecos internos
    contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plt.imshow(cv2.cvtColor(img_dilatada, cv2.COLOR_BGR2RGB))
    plt.title('Contornos de objetos')
    plt.axis('off')
    plt.show()

    return contornos

def clasificar_objetos(contornos, imagen_original):
    #guardamos información de cada objeto
    monedas_detectadas = []
    dados_detectados = []
    output_visual = imagen_original.copy()

    #Iteramos en cada uno de los contronos, obteniendo su perímetro y área
    for c in contornos:
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)
        
        if area < 500: continue # Filtramos ruido
        
        #Calculamos la "redondes" del objeto
        factor_forma = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
        # Obtenemos el rectángulo que encierra el objeto
        x, y, w, h = cv2.boundingRect(c)
        
        # Si es muy redondo (factor_forma > 0.80), es moneda. Si no, es dado.
        if factor_forma > 0.80:
            tipo, color = "Moneda", (0, 255, 0)
            monedas_detectadas.append({'area': area, 'bbox': (x, y, w, h)})
        else:
            tipo, color = "Dado", (255, 0, 0)
            dados_detectados.append({'bbox': (x, y, w, h), 'contour': c})
            
        cv2.rectangle(output_visual, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_visual, tipo, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    plt.imshow(cv2.cvtColor(output_visual, cv2.COLOR_BGR2RGB))
    plt.title('Objetos detectados')
    plt.axis('off')
    plt.show()

    return monedas_detectadas, dados_detectados, output_visual

# Suma del valor de las monedas
def clasificar_valor_monedas(monedas, imagen_para_dibujar):

    if not monedas: return 0, imagen_para_dibujar

    # Umbrales de medidas de monedas
    UMBRAL_10c = 81000  # Área para monedas de 10 centavos
    UMBRAL_1p = 111000 # Área para monedas de 1 peso
    
    total_dinero = 0
    for moneda in monedas:
        area = moneda['area']
        x, y, w, h = moneda['bbox']

        if area < UMBRAL_10c:
            valor, etiqueta = 0.10, "10 centavos"
        elif area < UMBRAL_1p:
            valor, etiqueta = 1.00, "1 peso"
        else:
            valor, etiqueta = 0.50, "50 centavos"
            
        total_dinero += valor
        cv2.putText(imagen_para_dibujar, etiqueta, (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return total_dinero, imagen_para_dibujar



def sumar_valores_dados(dados, imagen_original, imagen_para_dibujar):
    
    # PARÁMETROS
    UMBRAL_PIP = 135      
    AREA_MIN = 100        
    AREA_MAX = 3500 
    CIRCULARIDAD_MIN = 0.6 


    total_dados = 0
    
    for i, dado in enumerate(dados):
        x, y, w, h = dado['bbox']
        contorno_global = dado['contour']
        
        #Recorte y Máscara

        #Recortamos solo el pedacito de la imagen donde está el dado
        roi = imagen_original[y:y+h, x:x+w]

        # Solo analizaremos lo que está DENTRO del contorno del dado
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        contorno_local = contorno_global - [x, y]  # Ajustamos coordenadas al recorte
        cv2.drawContours(mask, [contorno_local], -1, 255, -1)

        #Aplicamos la máscara: todo lo que no sea dado se vuelve negro puro
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask)
        
        #Detectar Pips
        gray = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2GRAY)

        #Los pips se vuelven BLANCOS, el resto NEGRO
        _, thresh = cv2.threshold(gray, UMBRAL_PIP, 255, cv2.THRESH_BINARY_INV)

        #Nos aseguramos de borrar cualquier ruido fuera de la máscara del dado
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        
        #Limpieza morfológica (Apertura) para quitar ruido pequeño
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thresh_limpio = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Contamos los pips válidos
        # RETR_LIST L para ver lo que hay ADENTRO del borde
        cnts_pips, _ = cv2.findContours(thresh_limpio, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        pips_validos = 0
        img_debug = cv2.cvtColor(thresh_limpio, cv2.COLOR_GRAY2BGR)
        

        for c in cnts_pips:
            # Calculamos métricas de cada "mancha" blanca encontrada
            area = cv2.contourArea(c)
            perimetro = cv2.arcLength(c, True)
            if perimetro == 0: continue
            
            circularidad = (4 * np.pi * area) / (perimetro ** 2)
            
            #Si tiene el tamaño correcto, y la circularidad correcta, es un pip válido
            es_valido = (AREA_MIN < area < AREA_MAX) and (circularidad > CIRCULARIDAD_MIN)

            if es_valido:
                pips_validos += 1
                cv2.drawContours(img_debug, [c], -1, (0, 255, 0), -1) # Verde
            else:
                cv2.drawContours(img_debug, [c], -1, (0, 0, 255), 1)  # Rojo

        total_dados += pips_validos
        
        # Visualización
        plt.figure(figsize=(3, 3))
        plt.imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
        plt.title(f"Dado {i+1}: {pips_validos} pips")
        plt.axis('off')
        plt.show()

        cv2.putText(imagen_para_dibujar, f"Valor: {pips_validos}", (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return total_dados, imagen_para_dibujar


#================================================================================================================


# BLOQUE DE EJECUCIÓN
imagen_color = cv2.imread(r"monedas.jpg")

# Segmentamos para encontrar objetos
contornos = segmentar_objetos(imagen_color)

# Clasificamos entre monedas y dados
monedas, dados, img_clasificada = clasificar_objetos(contornos, imagen_color) 

# Calculamos el valor de las monedas
total_dinero, img_con_valores_monedas = clasificar_valor_monedas(monedas, img_clasificada)

# Calculaos el valor de los dados 
total_dados, img_final = sumar_valores_dados(dados, imagen_color, img_con_valores_monedas)

# Mostramo el resultado final con algunos pasos anteriores
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
plt.title(f'Total: ${total_dinero:.2f} y {total_dados} en dados')
plt.axis('off')
plt.show()



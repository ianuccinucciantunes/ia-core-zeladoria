import cv2
import numpy as np
import torch
import os
import queue
import threading
import json
from datetime import datetime
import gpsd
import base64
import mysql.connector
from sort import Sort  # Certifique-se de que sort.py se encontra no mesmo diretório
from ultralytics import YOLO
import time

# Conecta ao daemon GPSD
gpsd.connect()

# Carregar o modelo
model = YOLO("/home/andre/Documentos/ia/best.pt")

# Inicializa a fila para coordenadas GPS e a captura de vídeo
coord_queue = queue.Queue()
tracker = Sort()
path_video = "/home/andre/Documentos/ia/diadema_short.mp4"
cap = cv2.VideoCapture(0)
output_path = "/home/andre/Documentos/ia/Videos/output_detections_tracking.avi"

# Funções Auxiliares

def get_gps_coordinates():
    """Obtém as coordenadas GPS."""
    packet = gpsd.get_current()
    return (packet.lat, packet.lon)

def distancia_euclidiana(p1, p2):
    """Calcula a distância euclidiana entre dois pontos."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def encode_image_to_base64(image):
    """Converte uma imagem em uma string base64."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Conexão com o banco de dados

def connect_to_db():
    """Conecta-se ao banco de dados MySQL."""
    return mysql.connector.connect(
        host="72.167.52.110",
        user="andrelnucci",
        password="@Andre1020",
        database="data_city"
    )

def insert_detection(db_conn, class_name, confidence, latitude, longitude, detection_image_base64, track_id):
    """Insere uma detecção no banco de dados."""
    cursor = db_conn.cursor()
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    query = """
    INSERT INTO detections (class_name, confidence, latitude, longitude, detection_image_base64, track_id, created_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (class_name, confidence, latitude, longitude, detection_image_base64, track_id, created_at))
    db_conn.commit()

# Configuração de vídeo

def setup_video_writer(cap):
    """Configura o VideoWriter para salvar o vídeo em disco."""
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Definindo FPS fixo
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Funções principais do processamento

def process_frame(frame, gps_coordinates, db_conn):
    """Processa um frame para realizar detecções e tracking."""
    results = model.predict(frame, verbose=False)
    result = results[0]  # Assume-se apenas um frame por vez
    number_detec = len(result.boxes)
    detections_track = []

    for deteccao in result:
        data_deteccao = deteccao.boxes.data
        for i in range(len(data_deteccao)):
            confidence = data_deteccao[i, 4].item()
            class_id = int(data_deteccao[i, 5].item())
            class_name = deteccao.names[class_id]

            x1, y1, x2, y2 = map(int, data_deteccao[i, :4].tolist())
            detection_image = frame[y1:y2, x1:x2]
            detection_image_base64 = encode_image_to_base64(detection_image)

            # Armazenando as informações para tracking
            detections_track.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "image_base64": detection_image_base64
            })

    return detections_track, number_detec

def update_tracking(detections_track):
    """Atualiza o tracker com as novas detecções e retorna os objetos rastreados."""
    deteccoes_rastreadas = np.array([[*det["bbox"], det["confidence"], det["class_id"]] for det in detections_track])
    return tracker.update(deteccoes_rastreadas)

def draw_detections(frame, objetos_rastreados, gps_coordinates, db_conn, detections_track):
    """Desenha as detecções e insere os dados no banco de dados."""
    font_scale = 1.0
    font_thickness = 2

    for obj in objetos_rastreados:
        x1, y1, x2, y2, obj_id = obj
        obj_centro = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Encontrar a detecção correspondente usando a distância euclidiana
        min_distance = float('inf')
        closest_detection = None

        for det in detections_track:
            det_x1, det_y1, det_x2, det_y2 = det["bbox"]
            det_centro = ((det_x1 + det_x2) / 2, (det_y1 + det_y2) / 2)
            distance = distancia_euclidiana(obj_centro, det_centro)

            if distance < min_distance:
                min_distance = distance
                closest_detection = det

        if closest_detection:
            confidence = closest_detection["confidence"]
            class_name = closest_detection["class_name"]
            track_id = int(obj_id)  # Usando obj_id como track_id
            label = f"ID {track_id} {class_name}: {confidence:.2f}"

            # Salvar no banco com o track_id
            insert_detection(db_conn, class_name, confidence, gps_coordinates[0], gps_coordinates[1], closest_detection["image_base64"], track_id)

            # Log de detecção bem-sucedida
            print(f"Detecção bem-sucedida: {label} - Latitude: {gps_coordinates[0]}, Longitude: {gps_coordinates[1]}")

            # Desenhar o retângulo e o texto
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(frame, (int(x1), int(y1) - text_height - baseline), (int(x1) + text_width, int(y1)), (255, 0, 0), -1)
            cv2.putText(frame, label, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        else:
            # Log de falha em encontrar a detecção
            print("Falha ao associar a detecção ao objeto rastreado.")

# Main loop

def main():
    if not cap.isOpened():
        print("Câmera não encontrada. Usando o vídeo de fallback.")
        cap.open(path_video)

    if not cap.isOpened():
        print("Erro ao abrir a câmera ou o arquivo de vídeo.")
        exit()

    out = setup_video_writer(cap)
    db_conn = connect_to_db()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o frame do vídeo.")
            break

        gps_coordinates = get_gps_coordinates()
        detections_track, number_detec = process_frame(frame, gps_coordinates, db_conn)
        
        if number_detec > 0:
            objetos_rastreados = update_tracking(detections_track)
            draw_detections(frame, objetos_rastreados, gps_coordinates, db_conn, detections_track)

        out.write(frame)
        cv2.imshow("Detecta - MSAH", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    db_conn.close()

# Iniciar o script
if __name__ == "__main__":
    main()

import cv2
import numpy as np
import moviepy as mp
import os
import subprocess

# Função para converter o vídeo para MP4 usando moviepy
def converter_para_mp4(video_path, output_path):
    print("\n\t\t-----| 3. converter_para_mp4 |-----")
    video = mp.VideoFileClip(video_path)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Variáveis globais para armazenar as coordenadas do retângulo
x1, y1, x2, y2 = -1, -1, -1, -1
drawing = False

# Função para desenhar o retângulo com o mouse
def draw_rectangle(event, x, y, flags, param):
    print("\n\t\t-----| 5. draw_rectangle |-----")
    global x1, y1, x2, y2, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Quando o botão do mouse é pressionado
        drawing = True
        x1, y1 = x, y  # Começar a posição do retângulo
    
    elif event == cv2.EVENT_MOUSEMOVE:  # Quando o mouse se move
        if drawing:  # Se o botão do mouse está pressionado
            x2, y2 = x, y  # Atualizar a posição do retângulo
    
    elif event == cv2.EVENT_LBUTTONUP:  # Quando o botão do mouse é solto
        drawing = False
        x2, y2 = x, y  # Definir a posição final do retângulo

# Função para detectar a webcam
def detectar_webcam(video_path):
    print("\n\t\t-----| 4. detectar_webcam |-----")
    
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Não foi possível ler o vídeo {video_path}.")
    
    # Mostrar o primeiro frame
    cv2.imshow("Primeiro Frame", frame)
    
    # Definir a função de callback para desenhar o retângulo
    cv2.setMouseCallback("Primeiro Frame", draw_rectangle)
    
    # Esperar até que o usuário desenhe o retângulo e aperte uma tecla
    while True:
        img_copy = frame.copy()
        
        # Desenhar o retângulo na imagem
        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Exibir a imagem com o retângulo desenhado
        cv2.imshow("Primeiro Frame", img_copy)
        
        # Esperar por uma tecla para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressionar 'q' para finalizar
            break
    
    # Fechar a janela
    cv2.destroyAllWindows()
    
    # Verificar se o retângulo foi desenhado
    if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
        # Definir as coordenadas do retângulo
        x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
        cap.release()
        return (x, y, w, h)
    
    cap.release()
    raise ValueError("Webcam não detectada no vídeo.")

# Função para cortar o vídeo e obter a webcam
def cortar_webcam(video_path, output_path, x, y, w, h):
    print("\n\t\t-----| 6. cortar_webcam |-----")
    
    # Certifique-se de que o comando ffmpeg está correto para cortar a área da webcam
    command = [
        "ffmpeg", "-i", video_path, "-vf",
        f"crop={w}:{h}:{x}:{y}", "-c:a", "copy", output_path
    ]
    
    subprocess.run(command, check=True)
    print(f"Vídeo recortado para a webcam: {output_path}")

# Função para redimensionar o vídeo
def redimensionar_video(video_path, output_path):
    print("\n\t\t-----| redimensionar_video |-----")
    
    command = [
        "ffmpeg", "-i", video_path, "-vf",
        "scale=1080:1920", "-c:a", "copy", output_path
    ]
    subprocess.run(command, check=True)

# Função para combinar o vídeo da webcam com o vídeo principal
def combinar_videos(webcam_path, video_path, output_path):
    print("\n\t\t-----| combinar_videos |-----")
    
    video = mp.VideoFileClip(video_path)
    webcam = mp.VideoFileClip(webcam_path).resize(width=1080)
    
    webcam = webcam.set_position((0, 0))
    video = video.set_position((0, 1080 - video.size[1]))
    
    final = mp.CompositeVideoClip([video, webcam])
    final.write_videofile(output_path, fps=30)

# Função principal para processar o vídeo
def processar_video(video_path):
    print("\n\t\t-----| 2. processar_video |-----")
    
    os.makedirs("temp", exist_ok=True)
    nome_base = os.path.basename(video_path).rsplit(".", 1)[0]
    output_final = os.path.join("temp", f"{nome_base} - Shorts.mp4")
    
    # Converter o vídeo para MP4, se necessário
    video_convertido = os.path.join("temp", f"{nome_base}_convertido.mp4")
    converter_para_mp4(video_path, video_convertido)
    
    # Detectar a webcam
    x, y, w, h = detectar_webcam(video_convertido)
    print("\n")
    print(f"Webcam detectada na posição: x={x}, y={y}, largura={w}, altura={h}")
    print("\n")
    
    # Criar o vídeo da webcam com o recorte
    webcam_crop = os.path.join("temp", "webcam.mp4")
    cortar_webcam(video_convertido, webcam_crop, x, y, w, h)
    
    # Redimensionar o vídeo principal
    video_resized = os.path.join("temp", "video_resized.mp4")
    redimensionar_video(video_convertido, video_resized)
    
    # Combinar os vídeos
    combinar_videos(webcam_crop, video_resized, output_final)
    print("\n")
    print(f"Vídeo finalizado com sucesso: {output_final}")
    print("\n")


def processar_videos_na_pasta(pasta):
    print("\n\t\t-----| 1. processar_videos_na_pasta |-----")
    
    for nome_arquivo in os.listdir(pasta):
        caminho_completo = os.path.join(pasta, nome_arquivo)
        
        # Verificar se o arquivo é um vídeo suportado (por exemplo, MKV ou MP4)
        if nome_arquivo.endswith(('.mkv', '.mp4')):
            print("\n")
            print(f"Processando vídeo: {caminho_completo}")
            print("\n")
            processar_video(caminho_completo)

# Exemplo de uso:
pasta_videos = "videos"
processar_videos_na_pasta(pasta_videos)

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
    print("\n\t\t-----| 5. cortar_webcam |-----")
    
    # Verificar se o caminho do vídeo de entrada existe
    print(f"[DEBUG] Verificando se o arquivo de vídeo existe: {video_path}")
    if not os.path.exists(video_path):
        print(f"[ERRO] O arquivo de vídeo {video_path} não foi encontrado.")
        return
    else:
        print(f"[DEBUG] O arquivo de vídeo foi encontrado.")

    # Validar se as coordenadas e dimensões são válidas
    print(f"[DEBUG] Validando as coordenadas e dimensões: x={x}, y={y}, w={w}, h={h}")
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        print("[ERRO] As coordenadas ou dimensões são inválidas.")
        return
    
    # Garantir que o diretório de saída exista
    output_dir = os.path.dirname(output_path)
    print(f"[DEBUG] Verificando se o diretório de saída existe: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"[AVISO] O diretório {output_dir} não existe. Criando...")
        os.makedirs(output_dir)

    try:
        # Carregar o vídeo com o moviepy
        print(f"[DEBUG] Carregando o vídeo: {video_path}")
        video_clip = mp.VideoFileClip(video_path)
        
        # Verificar se o vídeo foi carregado corretamente
        if video_clip is None:
            print("[ERRO] Não foi possível carregar o vídeo.")
            return
        else:
            print(f"[DEBUG] Vídeo carregado com sucesso. Tamanho: {video_clip.size}, FPS: {video_clip.fps}")
        
        # Cortar a área do vídeo
        print(f"[DEBUG] Realizando o corte no vídeo: x={x}, y={y}, w={w}, h={h}")
        # cropped_clip = video_clip.crop(x1=x, y1=y, width=w, height=h)
        clip = video_clip.cropped(x1=x, y1=y, x2=w, y2=h) # Crop the video
        
        # Escrever o vídeo cortado no arquivo de saída
        print(f"[DEBUG] Gravando o vídeo cortado em: {output_path}")
        clip.write_videofile(output_path, codec="libx264")

        print(f"Vídeo recortado e salvo em: {output_path}")

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao cortar o vídeo: {e}")



# Função para redimensionar o vídeo
def redimensionar_video(video_path, output_path):
    print("\n\t\t-----| 6. redimensionar_video |-----")

    # Verifique se o arquivo de vídeo existe
    if not os.path.exists(video_path):
        print(f"[ERRO] O arquivo de vídeo não foi encontrado: {video_path}")
        return
    
    try:
        # Carregar o vídeo com moviepy
        video = mp.VideoFileClip(video_path)
        
        # Redimensionar o vídeo
        video_resized = video.resize(newsize=(1080, 1920))  # A nova dimensão (largura, altura)
        
        # Salvar o vídeo redimensionado
        video_resized.write_videofile(output_path, codec='libx264', audio_codec='aac')

        print(f"[INFO] Vídeo redimensionado com sucesso: {output_path}")

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao redimensionar o vídeo: {e}")

# Função para combinar o vídeo da webcam com o vídeo principal
def combinar_videos(webcam_path, video_path, output_path):
    print("\n\t\t-----| 7. combinar_videos |-----")
    
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
    print(f"Webcam detectada na posição: x={x}, y={y}, largura={w}, altura={h}")
    
    # Criar o vídeo da webcam com o recorte
    webcam_crop = os.path.join("temp", "webcam.mp4")
    cortar_webcam(video_convertido, webcam_crop, x, y, w, h)
    
    # Redimensionar o vídeo principal
    video_resized = os.path.join("temp", "video_resized.mp4")
    redimensionar_video(video_convertido, video_resized)
    
    # Combinar os vídeos
    combinar_videos(webcam_crop, video_resized, output_final)
    print(f"Vídeo finalizado com sucesso: {output_final}")

# Função para processar vídeos na pasta
def processar_videos_na_pasta(pasta):
    print("\n\t\t-----| 1. processar_videos_na_pasta |-----")
    
    for nome_arquivo in os.listdir(pasta):
        caminho_completo = os.path.join(pasta, nome_arquivo)
        
        # Verificar se o arquivo é um vídeo suportado (por exemplo, MKV ou MP4)
        if nome_arquivo.endswith(('.mkv', '.mp4')):
            print(f"Processando vídeo: {caminho_completo}")
            processar_video(caminho_completo)

# Exemplo de uso:
pasta_videos = "videos"
processar_videos_na_pasta(pasta_videos)


# Exemplo de chamada para testar o recorte
# video_path = "videos/alan bug games.mkv"  # Substitua pelo caminho real do seu vídeo
# output_path = "videos/webcam_recortado.mp4"  # Substitua pelo caminho de saída desejado

# # Defina as coordenadas do retângulo a ser cortado (x, y, largura, altura)
# x, y, w, h = detectar_webcam(video_path)
# print(f"Webcam detectada na posição: x={x}, y={y}, largura={w}, altura={h}")

# # Chamar a função para cortar o vídeo
# cortar_webcam(video_path, output_path, x, y, w, h)
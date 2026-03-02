import cv2
import whisper
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, concatenate_videoclips
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import json
from datetime import datetime, timedelta
import librosa
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

class VideoAnalyzer:
    def __init__(self, video_path, output_dir="highlights"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Criar diretório de saída
        os.makedirs(output_dir, exist_ok=True)
        
        # Inicializar modelos de IA local
        print("🤖 Carregando modelos de IA local...")
        self.whisper_model = whisper.load_model("base")  # Modelo pequeno e rápido
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Detector de emoções (local) 
        try:
            self.emotion_classifier = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            print("⚠️ Modelo de emoções não disponível, usando análise básica")
            self.emotion_classifier = None
            
        # Dados de análise
        self.segments = []
        self.highlights = []
        
        print("✅ Modelos carregados com sucesso!")

    def extract_audio_segments(self, segment_duration=30):
        """Extrai segmentos de áudio do vídeo para análise"""
        print("🎵 Extraindo e analisando áudio...")
        
        video = VideoFileClip(self.video_path)
        duration = video.duration
        
        segments = []
        for start_time in range(0, int(duration), segment_duration):
            end_time = min(start_time + segment_duration, duration)
            
            # Extrair segmento de áudio
            audio_segment = video.subclip(start_time, end_time).audio
            
            # Salvar temporariamente
            temp_audio = f"temp_audio_{start_time}.wav"
            audio_segment.write_audiofile(temp_audio, verbose=False, logger=None)
            
            segments.append({
                'start': start_time,
                'end': end_time,
                'audio_file': temp_audio,
                'duration': end_time - start_time
            })
            
        video.close()
        return segments

    def transcribe_segments(self, audio_segments):
        """Transcreve áudio usando Whisper local"""
        print("🗣️ Transcrevendo áudio...")
        
        for i, segment in enumerate(audio_segments):
            print(f"📝 Transcrevendo segmento {i+1}/{len(audio_segments)}")
            
            try:
                # Transcrever com Whisper
                result = self.whisper_model.transcribe(segment['audio_file'])
                segment['transcription'] = result['text']
                segment['language'] = result.get('language', 'pt')
                
                # Limpar arquivo temporário
                os.remove(segment['audio_file'])
                
            except Exception as e:
                print(f"⚠️ Erro na transcrição do segmento {i+1}: {e}")
                segment['transcription'] = ""
                segment['language'] = "pt"
        
        return audio_segments

    def analyze_sentiment_and_emotions(self, segments):
        """Analisa sentimentos e emoções do texto transcrito"""
        print("😄 Analisando sentimentos e emoções...")
        
        for segment in segments:
            text = segment['transcription']
            if not text.strip():
                segment['sentiment_score'] = 0
                segment['emotion'] = 'neutral'
                segment['funny_score'] = 0
                continue
            
            # Análise de sentimento com VADER
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Análise com TextBlob
            blob = TextBlob(text)
            
            # Análise de emoções se disponível
            emotion = 'neutral'
            if self.emotion_classifier:
                try:
                    emotion_result = self.emotion_classifier(text)
                    emotion = emotion_result[0]['label'].lower()
                except:
                    emotion = 'neutral'
            
            # Calcular score de "diversão"
            funny_keywords = ['haha', 'rsrs', 'kkk', 'lol', 'engraçado', 'hilário', 
                            'gargalhada', 'risada', 'piada', 'cômico', 'risos']
            
            funny_score = 0
            text_lower = text.lower()
            for keyword in funny_keywords:
                funny_score += text_lower.count(keyword)
            
            # Score baseado em emoções positivas e intensidade
            sentiment_score = vader_scores['compound'] + blob.sentiment.polarity
            if emotion in ['joy', 'surprise', 'amusement']:
                sentiment_score += 0.3
            
            segment.update({
                'sentiment_score': sentiment_score,
                'emotion': emotion,
                'funny_score': funny_score,
                'vader_scores': vader_scores,
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            })
        
        return segments

    def analyze_visual_features(self, segments):
        """Analisa características visuais como movimento e faces"""
        print("👀 Analisando características visuais...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Carregador de detector de faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        for segment in segments:
            start_frame = int(segment['start'] * fps)
            end_frame = int(segment['end'] * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            face_count = 0
            motion_score = 0
            frame_count = 0
            prev_frame = None
            
            while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detectar faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_count += len(faces)
                
                # Calcular movimento entre frames
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion_score += np.mean(diff)
                
                prev_frame = gray
            
            segment.update({
                'face_density': face_count / max(frame_count, 1),
                'motion_score': motion_score / max(frame_count, 1),
                'visual_activity': (face_count + motion_score) / max(frame_count, 1)
            })
        
        cap.release()
        return segments

    def calculate_highlight_scores(self, segments):
        """Calcula scores finais para identificar highlights"""
        print("⭐ Calculando scores de highlights...")
        
        for segment in segments:
            # Componentes do score
            sentiment_component = max(0, segment['sentiment_score']) * 0.3
            funny_component = min(segment['funny_score'], 5) * 0.2  # Cap em 5
            emotion_component = 0.2 if segment['emotion'] in ['joy', 'surprise', 'amusement'] else 0
            visual_component = min(segment['visual_activity'], 100) / 100 * 0.3  # Normalizar
            
            # Score final
            highlight_score = (sentiment_component + funny_component + 
                             emotion_component + visual_component)
            
            segment['highlight_score'] = highlight_score
            
            # Classificar como highlight se score > threshold
            segment['is_highlight'] = highlight_score > 0.4
        
        return segments

    def extract_highlights(self, segments, min_duration=5, max_highlights=10):
        """Extrai os melhores momentos baseado nos scores"""
        print("✂️ Extraindo highlights...")
        
        # Filtrar e ordenar highlights
        highlights = [s for s in segments if s['is_highlight']]
        highlights.sort(key=lambda x: x['highlight_score'], reverse=True)
        
        # Limitar número de highlights
        highlights = highlights[:max_highlights]
        
        # Expandir duração se muito curto
        video = VideoFileClip(self.video_path)
        clips = []
        
        for i, highlight in enumerate(highlights):
            start = max(0, highlight['start'] - 2)  # 2s antes
            end = min(video.duration, highlight['end'] + 2)  # 2s depois
            
            if end - start < min_duration:
                # Expandir para duração mínima
                center = (start + end) / 2
                start = max(0, center - min_duration/2)
                end = min(video.duration, center + min_duration/2)
            
            clip = video.subclip(start, end)
            clips.append(clip)
            
            print(f"🎬 Highlight {i+1}: {start:.1f}s - {end:.1f}s "
                  f"(Score: {highlight['highlight_score']:.3f})")
        
        video.close()
        return clips, highlights

    def create_highlight_video(self, clips):
        """Cria vídeo compilado com os highlights"""
        if not clips:
            print("❌ Nenhum highlight encontrado")
            return None
            
        print("🎬 Criando vídeo de highlights...")
        
        # Concatenar clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Salvar
        output_path = os.path.join(self.output_dir, f"{self.video_name}_highlights.mp4")
        final_video.write_videofile(
            output_path, 
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        final_video.close()
        print(f"✅ Highlights salvos: {output_path}")
        return output_path

    def save_analysis_report(self, segments):
        """Salva relatório detalhado da análise"""
        report = {
            'video_file': self.video_path,
            'analysis_date': datetime.now().isoformat(),
            'total_segments': len(segments),
            'highlights_found': len([s for s in segments if s['is_highlight']]),
            'segments': segments
        }
        
        report_path = os.path.join(self.output_dir, f"{self.video_name}_analysis.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Relatório salvo: {report_path}")

    def analyze_video(self, segment_duration=30):
        """Executa análise completa do vídeo"""
        print(f"🚀 Iniciando análise de: {self.video_path}")
        
        try:
            # 1. Extrair segmentos de áudio
            audio_segments = self.extract_audio_segments(segment_duration)
            
            # 2. Transcrever áudio
            segments = self.transcribe_segments(audio_segments)
            
            # 3. Analisar sentimentos e emoções
            segments = self.analyze_sentiment_and_emotions(segments)
            
            # 4. Analisar características visuais  
            segments = self.analyze_visual_features(segments)
            
            # 5. Calcular scores de highlights
            segments = self.calculate_highlight_scores(segments)
            
            # 6. Extrair highlights
            clips, highlights = self.extract_highlights(segments)
            
            # 7. Criar vídeo final
            highlight_video = self.create_highlight_video(clips)
            
            # 8. Salvar relatório
            self.save_analysis_report(segments)
            
            print("\n🎉 Análise concluída!")
            print(f"📈 {len(highlights)} highlights encontrados")
            print(f"🎬 Vídeo de highlights: {highlight_video}")
            
            return highlight_video, highlights, segments
            
        except Exception as e:
            print(f"❌ Erro durante análise: {e}")
            return None, [], []

def main():
    # Verificar se existe vídeo na pasta
    videos_dir = "videos"
    if not os.path.exists(videos_dir):
        print("❌ Pasta 'videos' não encontrada")
        return
    
    # Listar vídeos disponíveis
    video_files = [f for f in os.listdir(videos_dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    
    if not video_files:
        print("❌ Nenhum vídeo encontrado na pasta 'videos'")
        return
    
    print("🎥 Vídeos encontrados:")
    for i, video in enumerate(video_files, 1):
        print(f"{i}. {video}")
    
    # Selecionar vídeo (ou usar o primeiro se só houver um)
    if len(video_files) == 1:
        selected_video = video_files[0]
        print(f"📹 Analisando: {selected_video}")
    else:
        try:
            choice = int(input("Escolha um vídeo (número): ")) - 1
            selected_video = video_files[choice]
        except (ValueError, IndexError):
            print("❌ Escolha inválida")
            return
    
    # Executar análise
    video_path = os.path.join(videos_dir, selected_video)
    analyzer = VideoAnalyzer(video_path)
    
    # Configurar duração dos segmentos
    segment_duration = 30  # segundos
    print(f"⚙️ Configuração: segmentos de {segment_duration}s")
    
    # Analisar vídeo
    result = analyzer.analyze_video(segment_duration)
    
    if result[0]:  # Se gerou highlights
        print(f"\n🎯 Melhor momento encontrado!")
        print("💡 Dica: Verifique o arquivo JSON para detalhes da análise")

if __name__ == "__main__":
    main()

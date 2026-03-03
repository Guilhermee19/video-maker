"""
Video Analyzer - Versão Básica (sem MoviePy)
Análise básica de vídeo usando apenas OpenCV e Librosa
"""
import cv2
import numpy as np
import os
import json
from datetime import datetime
import librosa
import matplotlib.pyplot as plt
import signal
import sys
import threading
import time
import signal
import sys
import threading
import time

def convert_numpy_types(obj):
    """Converte tipos numpy para tipos Python nativos para serialização JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class BasicVideoAnalyzer:
    def __init__(self, video_path, output_dir="basic_analysis"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Estado para pause/resume
        self.interrupted = False
        self.current_segments = []
        self.save_interval = 10  # Salvar a cada 10 segmentos
        self.temp_file = os.path.join(output_dir, f"{self.video_name}_temp_progress.json")
        
        os.makedirs(output_dir, exist_ok=True)
        self._setup_signal_handlers()
        print("🔥 Video Analyzer Básico - Sem dependências complexas!")
        print("💡 Pressione Ctrl+C para pausar e salvar progresso")

    def _setup_signal_handlers(self):
        """Configura manipuladores de sinal para pause gracioso"""
        def signal_handler(signum, frame):
            print("\n⚠️  Interrupção detectada! Salvando progresso...")
            self.interrupted = True
            self._save_temp_progress()
            
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _save_temp_progress(self):
        """Salva progresso temporário"""
        if self.current_segments:
            progress_data = {
                'video_path': self.video_path,
                'timestamp': datetime.now().isoformat(),
                'segments_completed': len(self.current_segments),
                'segments': convert_numpy_types(self.current_segments),
                'last_frame_analyzed': getattr(self, 'last_frame_analyzed', 0)
            }
            
            with open(self.temp_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Progresso salvo: {len(self.current_segments)} segmentos analisados")
            print(f"📁 Arquivo temporário: {self.temp_file}")
            print("🔄 Execute novamente para continuar de onde parou")
    
    def _load_temp_progress(self):
        """Carrega progresso salvo anteriormente"""
        if os.path.exists(self.temp_file):
            try:
                with open(self.temp_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                
                if progress_data['video_path'] == self.video_path:
                    self.current_segments = progress_data['segments']
                    last_frame = progress_data.get('last_frame_analyzed', 0)
                    
                    print(f"🔄 Progresso anterior encontrado!")
                    print(f"📊 {len(self.current_segments)} segmentos já analisados")
                    print(f"⏭️  Continuando da posição {last_frame}...")
                    
                    return last_frame
                    
            except Exception as e:
                print(f"⚠️  Erro ao carregar progresso: {e}")
                
        return 0
    
    def _cleanup_temp_files(self):
        """Remove arquivos temporários após conclusão"""
        if os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
                print("🧹 Arquivos temporários removidos")
            except Exception as e:
                print(f"⚠️  Erro ao remover arquivos temporários: {e}")

    def extract_audio_from_video(self):
        """Extrai áudio do vídeo usando OpenCV"""
        print("🎵 Extraindo áudio...")
        
        # Usar FFmpeg através do OpenCV para extrair áudio
        audio_file = f"{self.video_name}_audio.wav"
        
        # Comando para extrair áudio
        import subprocess
        try:
            subprocess.run([
                'ffmpeg', '-i', self.video_path, 
                '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', 
                audio_file, '-y'
            ], check=True, capture_output=True)
            
            print(f"✅ Áudio extraído: {audio_file}")
            return audio_file
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Erro ao extrair áudio: {e}")
            print("💡 Dica: Instale FFmpeg para análise completa de áudio")
            return None

    def analyze_video_without_audio(self, segment_duration=15, start_from_frame=0):
        """Analisa apenas características visuais (sem áudio)"""
        print("👀 Analisando características visuais...")
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("❌ Erro ao abrir vídeo")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📹 Vídeo: {duration:.1f}s, {fps:.1f} FPS, {total_frames} frames")
        
        # Carregar progresso existente se houver
        if start_from_frame == 0:
            start_from_frame = self._load_temp_progress()
        
        segments = self.current_segments.copy()  # Manter segmentos já processados
        frames_per_segment = int(fps * segment_duration)
        
        # Analisar em segmentos a partir do frame inicial  
        start_range = max(start_from_frame, len(segments) * frames_per_segment)
        
        print(f"▶️  Iniciando análise a partir do frame {start_range}")
        
        for start_frame in range(start_range, total_frames, frames_per_segment):
            # Verificar se foi interrompido
            if self.interrupted:
                print("\n⏸️  Análise pausada pelo usuário")
                break
            end_frame = min(start_frame + frames_per_segment, total_frames)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            motion_values = []
            brightness_values = []
            edge_values = []
            face_count = 0
            
            prev_frame = None
            frame_count = 0
            
            # Detector de faces
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calcular brilho
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Detectar bordas
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.mean(edges) / 255.0
                edge_values.append(edge_density)
                
                # Detectar faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_count += len(faces)
                
                # Calcular movimento
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion = np.mean(diff)
                    motion_values.append(motion)
                
                prev_frame = gray
            
            # Calcular estatísticas do segmento
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            avg_motion = np.mean(motion_values) if motion_values else 0
            motion_variance = np.var(motion_values) if motion_values else 0
            avg_brightness = np.mean(brightness_values) if brightness_values else 0
            brightness_variance = np.var(brightness_values) if brightness_values else 0
            avg_edge_density = np.mean(edge_values) if edge_values else 0
            face_density = face_count / max(frame_count, 1)
            
            # Score de atividade (0-1)
            activity_score = (
                min(avg_motion / 50, 1) * 0.4 +         # Movimento normalizado
                min(motion_variance / 200, 1) * 0.2 +    # Variação de movimento
                min(avg_edge_density * 2, 1) * 0.2 +     # Densidade de bordas
                min(face_density * 5, 1) * 0.2           # Densidade de faces
            )
            
            # Se há variação no brilho, pode indicar mudanças de cena
            if brightness_variance > 100:
                activity_score += 0.1
            
            segment = {
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'avg_motion': avg_motion,
                'motion_variance': motion_variance,
                'avg_brightness': avg_brightness,
                'brightness_variance': brightness_variance,
                'avg_edge_density': avg_edge_density,
                'face_density': face_density,
                'activity_score': activity_score,
                'is_highlight': activity_score > 0.3  # Threshold
            }
            
            segments.append(segment)
            self.current_segments = segments  # Atualizar estado atual
            self.last_frame_analyzed = end_frame  # Marcar último frame analisado
            
            print(f"📊 {start_time:.1f}-{end_time:.1f}s: "
                  f"atividade={activity_score:.3f} "
                  f"{'✨' if activity_score > 0.3 else ''}")
            
            # Salvar progresso periodicamente
            if len(segments) % self.save_interval == 0:
                self._save_temp_progress()
                print(f"💾 Progresso automático salvo ({len(segments)} segmentos)")
        
        cap.release()
        return segments

    def create_highlight_timestamps(self, segments, max_highlights=8):
        """Cria timestamps dos highlights (já que não podemos cortar sem MoviePy)"""
        print("⭐ Identificando highlights...")
        
        # Filtrar e ordenar highlights
        highlights = [s for s in segments if s['is_highlight']]
        highlights.sort(key=lambda x: x['activity_score'], reverse=True)
        highlights = highlights[:max_highlights]
        
        if not highlights:
            print("❌ Nenhum highlight detectado")
            return []
        
        print(f"\n🎯 {len(highlights)} highlights encontrados:")
        for i, h in enumerate(highlights, 1):
            print(f"🎬 Highlight {i}: {h['start']:.1f}s - {h['end']:.1f}s "
                  f"(Score: {h['activity_score']:.3f})")
        
        return highlights

    def generate_analysis_chart(self, segments):
        """Gera gráfico da análise"""
        print("📊 Gerando gráfico de análise...")
        
        times = [s['start'] for s in segments]
        motion_scores = [s['avg_motion'] for s in segments]
        activity_scores = [s['activity_score'] for s in segments]
        face_densities = [s['face_density'] for s in segments]
        
        plt.figure(figsize=(14, 10))
        
        # Gráfico principal - Scores
        plt.subplot(3, 1, 1)
        plt.plot(times, activity_scores, 'g-', linewidth=3, label='Score de Atividade')
        plt.axhline(y=0.3, color='red', linestyle='--', label='Threshold Highlight')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Score de Atividade')
        plt.title(f'Análise de Highlights - {self.video_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Marcar highlights
        highlights = [s for s in segments if s['is_highlight']]
        for h in highlights:
            plt.axvspan(h['start'], h['end'], alpha=0.2, color='yellow')
        
        # Gráfico de movimento
        plt.subplot(3, 1, 2)
        plt.plot(times, motion_scores, 'blue', linewidth=2, label='Movimento')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Movimento Médio')
        plt.title('Análise de Movimento')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de faces
        plt.subplot(3, 1, 3)
        plt.plot(times, face_densities, 'orange', linewidth=2, label='Densidade de Faces')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Faces por Frame')
        plt.title('Detecção de Faces')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Salvar gráfico
        chart_path = os.path.join(self.output_dir, f"{self.video_name}_basic_analysis.png")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Gráfico salvo: {chart_path}")
        return chart_path

    def save_highlight_list(self, highlights):
        """Salva lista de highlights em formato de texto"""
        if not highlights:
            return None
        
        output_file = os.path.join(self.output_dir, f"{self.video_name}_highlights.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"🎬 HIGHLIGHTS - {self.video_name}\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, h in enumerate(highlights, 1):
                minutes = int(h['start'] // 60)
                seconds = int(h['start'] % 60)
                end_minutes = int(h['end'] // 60)
                end_seconds = int(h['end'] % 60)
                
                f.write(f"Highlight {i}:\n")
                f.write(f"  ⏰ Tempo: {minutes:02d}:{seconds:02d} - {end_minutes:02d}:{end_seconds:02d}\n")
                f.write(f"  📊 Score: {h['activity_score']:.3f}\n")
                f.write(f"  🎯 Movimento: {h['avg_motion']:.1f}\n")
                f.write(f"  👥 Faces: {h['face_density']:.2f}/frame\n")
                f.write(f"  🔗 Link direto: {h['start']:.1f}s\n\n")
        
        print(f"📋 Lista de highlights salva: {output_file}")
        return output_file

    def run_basic_analysis(self, segment_duration=20):
        """Executa análise básica completa"""
        print(f"🚀 Análise Básica: {self.video_path}")
        print("⚡ Versão sem dependências pesadas - apenas OpenCV")
        
        try:
            # 1. Analisar características visuais
            segments = self.analyze_video_without_audio(segment_duration)
            
            if not segments:
                print("❌ Nenhum segmento analisado")
                return None, []
            
            # 2. Identificar highlights
            highlights = self.create_highlight_timestamps(segments)
            
            # 3. Gerar gráfico
            chart = self.generate_analysis_chart(segments)
            
            # 4. Salvar lista de highlights
            highlight_file = self.save_highlight_list(highlights)
            
            # 5. Salvar dados completos
            report = {
                'video': self.video_path,
                'timestamp': datetime.now().isoformat(),
                'method': 'basic_visual_analysis',
                'segments_analyzed': len(segments),
                'highlights_found': len(highlights),
                'segments': segments,
                'highlights': highlights
            }
            
            # Converter tipos numpy para tipos Python antes da serialização
            report = convert_numpy_types(report)
            
            report_path = os.path.join(self.output_dir, f"{self.video_name}_basic_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Análise completada com sucesso - remover arquivos temporários
            if not self.interrupted:
                self._cleanup_temp_files()
                print(f"\n🎉 Análise básica concluída!")
            else:
                print(f"\n⏸️  Análise pausada - progresso salvo para retomar depois")
            
            print(f"📊 Gráfico: {chart}")
            print(f"📋 Highlights: {highlight_file}")
            print(f"📄 Relatório: {report_path}")
            
            return highlight_file, highlights
            
        except KeyboardInterrupt:
            print(f"\n⏸️  Análise interrompida pelo usuário")
            if self.current_segments:
                self._save_temp_progress()
            return None, []
        except Exception as e:
            print(f"❌ Erro durante análise: {e}")
            if self.current_segments:
                print("💾 Salvando progresso mesmo com erro...")
                self._save_temp_progress()
            return None, []

def main():
    """Execução principal da versão básica"""
    videos_dir = "videos"
    
    if not os.path.exists(videos_dir):
        print("❌ Pasta 'videos' não encontrada")
        return
    
    video_files = [f for f in os.listdir(videos_dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    
    if not video_files:
        print("❌ Nenhum vídeo encontrado")
        return
    
    # Usar primeiro vídeo
    selected_video = video_files[0]
    print(f"📹 Analisando (versão básica): {selected_video}")
    
    video_path = os.path.join(videos_dir, selected_video)
    analyzer = BasicVideoAnalyzer(video_path)
    
    # Verificar se há progresso anterior
    if os.path.exists(analyzer.temp_file):
        print("\n🔄 Progresso anterior detectado!")
        response = input("Deseja continuar de onde parou? (s/n): ").lower()
        if response.startswith('n'):
            analyzer._cleanup_temp_files()
            print("🧹 Progresso anterior removido. Iniciando do zero...")
    
    # Executar análise
    result = analyzer.run_basic_analysis(segment_duration=20)
    
    if result[0]:
        print("\n💡 Para cortar o vídeo, use as timestamps geradas com:")
        print("   - FFmpeg (linha de comando)")
        print("   - Qualquer editor de vídeo")
        print("   - Sites como Kapwing ou Clideo")
    else:
        print("\n🔄 Dica: Execute novamente para continuar a análise pausada")
        print("💡 Para resetar tudo, delete o arquivo temporário na pasta basic_analysis")

if __name__ == "__main__":
    main()
"""
Versão simplificada e rápida do Video Analyzer
Para testes rápidos sem usar modelos pesados de IA
"""
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import json
from datetime import datetime
import librosa
import matplotlib.pyplot as plt

class SimpleVideoAnalyzer:
    def __init__(self, video_path, output_dir="highlights_simple"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        os.makedirs(output_dir, exist_ok=True)
        print("🔥 Analyzer Simples - Sem dependências pesadas!")

    def analyze_audio_energy(self, segment_duration=10):
        """Analisa energia do áudio para detectar momentos intensos"""
        print("🎵 Analisando energia do áudio...")
        
        video = VideoFileClip(self.video_path)
        audio = video.audio
        duration = video.duration
        
        segments = []
        
        for start in range(0, int(duration), segment_duration):
            end = min(start + segment_duration, duration)
            
            # Extrair segmento
            audio_segment = audio.subclip(start, end)
            
            # Salvar temporariamente
            temp_file = f"temp_audio_{start}.wav"
            audio_segment.write_audiofile(temp_file, verbose=False, logger=None)
            
            # Analisar com librosa
            y, sr = librosa.load(temp_file, sr=22050)
            
            # Calcular métricas de áudio
            energy = np.mean(librosa.feature.rms(y=y))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Detectar beats/ritmo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_strength = len(beats) / (end - start) if end > start else 0
            
            segments.append({
                'start': start,
                'end': end,
                'energy': float(energy),
                'spectral_centroid': float(spectral_centroid),
                'zero_crossing_rate': float(zero_crossing_rate),
                'tempo': float(tempo),
                'beat_strength': float(beat_strength),
                'audio_score': 0  # Será calculado
            })
            
            os.remove(temp_file)
            print(f"⚡ Segmento {start}-{end}s: energia={energy:.3f}")
        
        video.close()
        return segments

    def analyze_visual_motion(self, segments):
        """Analisa movimento visual e atividade na tela"""
        print("👀 Analisando movimento visual...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for segment in segments:
            start_frame = int(segment['start'] * fps)
            end_frame = int(segment['end'] * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            motion_values = []
            prev_frame = None
            brightness_values = []
            edge_values = []
            
            frame_count = 0
            while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calcular brilho médio
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Detectar bordas (mudanças visuais)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.mean(edges) / 255.0
                edge_values.append(edge_density)
                
                # Calcular movimento entre frames
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion = np.mean(diff)
                    motion_values.append(motion)
                
                prev_frame = gray
            
            # Calcular métricas visuais
            avg_motion = np.mean(motion_values) if motion_values else 0
            motion_variance = np.var(motion_values) if motion_values else 0
            avg_brightness = np.mean(brightness_values) if brightness_values else 0
            avg_edge_density = np.mean(edge_values) if edge_values else 0
            
            segment.update({
                'avg_motion': avg_motion,
                'motion_variance': motion_variance,
                'avg_brightness': avg_brightness,
                'edge_density': avg_edge_density,
                'visual_activity': avg_motion + motion_variance + avg_edge_density,
                'visual_score': 0  # Será calculado
            })
            
            print(f"📹 Visual {segment['start']}-{segment['end']}s: "
                  f"movimento={avg_motion:.2f}, bordas={avg_edge_density:.3f}")
        
        cap.release()
        return segments

    def calculate_simple_scores(self, segments):
        """Calcula scores simples baseado em energia e movimento"""
        print("⭐ Calculando scores...")
        
        # Normalizar valores
        max_energy = max(s['energy'] for s in segments) if segments else 1
        max_motion = max(s['avg_motion'] for s in segments) if segments else 1
        max_beat = max(s['beat_strength'] for s in segments) if segments else 1
        
        for segment in segments:
            # Score de áudio (0-1)
            audio_score = (
                (segment['energy'] / max_energy) * 0.4 +
                (segment['beat_strength'] / max_beat) * 0.3 +
                min(segment['tempo'] / 120, 1.5) * 0.3  # Tempo ideal ~120 BPM
            )
            
            # Score visual (0-1) 
            visual_score = (
                (segment['avg_motion'] / max_motion) * 0.5 +
                segment['edge_density'] * 0.3 +
                (segment['motion_variance'] / max_motion) * 0.2
            )
            
            # Score final
            final_score = (audio_score * 0.6 + visual_score * 0.4)
            
            segment.update({
                'audio_score': audio_score,
                'visual_score': visual_score,
                'final_score': final_score,
                'is_highlight': final_score > 0.5  # Threshold
            })
            
            print(f"🎯 {segment['start']}-{segment['end']}s: "
                  f"Score={final_score:.3f} {'✨' if final_score > 0.5 else ''}")
        
        return segments

    def create_highlights(self, segments, max_highlights=8):
        """Cria vídeo com os highlights"""
        print("✂️ Criando highlights...")
        
        # Filtrar e ordenar
        highlights = [s for s in segments if s['is_highlight']]
        highlights.sort(key=lambda x: x['final_score'], reverse=True)
        highlights = highlights[:max_highlights]
        
        if not highlights:
            print("❌ Nenhum highlight detectado")
            return None
        
        # Criar clips
        video = VideoFileClip(self.video_path)
        clips = []
        
        for i, highlight in enumerate(highlights):
            # Adicionar buffer antes e depois
            start = max(0, highlight['start'] - 1)
            end = min(video.duration, highlight['end'] + 1)
            
            clip = video.subclip(start, end)
            clips.append(clip)
            
            print(f"🎬 Highlight {i+1}: {start:.1f}s-{end:.1f}s "
                  f"(score: {highlight['final_score']:.3f})")
        
        # Concatenar e salvar
        if clips:
            final_video = concatenate_videoclips(clips, method="compose")
            output_path = os.path.join(self.output_dir, f"{self.video_name}_highlights_simple.mp4")
            
            final_video.write_videofile(
                output_path, 
                codec='libx264',
                verbose=False,
                logger=None
            )
            
            final_video.close()
            print(f"✅ Highlights salvos: {output_path}")
            return output_path
        
        video.close()
        return None

    def generate_analysis_chart(self, segments):
        """Gera gráfico da análise"""
        print("📊 Gerando gráfico...")
        
        times = [s['start'] for s in segments]
        audio_scores = [s['audio_score'] for s in segments]
        visual_scores = [s['visual_score'] for s in segments]
        final_scores = [s['final_score'] for s in segments]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(times, audio_scores, 'b-', label='Score Áudio', linewidth=2)
        plt.plot(times, visual_scores, 'r-', label='Score Visual', linewidth=2)
        plt.plot(times, final_scores, 'g-', label='Score Final', linewidth=3)
        plt.axhline(y=0.5, color='orange', linestyle='--', label='Threshold Highlight')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Score')
        plt.title(f'Análise de Highlights - {self.video_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Marcar highlights
        highlights = [s for s in segments if s['is_highlight']]
        for h in highlights:
            plt.axvspan(h['start'], h['end'], alpha=0.2, color='yellow')
        
        plt.subplot(2, 1, 2)
        energies = [s['energy'] for s in segments]
        motions = [s['avg_motion'] for s in segments]
        
        plt.plot(times, energies, 'purple', label='Energia Áudio', linewidth=2)
        plt.plot(times, motions, 'orange', label='Movimento Visual', linewidth=2)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Valores Brutos')
        plt.title('Métricas Detalhadas')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Salvar gráfico
        chart_path = os.path.join(self.output_dir, f"{self.video_name}_analysis.png")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Gráfico salvo: {chart_path}")
        return chart_path

    def run_simple_analysis(self, segment_duration=10):
        """Executa análise simplificada"""
        print(f"🚀 Análise Simples: {self.video_path}")
        print("⚡ Versão rápida - sem modelos de IA pesados")
        
        try:
            # 1. Analisar áudio
            segments = self.analyze_audio_energy(segment_duration)
            
            # 2. Analisar visual
            segments = self.analyze_visual_motion(segments)
            
            # 3. Calcular scores
            segments = self.calculate_simple_scores(segments)
            
            # 4. Criar highlights
            output_video = self.create_highlights(segments)
            
            # 5. Gerar gráfico
            chart = self.generate_analysis_chart(segments)
            
            # 6. Salvar dados
            report = {
                'video': self.video_path,
                'timestamp': datetime.now().isoformat(),
                'method': 'simple_analysis',
                'segments': segments,
                'highlights_count': len([s for s in segments if s['is_highlight']])
            }
            
            report_path = os.path.join(self.output_dir, f"{self.video_name}_simple_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\n🎉 Análise simples concluída!")
            print(f"🎬 Highlights: {output_video}")
            print(f"📊 Gráfico: {chart}")
            print(f"📋 Relatório: {report_path}")
            
            return output_video, segments
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            return None, []

def main():
    """Execução principal da versão simples"""
    videos_dir = "videos"
    
    if not os.path.exists(videos_dir):
        print("❌ Pasta 'videos' não encontrada")
        return
    
    video_files = [f for f in os.listdir(videos_dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    
    if not video_files:
        print("❌ Nenhum vídeo encontrado")
        return
    
    # Selecionar vídeo
    selected_video = video_files[0]  # Usar primeiro vídeo
    print(f"📹 Analisando (versão simples): {selected_video}")
    
    video_path = os.path.join(videos_dir, selected_video)
    analyzer = SimpleVideoAnalyzer(video_path)
    
    # Executar
    result = analyzer.run_simple_analysis(segment_duration=15)
    
    if result[0]:
        print("\n💡 Dica: Execute 'python main.py' para análise completa com IA")

if __name__ == "__main__":
    main()
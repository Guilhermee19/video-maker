
# 🎬 Video Maker AI - Extrator de Highlights

Sistema inteligente para análise automática de vídeos e extração dos melhores momentos usando IA local **100% gratuita**.

## 🚀 Funcionamento

O sistema analisa vídeos usando múltiplas técnicas de IA:

- **🗣️ Transcrição Local**: Whisper (OpenAI) para transcrever áudio
- **😄 Análise de Sentimentos**: Detecta momentos engraçados e positivos
- **👀 Análise Visual**: Movimento, faces e atividade na tela
- **⚡ Análise de Áudio**: Energia, ritmo e intensidade sonora
- **🤖 IA de Emoções**: Classificação de emoções no texto

## 📦 Instalação

### 1. Configurar Ambiente
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac
```

### 2. Instalar Automaticamente
```bash
python setup.py
```

### 3. Instalar Manualmente
```bash
pip install -r requirements.txt
```

## 🎯 Uso

### Análise Completa (Recomendado)
```bash
python main.py
```
- Usa todos os modelos de IA
- Análise mais precisa
- Detecta momentos engraçados por texto
- Melhor qualidade de highlights

### Análise Rápida (Para Testes)
```bash
python simple_analyzer.py
```
- Sem modelos pesados de IA
- Análise baseada em áudio e movimento
- Execução mais rápida
- Boa para testes iniciais

## 📁 Estrutura

```
videos/                 # Coloque seus vídeos aqui
highlights/            # Highlights gerados (análise completa)
highlights_simple/     # Highlights gerados (análise simples)
main.py               # Análise completa com IA
simple_analyzer.py    # Análise rápida
setup.py             # Configuração automática
```

## 🎮 Funcionamento Detalhado

### Análise Completa (main.py)
1. **Extração de Áudio**: Divide vídeo em segmentos
2. **Transcrição**: Whisper converte fala em texto
3. **Análise de Sentimentos**: VADER + TextBlob detectam humor
4. **Detecção de Emoções**: IA classifica emoções no texto
5. **Análise Visual**: OpenCV detecta movimento e faces
6. **Score de Highlights**: Combina todas as métricas
7. **Corte Automático**: Gera vídeo final com melhores momentos

### Análise Simples (simple_analyzer.py)
1. **Energia do Áudio**: Librosa analisa intensidade sonora
2. **Movimento Visual**: OpenCV detecta atividade na tela
3. **Score Simplificado**: Combina áudio + visual
4. **Gráficos**: Visualização da análise
5. **Highlights**: Extrai momentos com maior atividade

## 🛠️ Configurações

### Duração dos Segmentos
- **Análise Completa**: 30 segundos (padrão)
- **Análise Simples**: 15 segundos (padrão)

### Parâmetros Ajustáveis
```python
# No código principal
segment_duration = 30      # Duração dos segmentos
max_highlights = 10        # Máximo de highlights
highlight_threshold = 0.4  # Sensibilidade (0.1-1.0)
```

## 📊 Outputs

### Arquivos Gerados
- **`*_highlights.mp4`**: Vídeo final com melhores momentos
- **`*_analysis.json`**: Relatório completo da análise
- **`*_analysis.png`**: Gráfico visual (análise simples)

### Métricas Incluídas
- Score de sentimento (humor/diversão)
- Energia e ritmo do áudio
- Movimento e atividade visual
- Detecção de faces e emoções
- Score final de cada segmento

## 🎯 Casos de Uso

- **Gaming**: Highlights de jogos e reações
- **Vlogs**: Melhores momentos de vídeos pessoais
- **Palestras**: Partes mais envolventes
- **Streams**: Momentos engraçados de lives
- **Entrevistas**: Trechos mais interessantes

## 🔧 Requisitos Técnicos

### Mínimo
- Python 3.8+
- 4GB RAM
- CPU multi-core

### Recomendado
- 8GB+ RAM
- GPU NVIDIA (para aceleração)
- SSD para arquivos temporários

## 💡 Dicas de Performance

1. **GPU**: Instale CUDA para aceleração GPU
2. **Modelos**: Use Whisper "tiny" ou "base" para velocidade
3. **Segmentos**: Reduza duração para vídeos longos
4. **Espaço**: Mantenha 2x o tamanho do vídeo livre

## 🐛 Troubleshooting

### Erro de Memória
```bash
# Usar modelo menor do Whisper
whisper_model = whisper.load_model("tiny")
```

### Dependências
```bash
# Reinstalar dependências
pip install --upgrade -r requirements.txt
```

### Performance Lenta
1. Use `simple_analyzer.py` primeiro
2. Reduza `segment_duration`
3. Limite `max_highlights`

## 🤝 Contribuição

1. Fork o projeto
2. Crie sua feature branch
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

MIT License - Use livremente e contribua!

## 🌟 Features Futuras

- [ ] Interface gráfica (GUI)
- [ ] Mais modelos de IA local
- [ ] Análise de música e ritmo
- [ ] Templates de edição
- [ ] Integração com redes sociais
- [ ] Análise de múltiplos vídeos

---

**🎬 Transforme seus vídeos longos em highlights épicos com IA!**
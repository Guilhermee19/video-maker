#!/usr/bin/env python3
"""
Script de instalação e configuração do Video Analyzer
"""
import subprocess
import sys
import os

def install_requirements():
    """Instala as dependências necessárias"""
    print("🔧 Instalando dependências...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependências instaladas com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False
    return True

def download_nltk_data():
    """Baixa dados necessários do NLTK"""
    print("📚 Baixando dados do NLTK...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("✅ Dados NLTK baixados!")
    except Exception as e:
        print(f"⚠️ Aviso: {e}")

def check_system():
    """Verifica sistema e dependências"""
    print("🔍 Verificando sistema...")
    
    # Verificar Python
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ é necessário")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}")
    
    # Verificar se GPU está disponível
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
        else:
            print("🖥️ Usando CPU (recomendado: GPU para melhor performance)")
    except:
        print("⚠️ PyTorch não instalado ainda")
    
    return True

def main():
    print("🚀 Configurando Video Analyzer AI...")
    print("=" * 50)
    
    if not check_system():
        return
    
    if install_requirements():
        download_nltk_data()
        print("\n🎉 Setup concluído!")
        print("💡 Execute 'python main.py' para começar")
    else:
        print("\n❌ Setup falhou. Verifique os erros acima.")

if __name__ == "__main__":
    main()
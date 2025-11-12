### Comandos para executar o treinamento do modelo de sumarização:
# pip install git+https://github.com/huggingface/transformers
# pip install datasets sacrebleu rouge_score sentencepiece evaluate nltk
# python treinamento_sumarizacao.py
# O dataset utilizado pode ser encontrado em:
# https://huggingface.co/datasets/csebuetnlp/xlsum


# Importação das bibliotecas necessárias
import os
import tarfile
import json
import subprocess
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)


# Download do dataset portuguese_XLSum
DATASET_URL = "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/portuguese_XLSum_v2.0.tar.bz2"
DATASET_FILE = "portuguese_XLSum_v2.0.tar.bz2"

if not os.path.exists(DATASET_FILE):
    print("[INFO] Baixando dataset...")
    subprocess.run(["wget", DATASET_URL])

if not os.path.exists("portuguese_train.jsonl"):
    print("[INFO] Extraindo dataset...")
    with tarfile.open(DATASET_FILE, "r:bz2") as tar:
        tar.extractall()
else:
    print("[INFO] Dataset já extraído.")


# Conversão dos arquivos .jsonl para .json
def jsonl_to_json(jsonl_path, json_path):
    """Converte arquivos .jsonl (um JSON por linha) em um único .json."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


print("[INFO] Convertendo arquivos .jsonl para .json...")
jsonl_to_json("portuguese_train.jsonl", "portuguese_train.json")
jsonl_to_json("portuguese_val.jsonl", "portuguese_val.json")
jsonl_to_json("portuguese_test.jsonl", "portuguese_test.json")


# Treinamento do modelo mT5
# Parâmetros de saída
OUTPUT_DIR = "mt5-sum-pt"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Comando de treinamento do modelo
train_command = [
    "python",
    "-m",
    "transformers.examples.pytorch.summarization.run_summarization",
    "--model_name_or_path", "google/mt5-small",
    "--do_train",
    "--do_eval",
    "--train_file", "portuguese_train.json",
    "--validation_file", "portuguese_val.json",
    "--output_dir", OUTPUT_DIR,
    "--logging_dir", LOG_DIR,
    "--text_column", "text",
    "--summary_column", "summary",
    "--per_device_train_batch_size=4",
    "--per_device_eval_batch_size=4",
    "--gradient_accumulation_steps=4",
    "--learning_rate=3e-5",
    "--num_train_epochs=1",
    "--overwrite_output_dir",
    "--predict_with_generate"
]

print("[INFO] Iniciando treinamento do modelo mT5...")
subprocess.run(train_command)
print("[INFO] Treinamento finalizado com sucesso.")


# Teste do modelo treinado
print("[INFO] Carregando modelo treinado...")

model_dir = OUTPUT_DIR
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)


# Exemplo de sumarização
texto = """
O Brasil registrou um aumento significativo na produção de energia renovável em 2023,
com destaque para a energia solar e eólica. Segundo especialistas, o país tem potencial
para se tornar um dos maiores produtores de energia limpa do mundo na próxima década.
"""

print("[INFO] Gerando resumo de exemplo...")
resumo = summarizer(
    texto,
    max_length=80,
    min_length=20,
    do_sample=False
)

print("\nResumo gerado:")
print(resumo[0]['summary_text'])
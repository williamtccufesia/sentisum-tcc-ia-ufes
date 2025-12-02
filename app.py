### Comandos para executar a aplicação:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# rmdir /s /q venv   # (ou delete manualmente a pasta venv)
# python -m venv venv
# .\venv\Scripts\Activate
# pip install -r requirements.txt
# python app.py

# ================== Importações ==================
import os
import re
import torch
import torch.nn.functional as F
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from langdetect import detect, detect_langs, LangDetectException
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline as hf_pipeline
)
from flask import Flask, request, render_template

# ================== Configuração Flask ==================
app = Flask(__name__)

# ================== YouTube API ==================
API_KEY = os.getenv("YOUTUBE_API_KEY", "")
youtube_service = googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY)

# ================== Modelo BERT (classificação de sentimento) ==================
model_path = os.getenv("SENT_MODEL_DIR", "bert-sentiment-pt")
try:
    sent_model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    sent_tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    print(f"[INFO] Carregado modelo BERT local em: {model_path}")
except Exception as e:
    print(f"[WARN] Não foi possível carregar modelo local '{model_path}': {e}")
    print("[INFO] Tentando baixar 'neuralmind/bert-base-portuguese-cased' do Hugging Face Hub...")
    sent_model = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased")
    sent_tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

max_seq_length = 128

# ================== Modelo de sumarização (mT5 local / fallback Hub) ==================
SUM_MODEL_DIR = os.getenv("SUM_MODEL_DIR", "./mt5-summarization-pt")
try:
    sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_DIR, local_files_only=True)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_DIR, local_files_only=True)
    print(f"[INFO] Carregado modelo de sumarização local em: {SUM_MODEL_DIR}")
except Exception as e:
    print(f"[WARN] Não foi possível carregar modelo local '{SUM_MODEL_DIR}': {e}")
    print("[INFO] Tentando baixar 'csebuetnlp/mT5_multilingual_XLSum' do Hugging Face Hub...")
    sum_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    sum_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

device = 0 if torch.cuda.is_available() else -1
summarizer = hf_pipeline("summarization", model=sum_model, tokenizer=sum_tokenizer, device=device)

# ================== Estrutura de resultado por vídeo ==================
class VideoResult:
    def __init__(self, video_title, video_description, total_positivos, total_negativos, resumo=""):
        self.video_title = video_title
        self.video_description = video_description
        self.total_positivos = total_positivos
        self.total_negativos = total_negativos
        self.positives = total_positivos
        self.negatives = total_negativos
        self.resumo = resumo

# ================== Função: Buscar vídeos (mantida) ==================
def search_product_review_videos(youtube, search_query, maxResults=1000):
    video_info = []
    alvo = max(1, min(int(maxResults), 1000))
    coletados = 0

    query_pt = f"{search_query} avaliação análise pt-br em português"

    base_kwargs = {
        'q': query_pt,
        'type': 'video',
        'part': 'id,snippet',
        'maxResults': 50,
        'regionCode': 'BR',
        'relevanceLanguage': 'pt',
        'safeSearch': 'none',
    }

    page_token = None
    while True:
        kwargs = dict(base_kwargs)
        if page_token:
            kwargs['pageToken'] = page_token

        response = youtube.search().list(**kwargs).execute()
        items = response.get('items', [])
        if not items:
            break

        batch_ids = [it['id'].get('videoId') for it in items if it['id'].get('videoId')]
        if not batch_ids:
            page_token = response.get('nextPageToken')
            if not page_token:
                break
            continue

        details = youtube.videos().list(id=",".join(batch_ids), part='snippet').execute()
        for det in details.get('items', []):
            vid = det.get('id')
            snip = det.get('snippet', {}) or {}
            title = snip.get('title', '') or ''
            desc = snip.get('description', '') or ''
            default_lang = (snip.get('defaultLanguage') or '').lower()
            default_audio = (snip.get('defaultAudioLanguage') or '').lower()

            is_pt = False
            if default_lang.startswith('pt') or default_audio.startswith('pt'):
                is_pt = True
            else:
                try:
                    langs = detect_langs(f"{title} {desc}")
                    if any(l.lang == 'pt' and l.prob >= 0.20 for l in langs):
                        is_pt = True
                except LangDetectException:
                    pass

            if is_pt:
                video_info.append({
                    'video_id': vid,
                    'video_title': title,
                    'video_description': desc,
                })
                coletados += 1
                if coletados >= alvo:
                    return video_info

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return video_info

# ================== Função: Processar comentários (SEM keywords agora) ==================
def process_and_save_comments(youtube, video_id, output_file, **kwargs):
    """
    Retorna (next_page_token, comments_list) ou None se comentários desabilitados.
    comments_list é uma lista de dicts: {'text': ..., 'prob_pos': float, 'class': 'Positivo'/'Negativo'}
    """
    comments = []
    next_page_token = None

    while True:
        try:
            results = youtube.commentThreads().list(videoId=video_id, **kwargs).execute()
        except HttpError as e:
            if e.resp.status == 403:
                print(f"[WARN] Comentários desabilitados para o vídeo {video_id}")
                return None
            else:
                raise e

        for item in results.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay'] or ""
            try:
                if detect(comment) == 'pt':
                    clean = re.sub(r'\s+', ' ', re.sub(r'\W', ' ', comment)).strip()
                    short = clean[:max_seq_length]

                    inputs = sent_tokenizer(short, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors="pt")
                    with torch.no_grad():
                        sent_model.eval()
                        scores = sent_model(**inputs)[0]
                    prob_pos = float(F.softmax(scores, dim=1)[:, 1].item())
                    classe = "Positivo" if prob_pos > 0.5 else "Negativo"

                    comments.append({'text': clean, 'prob_pos': prob_pos, 'class': classe})
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(f"Comentário: {clean}\n")
                        f.write(f"Classificação: {classe}\n")
                        f.write(f"Probabilidade da classe positiva: {prob_pos:.4f}\n\n")
            except LangDetectException:
                pass

        next_page_token = results.get('nextPageToken')
        if next_page_token:
            kwargs['pageToken'] = next_page_token
        else:
            break

    return next_page_token, comments

# ================== Funções auxiliares para limpeza e chunk ==================
def clean_comments(comments):
    """Remove duplicatas mantendo ordem e reduz repetições exageradas."""
    seen = set()
    out = []
    for c in comments:
        text = c.strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        text = re.sub(r'\b(\w+)( \1){2,}', r'\1', text)
        out.append(text)
    return out

def chunk_text(text, chunk_size=800):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def clean_summary(text):
    text = re.sub(r'\b(\w+)( \1){2,}', r'\1', text)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    cleaned = []
    for s in sentences:
        if not cleaned or s != cleaned[-1]:
            cleaned.append(s)
    return " ".join(cleaned).strip()

# ================== Função: Sumarizar comentários (melhorada) ==================
def summarize_comments(comments_list, max_length=120, min_length=40):
    """
    Gera resumo em duas etapas:
    - limpa comentários (remove duplicatas e repetições)
    - concatena e chunka se necessário
    - faz mini-resumos e depois resumo final
    """
    if not comments_list:
        return "Sem comentários suficientes para resumir."

    texts = []
    if isinstance(comments_list[0], dict):
        texts = [c['text'] for c in comments_list]
    else:
        texts = list(comments_list)

    texts = clean_comments(texts)
    joined = " ".join(texts).strip()
    if not joined:
        return "Sem comentários suficientes para resumir."

    if len(joined) > 4000:
        chunks = list(chunk_text(joined, 1000))
    else:
        chunks = [joined]

    mini_resumos = []
    for chunk in chunks:
        try:
            out = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                repetition_penalty=1.8,
                no_repeat_ngram_size=4
            )
            resumo = clean_summary(out[0]['summary_text'])
            mini_resumos.append(resumo)
        except Exception as e:
            print("[ERROR] erro ao gerar mini-resumo:", e)

    if len(mini_resumos) == 0:
        return "Erro ao gerar resumo."
    if len(mini_resumos) == 1:
        return clean_summary(mini_resumos[0])

    try:
        final_input = " ".join(mini_resumos)
        out = summarizer(
            final_input,
            max_length=200,
            min_length=60,
            do_sample=False,
            repetition_penalty=1.8,
            no_repeat_ngram_size=4
        )
        return clean_summary(out[0]['summary_text'])
    except Exception as e:
        print("[ERROR] erro no resumo final:", e)
        return clean_summary(" ".join(mini_resumos))

# ================== Rotas Flask ==================
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form.get('search_query', '').strip()
    max_results = int(request.form.get('max_results', 3))

    output_file = 'comentarios_classificados.txt'
    open(output_file, 'w', encoding='utf-8').close()

    video_info_list = search_product_review_videos(youtube_service, search_query, maxResults=max_results)

    video_results = []
    overall_total_positivos = 0
    overall_total_negativos = 0
    all_comments = []

    for video_info in video_info_list:
        video_id = video_info['video_id']
        video_title = video_info['video_title']
        video_description = video_info['video_description']

        print(f"[INFO] processando vídeo {video_id} - {video_title}")

        page_token = None
        comments_video = []
        while True:
            res = process_and_save_comments(
                youtube_service,
                video_id,
                output_file,
                part='snippet',
                textFormat='plainText',
                pageToken=page_token,
                maxResults=100
            )
            if res is None:
                break
            page_token, new_comments = res
            comments_video.extend(new_comments)
            if not page_token:
                break

        if not comments_video:
            video_results.append(VideoResult(video_title, video_description, 0, 0, resumo="Sem comentários."))
            continue

        total_positivos = sum(1 for c in comments_video if c.get('prob_pos', 0) > 0.5)
        total_negativos = len(comments_video) - total_positivos

        overall_total_positivos += total_positivos
        overall_total_negativos += total_negativos

        resumo_video = summarize_comments(comments_video, max_length=100, min_length=30)

        all_comments.extend(comments_video)

        video_results.append(VideoResult(video_title, video_description, total_positivos, total_negativos, resumo=resumo_video))

    resumo_global = summarize_comments(all_comments, max_length=220, min_length=70)

    return render_template(
        'results.html',
        video_results=video_results,
        search_query=search_query,
        overall_total_positivos=overall_total_positivos,
        overall_total_negativos=overall_total_negativos,
        resumo_global=resumo_global
    )

# ================== Execução ==================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
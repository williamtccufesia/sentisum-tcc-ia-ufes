# 🧠 SentiSum — Análise de Sentimentos e Sumarização Automática de Comentários do YouTube  

Este projeto foi desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) da **Especialização em Inteligência Artificial e Ciência de Dados** da **Universidade Federal do Espírito Santo (UFES)**.  
O sistema tem como objetivo coletar comentários do YouTube em português, analisar seus sentimentos e gerar um resumo textual automático que sintetiza as principais opiniões expressas pelos usuários.

---

## 📘 Sumário

- [Visão Geral](#-visão-geral)
- [Arquitetura do Sistema](#-arquitetura-do-sistema)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação e Execução](#-instalação-e-execução)
- [Descrição dos Arquivos Principais](#-descrição-dos-arquivos-principais)
- [Modelos Utilizados](#-modelos-utilizados)
- [Resultados Obtidos](#-resultados-obtidos)
- [Limitações e Trabalhos Futuros](#-limitações-e-trabalhos-futuros)
- [Autor](#-autor)

---

## 🌐 Visão Geral

O **SentiSum** é um sistema web que integra duas tarefas fundamentais de **Processamento de Linguagem Natural (PLN)**:
1. **Análise de Sentimentos** — identifica se um comentário é positivo ou negativo;
2. **Sumarização Automática** — gera um resumo textual curto e coerente das principais opiniões.

Essas tarefas são realizadas por modelos modernos baseados em **Transformers**, aplicados a textos em português.  
A aplicação coleta comentários em tempo real através da **YouTube Data API v3**, processa os dados e apresenta resultados interpretáveis via uma interface web simples e intuitiva.

---

## 🏗️ Arquitetura do Sistema

A arquitetura do projeto é composta por cinco componentes principais:

```
YouTube API → Coleta de Comentários → Pré-processamento → 
BERT (Análise de Sentimentos) → mT5 (Sumarização) → Flask Web App → Resultado Final
```

📊 **Fluxo Geral:**
1. O usuário informa um termo de busca (ex: “fone de ouvido Bluetooth”);
2. O sistema coleta comentários de vídeos relacionados via YouTube API;
3. Os textos são limpos e padronizados;
4. O modelo **BERT-base-portuguese-cased** classifica os comentários (positivo/negativo);
5. O modelo **mT5-small** gera um resumo abstrativo;
6. O resultado é exibido na interface web, incluindo o resumo e a proporção de sentimentos.

---

## 🧩 Tecnologias Utilizadas

| Categoria | Ferramenta / Biblioteca |
|------------|--------------------------|
| **Linguagem principal** | Python 3.10 |
| **Framework web** | Flask |
| **Front-end** | HTML5, CSS3, Bootstrap 5, JavaScript |
| **APIs externas** | YouTube Data API v3 |
| **Modelos de IA** | BERT-base-portuguese-cased, mT5-small |
| **Bibliotecas de IA** | Hugging Face Transformers, PyTorch, Scikit-learn |
| **Manipulação de dados** | Pandas, NumPy |
| **Detecção de idioma** | langdetect |
| **Versionamento** | Git + GitHub |
| **Ambiente sugerido** | Python (Ubuntu / WSL / Docker Desktop) |

---

## 📁 Estrutura do Projeto

```
sentisum-tcc-ia-ufes/
├── .vscode/
│   └── (Arquivos de configuração do VS Code, como settings.json, não visíveis)
|
├── bert-sentiment-pt/           # Pasta do modelo BERT treinado para Análise de Sentimentos em Português
│   ├── config.json
│   ├── pytorch_model.bin
|   └── (Outros arquivos do modelo)
|
├── mt5-summarization-pt/        # Pasta do modelo mT5 treinado para Sumarização em Português
│   ├── all_results.json
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── train_results.json
│   ├── trainer_state.json
│   └── training_args.bin
|
├── templates/                   # Arquivos HTML para a interface web
│   ├── index.html               # Página principal (ou de entrada de dados)
│   └── results.html             # Página de resultados (sentimento e sumarização)
|
├── Treinamento - BERT/          # Pasta com scripts e dados relacionados ao treino do modelo BERT
│   ├── imdb-reviews-pt-br.csv   # Dataset de reviews para treinamento
│   └── treinamento_analise_sentimentos.py # Script de treinamento/teste do modelo BERT
|
├── Treinamento - Sum/           # Pasta com scripts relacionados ao treino do modelo de Sumarização
│   └── treinamento_sumarizacao.py # Script de treinamento/teste do modelo mT5
|
├── venv/                        # Ambiente virtual (virtual environment)
│   └── (Conteúdo do ambiente virtual)
|
├── app.py                       # Script principal da aplicação (Flask)
├── comentarios_classificados.txt # Arquivo de saída (exemplo de dados processados)
├── requirements.txt             # Lista de dependências Python do projeto
└── scripts.js                   # Script JavaScript (para a interface web)           
```

---

## ⚙️ Instalação e Execução

### 🔹 1. Clonar o repositório
```bash
git clone https://github.com/williamtccufesia/sentisum-tcc-ia-ufes.git
cd sentisum-tcc-ia-ufes
```

### 🔹 2. Criar e ativar ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### 🔹 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 🔹 4. Executar o servidor Flask
```bash
python app.py
```

### 🔹 5. Acessar a aplicação
Abra o navegador e vá até:
```
http://127.0.0.1:5000
```

---

## 📜 Descrição dos Arquivos Principais

### 🧩 `app.py`
Responsável por integrar todo o pipeline:
- Comunicação com a YouTube Data API;
- Coleta e pré-processamento de comentários;
- Execução dos modelos **BERT** e **mT5**;
- Retorno dos resultados ao front-end via JSON;
- Controle do fluxo de requisições com Flask.

---

### 🤖 `treinamento_analise_sentimentos.py`
Treina e ajusta o modelo **BERT-base-portuguese-cased**:
- Dataset utilizado: **IMDb-PT-BR**;
- Tarefa: classificação binária (positivo / negativo);
- Frameworks: `PyTorch`, `Transformers`, `Scikit-learn`;
- Métrica de desempenho: **Acurácia ≈ 93%**.

---

### 🧠 `treinamento_sumarizacao.py`
Treina o modelo **mT5-small** para gerar resumos abstrativos:
- Dataset utilizado: **XLSum (BBC Multilíngue)**;
- Limite de entrada: 512 tokens;
- Métrica de avaliação: **ROUGE-1 ≈ 0.45**, **ROUGE-L ≈ 0.42**;
- Saída: modelo salvo para inferência no Flask.

---

### 💻 `index.html`
Interface principal da aplicação web:
- Campo de busca de termo;
- Botão de execução da análise;
- Exibição dos resultados (sentimentos + resumo).

---

### ⚙️ `scripts.js`
Gerencia a interação entre o front-end e o backend Flask:
- Envia requisições `POST`;
- Recebe resultados de análise e resumo;
- Atualiza a interface dinamicamente.

---

## 🧠 Modelos Utilizados

| Modelo | Função | Fonte |
|---------|--------|--------|
| **BERT-base-portuguese-cased** | Classificação de sentimentos | Neuralmind |
| **mT5-small** | Sumarização abstrativa | Google Research |
| **IMDb-PT-BR** | Dataset de fine-tuning do BERT | Adaptação PT-BR |
| **XLSum (Português)** | Dataset de fine-tuning do mT5 | BBC Research |

---

## 📊 Resultados Obtidos

- **Acurácia BERT (validação):** ~93%  
- **ROUGE-1 (mT5):** 0.45  
- **ROUGE-L (mT5):** 0.42  
- **Tempo médio de execução:** 30–60 segundos por requisição  

**Exemplo de saída:**
> “A maioria dos usuários elogiou a qualidade sonora e o conforto, mas alguns relataram falhas na conexão e baixa durabilidade da bateria.”

---

## ⚠️ Limitações e Trabalhos Futuros

- O modelo de sumarização apresenta desempenho limitado em textos curtos ou dispersos;  
- Pretende-se integrar modelos mais avançados, como **GPT** e **Mistral**, para resumos mais fluentes;  
- Futuras versões incluirão métricas adicionais (**BLEU**, **BERTScore**) e integração com plataformas como **Twitter** e **Reddit**.

---

## 👨‍💻 Autor

**William Desteffani Soares**  
Especialização em Inteligência Artificial e Ciência de Dados  
Universidade Federal do Espírito Santo (UFES) — Universidade Aberta Capixaba (UnAC)  

📎 GitHub: [@williamtccufesia](https://github.com/williamtccufesia)  

---

## 📄 Licença

Este projeto é distribuído sob a licença **MIT**.  
Consulte o arquivo `LICENSE` para mais informações.

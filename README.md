# CardioIA — Fase 4: Assistente Cardiológico Virtual com Visão Computacional

> FIAP — 2° Ano — Fase 4 (2026)

> 🎥 **Demonstração em vídeo:** https://youtu.be/t8ixEr7cpXs

> ⚠️ **Protótipo acadêmico. Não é dispositivo médico, não foi validado clinicamente e não deve ser usado
> para diagnóstico real.** O valor exibido é um *score experimental*, não uma probabilidade clínica.

---

## Integrante

| Nome | RM | Turma |
|------|----|-------|
| Diogo Zequini | 565535 | 2TIAOA |

Entrega **individual** — fiz tudo sozinho (ver a observação sobre trabalho em equipe no final).

---

## Continuidade das fases anteriores

Esta é a quarta etapa do CardioIA. Cada fase puxa a anterior:

- **Fase 1 — Busca de Dados:** levantei e curei os três tipos de dados cardiológicos (numéricos do Cleveland, textuais das diretrizes da SBC e visuais de ECG), com governança, tratamento de ausentes e análise de viés.
- **Fase 2 — Diagnóstico Automatizado com IA:** NLP de sintomas, classificador de risco, MLP para ECG, portal React e XAI (SHAP/LIME).
- **Fase 3 — IoT, Edge e Cloud:** ESP32 + MQTT + Node-RED, monitoramento contínuo com alerta em tempo real.

Na Fase 4 eu coloco a **Visão Computacional** no centro: em vez de ler um número ou uma frase, o sistema agora **lê uma imagem médica** (raio-X de tórax) e devolve uma classificação interpretável, com mapa de calor mostrando *onde* a rede olhou.

```text
Fase 1: dados preparados
Fase 2: diagnóstico assistido por IA
Fase 3: monitoramento contínuo conectado
Fase 4: visão computacional sobre imagem médica (esta entrega)
```

Mantive a marca registrada do CardioIA: **foco cardiológico, governança/LGPD e explicabilidade** — por isso escolhi a **cardiomegalia** (aumento da área cardíaca, mensurável pelo índice cardiotorácico) como alvo dentro do raio-X de tórax.

---

## O que foi pedido vs. o que entreguei

### Parte 1 — Pré-processamento e organização das imagens *(3 pts)*

| Requisito | Entrega | Onde |
|-----------|---------|------|
| Selecionar um dataset público de imagens médicas | NIH Chest X-ray14 (112.120 imagens / 30.805 pacientes), recorte **Cardiomegaly vs No Finding** | `notebooks/1_preprocessamento.ipynb` |
| Redimensionamento, normalização e conversão | 224×224, RGB garantido, normalização travada por modelo (embutida no `.keras`) | `notebooks/1_preprocessamento.ipynb` |
| Conjuntos de treino, validação e teste | Subconjunto balanceado de 2.400 imagens (1.200/classe), split **70/15/15 por paciente** (não por imagem) | `dados/processed/data_manifest.csv` |
| Documentar o pipeline | Relatório de 1-2 páginas com etapas e justificativas | `docs/relatorio_parte1_preprocessamento.md` |

> **Decisão crítica:** o NIH tem várias imagens do mesmo paciente. Se eu dividisse por imagem, o mesmo paciente cairia em treino e teste e as métricas inflariam (erro clássico do dataset). Usei `GroupShuffleSplit` por `Patient ID` e **provei que não há paciente repetido entre as partições**.

### Parte 2 — Classificação com CNN *(CNN do zero 2 pts + Transfer Learning 2 pts)*

| Requisito | Entrega | Onde |
|-----------|---------|------|
| CNN simples treinada do zero | CNN convolucional própria (downsampling agressivo + BatchNorm + Dropout) | `notebooks/2_modelagem_cnn_e_transfer.ipynb`, `modelos/cnn_scratch.keras` |
| Transfer Learning com modelo pré-treinado | **MobileNetV2** (ImageNet) com cabeça nova **+ fine-tuning** das últimas camadas | mesmo notebook, `modelos/transfer_mobilenetv2.keras` |
| Métricas (acurácia, matriz de confusão, precisão, recall, F1) | Todas, **+ AUC-ROC e AUC-PR** (mais honestas para classe desbalanceada) | `resultados/` (CSVs + PNGs) |
| Protótipo de apresentação dos resultados | **Portal web Flask** com upload, score, Grad-CAM, painel de métricas, dados, equidade e histórico | `app/backend/`, `start.py` |

> Sobre o modelo pré-treinado: a demanda pede "como VGG16 ou ResNet, **quando aplicável**". Escolhi o **MobileNetV2** porque entrega o mesmo papel didático sendo bem mais leve (≈14 MB vs ≈550 MB da VGG16) — roda no Colab/CPU e carrega instantâneo no Flask. A troca de backbone é de uma linha, então VGG/ResNet ficam a um passo.

### Critérios de avaliação

| Critério | Pontos | Status |
|----------|:------:|--------|
| Pipeline de pré-processamento | 3 | ✅ |
| Treinamento e avaliação de CNN do zero | 2 | ✅ |
| Implementação de Transfer Learning funcional | 2 | ✅ |
| Apresentação dos resultados em protótipo | 2 | ✅ (portal web) |
| Documentação clara | 1 | ✅ |
| Trabalho em equipe (grupo de 2-5) | +1 extra | ➖ entrega individual (ver observação) |

### Ir Além 1 — Ética e Governança em Visão Computacional

| Requisito | Entrega | Onde |
|-----------|---------|------|
| Identificar limitações do dataset (desbalanceamento, representatividade) | Desbalanceamento, rótulos fracos (NLP), viés demográfico e de aquisição | `docs/relatorio_etica_fairness.md` |
| Aplicar métricas de fairness e discutir implicações | TPR/FPR por **gênero, faixa etária e incidência (AP/PA)**, com IC 95% (bootstrap) | `notebooks/3_fairness_governanca.ipynb`, `resultados/fairness_metrics.csv` |
| Relatório (até 2 páginas) | Limitações + metodologia + mitigação | `docs/relatorio_etica_fairness.md` |

> **Achado mais forte:** o maior viés não é de gênero, é da **posição da incidência (AP vs PA)** — FPR 0,60 (AP) contra 0,24 (PA), gap de 0,36. Isso confirma um *atalho técnico*: exames AP, feitos em pacientes mais graves/acamados, inflam a silhueta cardíaca, e a rede aprende parte disso em vez da patologia.

### Ir Além 2 — Integração com aplicativo mobile

| Requisito | Entrega | Onde |
|-----------|---------|------|
| Interface React Native (upload + resultado) | App Expo com tela de upload/câmera e tela de resultado (classe + score + Grad-CAM) | `app/mobile/` |
| Integração com backend (Flask) | Consome o mesmo `/predict` do backend; host configurável em `app.json` | `app/mobile/src/services/api.ts` |
| Repositório GitHub + vídeo de até 3 min | Repositório publicado + vídeo de demonstração | ✅ (links abaixo) |

---

## Resultados (execução real no Kaggle — conjunto de teste, 345 imagens)

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC | AUC-PR |
|--------|:--------:|:--------:|:------:|:--:|:-------:|:------:|
| CNN do zero | 0,513 | 0,526 | 0,466 | 0,494 | 0,518 | 0,548 |
| Transfer (base congelada) | 0,693 | 0,694 | 0,710 | 0,702 | 0,765 | 0,768 |
| **Transfer (fine-tuning)** | **0,693** | 0,673 | **0,773** | **0,720** | **0,770** | **0,770** |

A leitura honesta: a CNN do zero fica **praticamente no chute** (AUC ≈ 0,52) — esperado para uma rede pequena nessa tarefa difícil. O **Transfer Learning dispara para AUC ≈ 0,77**, e o **fine-tuning** melhora o recall de cardiomegalia (0,71 → **0,77**), que é o que importa em triagem (não deixar passar). Matrizes de confusão, curvas ROC/PR, curvas de treino e os Grad-CAMs estão em `resultados/`.

---

## Início rápido (Start Supremo)

Um comando faz tudo — monta o ambiente, mostra os resultados, roda um autoteste de inferência e abre o portal:

```bash
python start.py
```

Requer **Python 3.10–3.12** (o TensorFlow ainda não suporta o 3.13; o script acha um Python compatível sozinho e cria um `.venv` isolado na primeira execução). O portal abre em `http://localhost:5000` — é só arrastar um raio-X de tórax (ou clicar num exemplo) e ver classe, score e Grad-CAM.

Variações: `python start.py --test` (só métricas + autoteste, sem subir a interface) · `python start.py --port 8000`.

---

## Arquitetura

```text
NIH Chest X-ray14
      │  filtro Cardiomegaly vs No Finding + split POR PACIENTE (anti-vazamento)
      ▼
 data_manifest.csv ──► [Notebook 1: pré-processamento]
      │
      ▼
 [Notebook 2: CNN do zero  +  Transfer Learning MobileNetV2 + fine-tuning]
      │  avaliação (acurácia, matriz, precisão, recall, F1, AUC-ROC, AUC-PR) + Grad-CAM
      ▼
 modelos/*.keras + preprocess_config.json
      │
      ├──► [Backend Flask] ──► Portal web (upload → classe + score + Grad-CAM + painéis)
      │                         └► App mobile React Native (mesmo /predict)
      └──► [Notebook 3: Fairness] ──► TPR/FPR por subgrupo + relatório ético
```

---

## Estrutura do repositório

```text
.
├── start.py                     # Start Supremo: ambiente + autoteste + portal
├── notebooks/
│   ├── 1_preprocessamento.ipynb
│   ├── 2_modelagem_cnn_e_transfer.ipynb
│   ├── 3_fairness_governanca.ipynb
│   └── utils_cardioia.py         # split por paciente, avaliação, Grad-CAM (compartilhado)
├── app/
│   ├── backend/                  # Flask: API + portal web (templates/index.html)
│   └── mobile/                   # React Native (Expo) — Ir Além 2
├── modelos/                      # cnn_scratch.keras · transfer_mobilenetv2.keras · preprocess_config.json
├── resultados/                   # métricas, matrizes, ROC/PR, Grad-CAM, fairness, dashboard_data.json
├── docs/
│   ├── relatorio_parte1_preprocessamento.md
│   ├── relatorio_etica_fairness.md
│   └── prints/                   # evidências (métricas + teste do Flask)
├── dados/
│   ├── processed/data_manifest.csv   # proveniência (arquivo, paciente, classe, split)
│   └── raw/sample/                   # 4 raios-X de teste prontos para usar
└── scripts/                      # EDA, preparo de insumos, geração do dashboard
```

---

## Como executar (manual, sem o Start Supremo)

### Backend Flask + portal web

```bash
cd app/backend
py -3.11 -m venv .venv && .venv\Scripts\activate    # Windows (Python 3.10–3.12)
pip install -r requirements.txt
python app.py        # http://localhost:5000
```

O caminho de inferência (carga do modelo, pré-processamento embutido, predição e Grad-CAM) foi **validado localmente** (Python 3.11 + TensorFlow 2.16) — evidência em `docs/prints/`.

### App mobile (Ir Além 2)

```bash
cd app/mobile
npm install
# defina o backend em app.json -> expo.extra.apiHost (IP local p/ Android, ou URL ngrok https p/ iOS)
npx expo start       # abrir no Expo Go (dispositivo físico p/ câmera)
```

---

## Governança e ética

Mantive a linha do CardioIA desde a Fase 1:

- **"Score experimental", nunca "diagnóstico"** ou "probabilidade clínica".
- **Retenção zero:** a imagem enviada é processada em memória e nunca gravada em disco; logs sem imagem.
- **Disclaimer médico** visível em toda tela (portal e mobile).
- **Grad-CAM como auditoria de atalho** — o modelo deve olhar a silhueta cardíaca, não marcadores/bordas.
- **Split por paciente** (anti-vazamento) e **AUC-PR** para classe desbalanceada — rigor que a própria rubrica nem exige.
- Análise de equidade por subgrupo em `docs/relatorio_etica_fairness.md`.

---

## Sobre o trabalho em equipe

A demanda dá +1 ponto extra para grupos de 2 a 5 integrantes. Optei por fazer **sozinho**, mantendo a continuidade das fases anteriores (que também entreguei individualmente). Portanto, abro mão do ponto extra de equipe — o que está aqui foi todo desenvolvido pelo Diogo Zequini.

---

## Links públicos para correção

- **Repositório GitHub:** https://github.com/diogozeq/fase4-Cap-1.VisaoComputacionalnaClinica
- **Notebooks Kaggle (públicos, executados, com resultados):**
  - Pré-processamento: https://www.kaggle.com/code/diogozequini/cardioia-fase4-etapa1-eda
  - Modelagem (CNN + Transfer): https://www.kaggle.com/code/diogozequini/cardioia-fase4-treino
  - Grad-CAM + Fairness (Ir Além 1): https://www.kaggle.com/code/diogozequini/cardioia-fase4-gradcam-fairness
- **Vídeo demo:** https://youtu.be/t8ixEr7cpXs

---

## Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1">

Creative Commons Attribution 4.0 International — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

# Relatório — Parte 1: Pré-processamento e Organização das Imagens

**Projeto:** CardioIA — Fase 4 (Assistente Cardiológico Virtual com Visão Computacional)
**Aluno:** Diogo Zequini · RM 565535 · Turma 2TIAOA
**Dataset:** NIH Chest X-ray14 (Kaggle: `nih-chest-xrays/data`) — recorte **Cardiomegaly vs. No Finding**

> ⚠️ Protótipo acadêmico. Não é dispositivo médico e não deve ser usado para diagnóstico real.

## 1. Escolha e justificativa do dataset

O NIH Chest X-ray14 reúne **112.120 radiografias de tórax de 30.805 pacientes**, com rótulos para 14
achados torácicos e metadados demográficos (idade, gênero) e técnicos (posição da incidência). Para
manter a coerência cardiológica do CardioIA (Fases 1-3), recortamos o problema na patologia
**Cardiomegaly** (cardiomegalia — aumento da área cardíaca, mensurável pelo índice cardiotorácico),
contra a classe **No Finding** (sem achados). A demanda autoriza explicitamente o uso do NIH Chest X-ray.

**Disponibilidade no dataset:** 2.776 imagens com Cardiomegaly e 60.353 com No Finding.

**Limitação central declarada:** os rótulos do NIH são derivados de laudos por NLP/*weak supervision*,
não de validação cardiológica manual imagem a imagem. Isso é registrado como limitação no Ir Além 1.

## 2. Recorte e balanceamento

Para um treino viável e balanceado, amostramos **1.200 imagens por classe** (seed fixa `565535`),
totalizando **2.400 imagens**. O balanceamento ~1:1 reduz o viés de classe majoritária; o
desbalanceamento original (1:22) é discutido no relatório de fairness.

## 3. Split por paciente (decisão crítica anti-vazamento)

O NIH contém **múltiplas imagens do mesmo paciente**. Particionar por *imagem* (erro clássico do
dataset) colocaria o mesmo paciente em treino e teste, fazendo a rede memorizar a anatomia individual
e **inflar artificialmente as métricas**. Para evitar isso, usamos
`sklearn.model_selection.GroupShuffleSplit` com `groups = Patient ID`, garantindo **interseção vazia de
pacientes** entre as partições (validado por asserção no código).

| Split | Imagens | Pacientes únicos | Cardiomegaly | No Finding |
|---|---:|---:|---:|---:|
| Treino | 1.707 | 1.306 | 857 | 850 |
| Validação | 348 | 280 | 167 | 181 |
| Teste | 345 | 281 | 176 | 169 |
| **Total** | **2.400** | **1.867** | 1.200 | 1.200 |

**Prova anti-vazamento:** `train∩val = train∩test = val∩test = ∅` (0 pacientes em comum).

## 4. Pré-processamento aplicado

- **Redimensionamento:** todas as imagens para **224×224** (compatível com MobileNetV2/VGG16/ResNet50).
- **Canais:** garantia de **3 canais RGB** (`decode_png(channels=3)`), pois há radiografias em escala de cinza.
- **Normalização travada por modelo:** `Rescaling(1/255)` para a CNN do zero e `preprocess_input` do
  MobileNetV2 para o Transfer Learning — embutidos no próprio modelo, garantindo **paridade
  treino ↔ inferência** (mesmo pré-processamento no notebook e no app Flask).
- **Data augmentation** (somente no treino): flip horizontal, rotação leve (±3%) e zoom (±5%).
  Não se usa flip vertical (a anatomia torácica tem orientação).
- **Sanitização de idade:** o campo `Patient Age` do NIH tem valores absurdos (>120); filtrados para [0, 120].

## 5. Controle de confundidores e governança

Registramos a distribuição de **posição da incidência** (AP vs. PA) e **gênero** por split, pois ambos
são confundidores conhecidos em radiografia de tórax (AP correlaciona com pacientes acamados; o
aumento aparente da área cardíaca varia com a projeção). Esses metadados alimentam a análise de
*fairness* (Ir Além 1).

| Split | AP | PA | | Gênero (geral) | Cardiomegaly | No Finding |
|---|---:|---:|---|---|---:|---:|
| Treino | 705 | 1.002 | | Feminino | 641 | 499 |
| Validação | 127 | 221 | | Masculino | 559 | 701 |
| Teste | 139 | 206 | | | | |

## 6. Reprodutibilidade

- Seed global única (`565535`) em NumPy/Python/TensorFlow e no `random_state` do split.
- Manifesto de proveniência `data_manifest.csv` (arquivo, Patient ID, classe, split, gênero, idade,
  posição, caminho) salvo como artefato.
- Dependências com versões fixadas em `requirements.txt`.
- Pipeline determinístico (mesma seed → mesmo split/manifesto). O treino em GPU pode ter pequena
  variação numérica não-determinística; o split e a amostragem, esses sim, são exatamente reproduzíveis.

## 7. Entregáveis desta etapa

- Notebook de pré-processamento (Kaggle) — split por paciente + EDA.
- `data_manifest.csv` (manifesto de proveniência).
- Este relatório.

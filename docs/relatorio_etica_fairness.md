# Relatório — Ir Além 1: Ética e Governança em Visão Computacional

**Projeto:** CardioIA — Fase 4 · **Aluno:** Diogo Zequini · RM 565535 · Turma 2TIAOA
**Modelo avaliado:** Transfer Learning (MobileNetV2 + fine-tuning) · **Conjunto:** teste (split por paciente)

> ⚠️ Protótipo acadêmico. Não é dispositivo médico e não deve ser usado para diagnóstico real.

## 1. Limitações do dataset (NIH Chest X-ray14)

- **Rótulos fracos:** derivados de laudos por NLP/*weak supervision*, com erro estimado de ~10% pelos
  próprios autores. Não há validação cardiológica manual imagem a imagem. Logo, o "ground truth" é ruidoso.
- **Desbalanceamento de origem:** Cardiomegaly representa ~2,5% do dataset (2.776 / 112.120). Mitigamos com
  amostragem balanceada 1:1 no subset, mas isso não corrige o viés de prevalência do mundo real.
- **Representatividade:** população hospitalar dos EUA (NIH Clinical Center); não generaliza para todas as
  origens demográficas/equipamentos. Sem informação de etnia.
- **Confundidores técnicos:** a posição da incidência (**AP** vs **PA**) altera a silhueta cardíaca aparente —
  exames AP (frequentes em pacientes acamados/graves) tendem a superestimar a área cardíaca. É um atalho que
  a CNN pode aprender no lugar da patologia.

## 2. Metodologia de fairness

Avaliamos **equidade de erro** por subgrupo no conjunto de teste, comparando:

- **TPR** (sensibilidade / *equal opportunity*) — fração de casos com cardiomegalia corretamente detectados.
- **FPR** — fração de casos normais erroneamente sinalizados.
- **Precisão** e **acurácia** por subgrupo, com **intervalo de confiança 95% (bootstrap, 1.000 reamostragens)**.

**Atributos:** gênero (M/F), faixa etária (<45, 45-60, >60) e posição da incidência (AP/PA).

**Por que equalized odds / equal opportunity, e não demographic parity:** em diagnóstico médico, a
prevalência da doença difere *legitimamente* entre subgrupos (ex.: cardiomegalia é mais comum em idosos).
Forçar *demographic parity* (mesma taxa de predições positivas por grupo) mascararia diferenças clínicas
reais e poderia até reduzir a qualidade do cuidado. O critério defensável é **igualdade de TPR/FPR** entre
subgrupos — todos devem ser detectados com a mesma sensibilidade.

## 3. Resultados por subgrupo (execução real, n=345 no teste)

Modelo avaliado: Transfer Learning (MobileNetV2 + fine-tuning), limiar 0,5.

| Atributo | Subgrupo | n | TPR (IC95%) | FPR (IC95%) | Precisão |
|---|---|---:|---|---|---:|
| Gênero | Feminino | 164 | 0,839 [0,76–0,91] | 0,493 [0,38–0,61] | 0,690 |
| Gênero | Masculino | 181 | 0,699 [0,59–0,80] | 0,316 [0,22–0,40] | 0,652 |
| Idade | <45 | 134 | 0,712 [0,60–0,81] | 0,295 [0,19–0,41] | 0,743 |
| Idade | 45-60 | 109 | 0,809 [0,68–0,91] | 0,468 [0,35–0,60] | 0,567 |
| Idade | >60 | 102 | 0,821 [0,72–0,91] | 0,413 [0,27–0,55] | 0,708 |
| Incidência | AP | 139 | 0,841 [0,75–0,92] | **0,600** [0,49–0,71] | 0,580 |
| Incidência | PA | 206 | 0,729 [0,64–0,81] | **0,242** [0,16–0,33] | 0,765 |

**Gaps de equalized odds (máx − mín):** Gênero — TPR 0,14 / FPR 0,18 · Idade — TPR 0,11 / FPR 0,17 ·
**Incidência — TPR 0,11 / FPR 0,36**. Gráficos: `resultados/fairness_genero.png`, `fairness_idade.png`.

## 4. Discussão e implicações

- **Maior viés = posição da incidência (AP vs PA).** O FPR em AP (0,60) é mais que o dobro do PA (0,24) —
  gap de 0,36. Isso **confirma a hipótese de atalho**: exames AP são feitos majoritariamente em pacientes
  acamados/graves, e a projeção AP aumenta a silhueta cardíaca aparente. O modelo aprende parcialmente a
  *projeção*, não só a patologia → superdiagnostica em AP. É o achado ético mais relevante do trabalho.
- **Gênero:** mulheres têm TPR maior (0,84 vs 0,70), mas também FPR maior (0,49 vs 0,32) — o modelo é "mais
  agressivo" em sinalizar cardiomegalia em mulheres, gerando mais alarmes falsos nesse grupo.
- **Idade:** sensibilidade cresce com a idade (0,71 → 0,82), coerente com maior prevalência real, mas o FPR
  também sobe — atenção ao equilíbrio.
- **Incerteza:** subgrupos têm IC 95% largos (ex.: ±0,10–0,12 em TPR); as conclusões respeitam essa margem e
  não devem ser superinterpretadas com n pequeno.

## 5. Práticas de mitigação propostas

1. **Reamostragem/estratificação por subgrupo** no treino (balancear não só a classe, mas a composição demográfica).
2. **Limiar por subgrupo** (*threshold tuning*) para equalizar a sensibilidade — discutido como hipótese ética,
   com a ressalva de que limiares distintos por gênero/idade levantam questões regulatórias.
3. **Controle do confundidor AP/PA:** estratificar avaliação por incidência; idealmente, treinar com a
   incidência como variável de controle ou mascarar marcadores.
4. **Curadoria de rótulos** em uma amostra (validação por especialista) para medir o ruído de rótulo.
5. **Transparência:** publicar *model card* e *data card* com estas limitações (princípio de governança do CardioIA).

## 6. Conclusão

A Visão Computacional pode apoiar a triagem cardiológica, mas o uso responsável exige (a) evitar vazamento de
dados (resolvido com split por paciente), (b) métricas honestas em classe desbalanceada (AUC-PR), (c)
auditoria de atalhos (Grad-CAM) e (d) verificação de equidade entre subgrupos. Sem isso, um número alto de
acurácia pode esconder um sistema injusto ou cientificamente inválido.

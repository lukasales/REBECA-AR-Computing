# Relatório da contribuição computacional — Entrega 4

## Projeto

**Disciplina:** IN1123 — Realidade Virtual e Aumentada  
**Professor:** Francisco Paulo Magalhães Simões  
**Implementação computacional:** Lucas Sales  
**Equipe:** Lucas Sales, Nathalia Cruz Farias e Thais Mendonça dos Santos

## 1. Resumo

A contribuição computacional foi estruturada como um benchmark de
aplicabilidade técnica de modelos existentes de classificação de
expressões faciais.

Não foi desenvolvido um novo algoritmo.

Foram investigados:

- EmoAR / DenseNet121;
- Vision Transformer (ViT);
- EmotionRAM-faces / EfficientNet-B0.

Os experimentos avaliaram, quando aplicável:

- desempenho classificatório;
- custo computacional;
- latência;
- confiança e calibração;
- robustez a perturbações;
- estabilidade temporal em vídeo;
- disponibilidade de pipeline live.

## 2. EmoAR

O EmoAR foi utilizado como baseline histórico.

Características auditadas:

- DenseNet121;
- 7 classes;
- checkpoint original auditado;
- preprocessing histórico reproduzido;
- avaliação no FER2013 privateTest.

Resultados principais:

| Métrica | EmoAR |
|---|---:|
| Accuracy | 66,51% |
| Balanced Accuracy | 63,82% |
| Macro-F1 | 62,45% |
| Parâmetros | 6,96 M |
| Latência CPU média | 70,595 ms |
| ECE | 0,0839 |

## 3. ViT

Foi incluído o modelo `trpakov/vit-face-expression`.

Resultados no mesmo FER2013 privateTest:

| Métrica | ViT |
|---|---:|
| Accuracy | 71,75% |
| Balanced Accuracy | 71,01% |
| Macro-F1 | 71,07% |
| Parâmetros | 85,80 M |
| Latência CPU média | 151,343 ms |
| ECE | 0,1836 |

O ViT apresentou maior desempenho classificatório.

Entretanto, também apresentou:

- custo computacional muito maior;
- latência maior;
- confiança softmax mais descalibrada;
- menor estabilidade diante das perturbações estáticas avaliadas.

A comparação pareada EmoAR x ViT no FER2013 apresentou:

- EmoAR correto / ViT errado: 299;
- EmoAR errado / ViT correto: 487;
- previsões discordantes: 786;
- teste exato de McNemar: `p = 2.0633e-11`.

## 4. Robustez a perturbações

Protocolo:

- 350 imagens;
- 50 imagens por classe;
- 8 perturbações por imagem;
- 2.800 pares original-perturbado por modelo.

Perturbações:

- rotação ±5 graus;
- translação horizontal ±2 pixels;
- brilho 0.90 / 1.10;
- contraste 0.90 / 1.10.

Resultados:

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Accuracy perturbada | 0,6257 | 0,6868 |
| Pairwise stability | 0,9089 | 0,8764 |
| Fully stable fraction | 0,6886 | 0,5543 |
| Mean abs. confidence change | 0,0516 | 0,0624 |

O ViT continuou mais acurado, mas o EmoAR apresentou maior
estabilidade de rótulo sob as perturbações avaliadas.

## 5. Benchmark temporal

Foi executado um benchmark separado no RAVDESS.

Configuração:

- 14 vídeos;
- 2 atores;
- 7 categorias compatíveis;
- 1.620 frames efetivamente decodificados;
- 3.240 previsões;
- crop facial fixo por vídeo;
- escala de crop 1.7;
- sem temporal smoothing no protocolo primário.

Resultados:

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Class switches | 124 | 116 |
| Flickering / change rate | 7,721% | 7,223% |
| Consecutive agreement | 92,279% | 92,777% |
| Mean run length | 11,74 frames | 12,46 frames |
| Mean abs. confidence delta | 0,0431 | 0,0489 |
| P95 abs. confidence delta | 0,1601 | 0,2470 |
| Mean temporal latency | 50,49 ms | 116,82 ms |
| Effective classifier FPS | 19,81 | 8,56 |

O ViT apresentou ligeiramente menor flickering agregado.

Por outro lado:

- apresentou maior oscilação de confiança;
- apresentou menor throughput;
- continuou substancialmente mais lento.

Estabilidade temporal não implica correção semântica.

Os rótulos RAVDESS são rótulos de clipe, e não ground truth
frame a frame.

## 6. EmotionRAM-faces

Após solicitação da equipe, o EmotionRAM-faces foi investigado
como terceiro modelo.

Repositório auditado:

`voxarlabs/emotionram_faces_demo`

Commit:

`e65a85d9cb3e7ba9f62ab5b84a1bcc06d9614172`

Características:

- EfficientNet-B0;
- treinamento declarado no AfeW;
- entrada RGB 224x224;
- normalização ImageNet;
- 4.017.796 parâmetros;
- 8 classes.

Classes:

- Anger;
- Contempt;
- Disgust;
- Fear;
- Happiness;
- Neutral;
- Sadness;
- Surprise.

Checkpoint:

`enet_b0_8_best_afew.pt`

SHA-256:

`47c1423f3e6f50e3750bf7b0eda7db947c9ce0c2637e1766bf2187eddc652b17`

## 7. Avaliação cross-dataset do EmotionRAM

O modelo foi avaliado no mesmo FER2013 privateTest utilizado nos
experimentos anteriores.

Foram processadas as 3.589 imagens.

Resultados:

| Métrica | EmotionRAM |
|---|---:|
| Accuracy | 55,70% |
| Balanced Accuracy | 52,20% |
| Macro-F1 | 50,92% |
| Mean confidence | 0,6468 |
| Parâmetros | 4,02 M |

O EmotionRAM possui uma classe adicional (`Contempt`) que não
existe no FER2013.

Foram obtidas:

- 52 previsões `Contempt`;
- 1,45% das 3.589 imagens.

Essas previsões foram contabilizadas como incorretas.

Por isso, o resultado deve ser descrito como uma avaliação
**cross-dataset/off-the-shelf**.

Ele não constitui uma comparação perfeitamente equivalente com
EmoAR e ViT, pois:

1. foi treinado no AfeW;
2. está sendo avaliado no FER2013;
3. possui um espaço de saída com 8 classes em vez de 7.

## 8. Tabela comparativa

| Aspecto | EmoAR | ViT | EmotionRAM-faces |
|---|---|---|---|
| Arquitetura | DenseNet121 | Vision Transformer | EfficientNet-B0 |
| Dataset de treinamento | FER2013-based | FER2013 | AfeW |
| Classes | 7 | 7 | 8 |
| FER2013 Accuracy | 66,51% | 71,75% | 55,70% |
| Balanced Accuracy | 63,82% | 71,01% | 52,20% |
| Macro-F1 | 62,45% | 71,07% | 50,92% |
| Parâmetros | 6,96 M | 85,80 M | 4,02 M |
| Latência dedicada | 70,60 ms | 151,34 ms | Não medida |
| Pipeline webcam | Não faz parte do benchmark | Não faz parte do benchmark | Disponível |
| Face detection live | Não faz parte do benchmark | Não faz parte do benchmark | MediaPipe |

## 9. Investigação do pipeline live do EmotionRAM

O repositório original disponibiliza `webcam.py`.

Pipeline identificado:

`Webcam -> OpenCV -> MediaPipe -> face ROI -> EfficientNet-B0 ->
classe de expressão -> overlay`

Durante a reprodução no ambiente atual foram encontradas duas
questões de compatibilidade.

### MediaPipe

A versão moderna inicialmente instalada não disponibilizava a API
`mp.solutions` esperada pelo código legado.

Foi utilizada:

`mediapipe==0.10.21`

Com isso foi confirmado:

- `mp.solutions` disponível;
- `FaceDetection` disponível;
- OpenCV 4.11 funcional.

### PyTorch

O checkpoint foi salvo como objeto completo.

Em versões novas do PyTorch, `torch.load` utiliza
`weights_only=True` por padrão.

Para o checkpoint original foi necessária a adaptação local:

`weights_only=False`

O modelo já havia sido previamente carregado e executado com
sucesso durante a avaliação FER2013.

### Estado da webcam

Após as adaptações, o pipeline chegou à etapa de captura física da
câmera.

Na máquina utilizada nesta sessão, o OpenCV não recebeu frames,
resultando em:

`[INFO] Skipping due to lack of frames.`

Portanto:

**a webcam ainda não foi validada visualmente end-to-end.**

O teste será repetido posteriormente em um notebook com câmera.

Não se deve afirmar, neste momento, que a execução live local foi
completamente validada.

## 10. Síntese dos trade-offs

Não existe um vencedor absoluto.

### ViT

Vantagens:

- melhor desempenho classificatório no FER2013;
- ligeiramente menos flickering no benchmark temporal.

Desvantagens:

- aproximadamente 12,3x mais parâmetros que EmoAR;
- maior latência;
- pior calibração da confiança;
- maior oscilação temporal da confiança.

### EmoAR

Vantagens:

- significativamente menor que ViT;
- menor latência;
- melhor calibração relativa;
- maior estabilidade nas perturbações avaliadas.

### EmotionRAM

Vantagens:

- menor modelo dos três;
- pipeline live já disponível no repositório;
- MediaPipe integrado à detecção facial.

Cuidados:

- treinamento em AfeW;
- resultado FER2013 cross-dataset;
- classe adicional Contempt;
- latência dedicada ainda não medida;
- benchmark temporal não realizado nesta entrega.

## 11. Limitações

- FER2013 não é específico de crianças.
- FER2013 não é específico de crianças com TEA.
- RAVDESS não representa a população-alvo.
- Não houve avaliação clínica.
- Não houve avaliação educacional.
- Classificação de expressão facial não equivale à inferência do
  estado emocional interno.
- A latência avaliada para EmoAR/ViT é do classificador, não de uma
  aplicação de RA completa.
- O EmotionRAM foi avaliado cross-dataset.
- Os rótulos RAVDESS não constituem ground truth frame a frame.
- O teste físico da webcam do EmotionRAM ainda está pendente.

## 12. Arquivos produzidos

Principais documentos:

- `docs/implementation_handoff.md`
- `docs/computational_results_for_paper.md`
- `docs/emotionram_technical_evaluation.md`
- `docs/implementation_report_delivery4.md`

Resultados:

- `results/model_comparison_delivery4.md`
- `results/paper_summary_table.md`
- `results/paper_summary_table.csv`
- `results/emotionram_fer2013_results.json`
- `results/emotionram_fer2013_predictions.csv`
- `results/temporal_video_predictions.csv`
- `results/temporal_video_summary_aggregate.csv`

Figuras:

- `figures/01_private_test_performance.png`
- `figures/02_cpu_latency.png`
- `figures/03_reliability_diagram.png`
- `figures/04_accuracy_coverage.png`
- `figures/05_perturbation_stability.png`
- `figures/06_temporal_applicability_tradeoff.png`

## 13. Estado da implementação

| Item | Status |
|---|---|
| Reprodução/auditoria EmoAR | Concluído |
| Avaliação ViT | Concluído |
| Latência CPU | Concluído para EmoAR/ViT |
| Calibração | Concluído para EmoAR/ViT |
| Robustez | Concluído para EmoAR/ViT |
| Benchmark temporal RAVDESS | Concluído para EmoAR/ViT |
| Auditoria EmotionRAM | Concluída |
| EmotionRAM no FER2013 | Concluído (cross-dataset) |
| Comparação dos três modelos | Concluída |
| Pipeline webcam EmotionRAM | Preparado; validação visual pendente |
| Aplicação completa de RA | Fora do escopo desta entrega |

## 14. Repositório

https://github.com/lukasales/REBECA-AR-Computing

Commits principais de fechamento:

- `7bb1580` — Prepare implementation handoff for delivery 4
- `17fda44` — Evaluate EmotionRAM faces for delivery 4


# Handoff da implementação computacional — Entrega 4

Este é o arquivo principal para a responsável pela escrita do paper.

## Implementação concluída

Foram avaliados EmoAR/DenseNet121 e ViT em:

1. desempenho classificatório;
2. custo computacional;
3. latência CPU;
4. confiança e calibração;
5. robustez a perturbações;
6. estabilidade temporal em vídeo.

## FER2013

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Accuracy | 0.6651 | 0.7175 |
| Balanced accuracy | 0.6382 | 0.7101 |
| Macro-F1 | 0.6245 | 0.7107 |

Teste exato de McNemar:

`p = 2.0633e-11`

## Custo e latência

- EmoAR: 6,96 M parâmetros;
- ViT: 85,80 M parâmetros.

| Modelo | Latência média CPU |
|---|---:|
| EmoAR | 70,60 ms |
| ViT | 151,34 ms |

## Calibração

| Métrica | EmoAR | ViT |
|---|---:|---:|
| ECE | 0.0839 | 0.1836 |
| MCE | 0.1477 | 0.3750 |

## Robustez

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Perturbed accuracy | 0.6257 | 0.6868 |
| Pairwise stability | 0.9089 | 0.8764 |
| Fully stable fraction | 0.6886 | 0.5543 |

## Avaliação temporal

- dataset: RAVDESS;
- 14 vídeos;
- 1.620 frames;
- 3.240 previsões;
- 7 categorias;
- crop fixo por vídeo;
- sem smoothing temporal.

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Class switches | 124 | 116 |
| Flickering | 7.72% | 7.22% |
| Consecutive agreement | 92.28% | 92.78% |
| Mean |Δ confidence| | 0.0431 | 0.0489 |
| P95 |Δ confidence| | 0.1601 | 0.2470 |
| Mean latency | 50.49 ms | 116.82 ms |
| Effective FPS | 19.81 | 8.56 |

## Principal achado

O ViT apresentou melhor desempenho classificatório no FER2013 e
ligeiramente menor flickering no benchmark temporal.

Entretanto, possui custo computacional muito maior, maior latência,
pior calibração da confiança, menor estabilidade nas perturbações
estáticas e maior oscilação temporal da confiança.

Portanto, maior accuracy não implica automaticamente maior
aplicabilidade para uma futura experiência interativa de RA.

## Figuras disponíveis

- `figures/01_private_test_performance.png`
- `figures/02_cpu_latency.png`
- `figures/03_reliability_diagram.png`
- `figures/04_accuracy_coverage.png`
- `figures/05_perturbation_stability.png`
- `figures/06_temporal_applicability_tradeoff.png`

## Limitações importantes

Não afirmar que:

- os modelos reconhecem o estado emocional interno;
- houve validação com crianças;
- houve validação com crianças com TEA;
- houve validação clínica;
- o RAVDESS fornece accuracy frame a frame;
- o benchmark representa uma aplicação RA completa;
- já existe webcam/câmera ao vivo.

## Estado da implementação

A contribuição experimental para a Entrega 4 está finalizada.

O sistema atual processa datasets de imagens e vídeos gravados.

Ainda não foi implementado:

- webcam ao vivo;
- face detection/tracking live;
- feedback visual em tempo real;
- integração RA completa.

## EmotionRAM-faces

Após a solicitação da equipe, o EmotionRAM-faces também foi
investigado.

Características identificadas:

- arquitetura EfficientNet-B0;
- treinamento declarado no AfeW;
- entrada RGB 224x224;
- normalização ImageNet;
- 8 classes, incluindo Contempt;
- 4,017,796 parâmetros;
- checkpoint original auditado;
- pipeline de webcam disponível;
- detecção facial live com MediaPipe.

Foi realizada uma avaliação off-the-shelf no mesmo FER2013
privateTest de 3.589 imagens.

Resultados:

| Métrica | EmotionRAM |
|---|---:|
| Accuracy | 0.5570 |
| Balanced accuracy | 0.5220 |
| Macro-F1 | 0.5092 |
| Mean confidence | 0.6468 |
| Parameters | 4,017,796 |
| Contempt predictions | 52 (1.45%) |

Sete das oito classes apresentam correspondência direta com o
protocolo FER2013. A classe adicional Contempt foi contabilizada
como previsão incorreta.

Essa avaliação deve ser descrita como cross-dataset/off-the-shelf,
pois o EmotionRAM foi treinado no AfeW. Portanto, o valor de
accuracy não deve ser interpretado como uma comparação
perfeitamente equivalente com EmoAR e ViT.

O EmotionRAM também é particularmente relevante para o protótipo
futuro porque seu repositório inclui uma implementação de webcam
com OpenCV, MediaPipe, inferência EfficientNet-B0 e overlay da
categoria prevista.

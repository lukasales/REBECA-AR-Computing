# Computational Results for Paper

Documento técnico de apoio ao paper.

Este arquivo consolida os protocolos e resultados computacionais obtidos até o momento. Ele deve ser usado como fonte de consulta para números, configurações experimentais, figuras e limitações.

## 1. Escopo da contribuição computacional

A implementação avalia a aplicabilidade técnica de modelos existentes de classificação de expressões faciais para uma futura experiência interativa de Realidade Aumentada.

Não foi desenvolvido um novo algoritmo de reconhecimento de emoções.

Os principais eixos avaliados são:

- desempenho classificatório;
- custo computacional;
- latência;
- confiança e calibração;
- robustez a pequenas perturbações visuais;
- estabilidade temporal em vídeo.

## 2. Modelos

### EmoAR

- arquitetura: DenseNet121;
- 7 categorias FER2013;
- checkpoint original do projeto EmoAR;
- commit do repositório EmoAR: 33d0c6b46775a830744625ee97f736b3255b8060;
- preprocessing histórico: Resize(255), CenterCrop(224), tensor e normalização ImageNet.

### ViT

- modelo: trpakov/vit-face-expression;
- revisão: ef0bc6fc34241b6587e7e009e7711357be28c024;
- arquitetura baseada em ViT-base;
- 7 categorias;
- preprocessing definido pelo processor do modelo;
- saída remapeada para a ordem de classes FER2013.

## 3. Dataset

- dataset: Aaryan333/fer2013_train_publicTest_privateTest;
- revisão: da8d1643649c86bfc2cffba5b744b51fdf329212;
- split principal de avaliação: privateTest;
- tamanho do privateTest: 3.589 imagens;
- categorias: Angry, Disgust, Fear, Happy, Sad, Surprise e Neutral.

## 4. Desempenho classificatório

Resultados no FER2013 privateTest:

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Accuracy | 0.6651 | 0.7175 |
| Balanced Accuracy | 0.6382 | 0.7101 |
| Macro-F1 | 0.6245 | 0.7107 |

Comparação pareada:

- EmoAR correto / ViT errado: 299;
- EmoAR errado / ViT correto: 487;
- previsões discordantes: 786;
- teste exato de McNemar: p = 2.0633e-11.

Interpretação técnica:

No protocolo utilizado, o ViT apresentou desempenho classificatório superior ao baseline EmoAR.

Esse resultado não representa validação em crianças, crianças com TEA ou reconhecimento do estado emocional interno.

## 5. Custo computacional

- EmoAR: aproximadamente 6,96 milhões de parâmetros;
- ViT: aproximadamente 85,80 milhões de parâmetros;
- razão aproximada: 12,3 vezes mais parâmetros no ViT.

## 6. Latência em CPU

A avaliação final de latência utilizou protocolo contrabalanceado com 6 execuções.

Hardware principal:

- AMD Ryzen 7 5700X;
- 15,93 GB RAM;
- Windows 11;
- PyTorch CPU.

Resultados agregados:

| Modelo | Latência média | Mediana |
|---|---:|---:|
| EmoAR | 70.595 ms | 65.322 ms |
| ViT | 151.343 ms | 149.374 ms |

- razão média dentro das execuções: 2.171x;
- razão das médias agregadas: aproximadamente 2.144x.

Importante:

A latência representa preprocessing e classificação. Não inclui captura da câmera, detecção facial, tracking ou renderização de uma aplicação de RA.

## 7. Confiança e calibração

Confiança foi definida como a maior probabilidade softmax da previsão.

A análise utilizou top-label calibration com 15 bins de largura uniforme.

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Mean confidence | 0.7487 | 0.9010 |
| ECE | 0.0839 | 0.1836 |
| MCE | 0.1477 | 0.3750 |
| Confidence gap | 0.0836 | 0.1836 |

Erros de alta confiança:

- EmoAR: 97 erros com confiança >= 0.90, correspondendo a 8,07% de seus erros;
- ViT: 463 erros com confiança >= 0.90, correspondendo a 45,66% de seus erros.

Interpretação técnica:

O ViT apresentou maior accuracy, mas sua confiança softmax bruta foi mais descalibrada no protocolo utilizado.

Probabilidade softmax não deve ser interpretada como probabilidade do estado emocional real da pessoa.

## 8. Robustez a perturbações

Subconjunto:

- 350 imagens;
- 50 por classe;
- seed: 20260921;
- 8 perturbações por imagem;
- 2.800 comparações original-perturbada por modelo.

Perturbações:

- rotação -5 graus;
- rotação +5 graus;
- translação horizontal -2 pixels;
- translação horizontal +2 pixels;
- brilho 0.90;
- brilho 1.10;
- contraste 0.90;
- contraste 1.10.

Resultados agregados:

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Accuracy perturbada | 0.6257 | 0.6868 |
| Pairwise label stability | 0.9089 | 0.8764 |
| Fully stable fraction | 0.6886 | 0.5543 |
| Mean absolute confidence change | 0.0516 | 0.0624 |

Interpretação técnica:

O ViT permaneceu mais acurado, mas apresentou menor estabilidade de rótulo, principalmente diante de perturbações geométricas.

Este experimento é um proxy controlado de robustez e não uma avaliação de estabilidade temporal em vídeo.

## Avaliação temporal em vídeo

Foi executado um benchmark temporal utilizando 14 vídeos do RAVDESS,
com 2 atores e 7 categorias compatíveis com os classificadores.

Foram efetivamente decodificados 1.620 frames. Cada frame foi
processado pelos dois modelos, totalizando 3.240 previsões.

Foi utilizado crop facial fixo por vídeo, com escala 1.7, sem
temporal smoothing.

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Class switches | 124 | 116 |
| Flickering/change rate | 7.72% | 7.22% |
| Consecutive agreement | 92.28% | 92.78% |
| Mean confidence | 0.6771 | 0.8544 |
| Mean abs. confidence delta | 0.0431 | 0.0489 |
| P95 abs. confidence delta | 0.1601 | 0.2470 |
| Mean latency | 50.49 ms | 116.82 ms |
| Effective classifier FPS | 19.81 | 8.56 |

O ViT apresentou ligeiramente menos flickering agregado, porém
maior latência e maior variação temporal da confiança.

Os rótulos do RAVDESS são rótulos de clipe e não são utilizados
como ground truth frame a frame.

A avaliação utiliza vídeos previamente gravados. Ela não constitui
uma aplicação de webcam ou RA em tempo real.

## 9. Síntese do trade-off

| Aspecto | EmoAR | ViT |
|---|---:|---:|
| Accuracy maior | 66,51% | 71,75% |
| Macro-F1 maior | 62,45% | 71,07% |
| Menos parâmetros | 6,96 M | 85,80 M |
| Menor latência | 70,60 ms | 151,34 ms |
| Menor ECE | 0,0839 | 0,1836 |
| Maior estabilidade pareada | 90,89% | 87,64% |
| Maior fração totalmente estável | 68,86% | 55,43% |

Resumo:

O ViT apresenta melhor desempenho classificatório, enquanto o EmoAR apresenta menor custo computacional, menor latência, melhor calibração relativa e maior estabilidade agregada nas perturbações avaliadas.

## 10. Figuras disponíveis

- figures/01_private_test_performance.png;
- figures/02_cpu_latency.png;
- figures/03_reliability_diagram.png;
- figures/04_accuracy_coverage.png;
- figures/05_perturbation_stability.png;
- figures/06_temporal_applicability_tradeoff.png.

## 11. Resultados e arquivos de apoio

- results/paper_summary_table.csv;
- results/paper_summary_table.md;
- results/computational_results_team_review.xlsx;
- docs/emoar_fer2013_baseline_results.md;
- docs/emoar_vs_vit_comparative_results.md;
- docs/computational_cost_results.md;
- docs/confidence_calibration_results.md;
- docs/perturbation_stability_results.md.

## 12. Limitações que devem acompanhar os resultados

- FER2013 não é específico de crianças;
- FER2013 não é específico de crianças com TEA;
- não existe validação clínica;
- classificação de expressão facial não equivale a inferência do estado emocional interno;
- o benchmark de latência não representa uma aplicação RA completa;
- a avaliação de perturbações é distinta da avaliação temporal em vídeo;
- o ViT foi fine-tuned em FER2013, portanto o privateTest utilizado aqui não deve ser apresentado como evidência garantida de generalização externa independente.

## 13. Estado atual e próxima lacuna computacional

A avaliação temporal em vídeos previamente gravados foi concluída.

A implementação atual constitui um benchmark computacional
reproduzível de aplicabilidade.

Ainda não foi implementado um protótipo live com webcam, detecção
ou tracking facial em tempo real, feedback visual ou integração
com uma experiência completa de Realidade Aumentada.

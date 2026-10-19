# Resumo final da implementação computacional — Entrega 4

## O que foi implementado

- reprodução do baseline EmoAR/DenseNet121;
- comparação com ViT no FER2013;
- desempenho classificatório;
- custo computacional;
- benchmark de latência CPU;
- confiança e calibração;
- robustez a perturbações;
- avaliação temporal em vídeos RAVDESS;
- análise de flickering, runs, confiança e latência temporal;
- seis figuras experimentais.

## Resultados principais

| Métrica | EmoAR | ViT |
|---|---:|---:|
| FER2013 Accuracy | 66,51% | 71,75% |
| Macro-F1 | 62,45% | 71,07% |
| Parâmetros | 6,96 M | 85,80 M |
| Latência CPU dedicada | 70,60 ms | 151,34 ms |
| ECE | 0,0839 | 0,1836 |
| Perturbation stability | 90,89% | 87,64% |
| Temporal flickering | 7.72% | 7.22% |
| Temporal agreement | 92.28% | 92.78% |
| Temporal FPS efetivo | 19.81 | 8.56 |

## Benchmark temporal

- 14 vídeos RAVDESS;
- 2 atores;
- 7 categorias;
- 1.620 frames;
- 3.240 previsões;
- crop facial fixo por vídeo;
- escala 1.7;
- sem smoothing temporal.

## Estado atual

A parte experimental prevista para a Entrega 4 está concluída.

O repositório processa imagens e vídeos previamente gravados.

Ainda não existe webcam ao vivo, tracking facial em tempo real ou
aplicação RA completa.

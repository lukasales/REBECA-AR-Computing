# REBECA-AR Computing

Componente computacional do projeto da disciplina
**2026.2 - IN1123 - Realidade Virtual e Aumentada (CIn/UFPE)**.

## Objetivo

Este repositório investiga a aplicabilidade técnica de modelos
existentes de classificação de expressões faciais para uma possível
extensão futura do projeto REBECA com Realidade Aumentada.

A proposta atual não é criar um novo algoritmo de reconhecimento de
emoções, mas reproduzir e avaliar abordagens existentes.

## O que foi implementado

- reprodução e auditoria do baseline EmoAR;
- avaliação do EmoAR no FER2013;
- avaliação de um modelo ViT contemporâneo;
- comparação pareada EmoAR x ViT;
- análise de custo computacional;
- benchmark de latência CPU;
- análise de confiança e calibração;
- análise de erros de alta confiança;
- análise accuracy x coverage;
- avaliação de robustez a pequenas perturbações;
- geração de tabelas e figuras preliminares.

## Resultados preliminares

### Classificação - FER2013 privateTest

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Accuracy | 0.6651 | 0.7175 |
| Balanced accuracy | 0.6382 | 0.7101 |
| Macro-F1 | 0.6245 | 0.7107 |

O ViT apresentou melhor desempenho classificatório no protocolo
avaliado.

### Latência CPU

| Modelo | Latência média |
|---|---:|
| EmoAR | 70.60 ms |
| ViT | 151.34 ms |

O ViT apresentou maior custo computacional e aproximadamente 2.1x
mais latência nesse benchmark CPU.

### Calibração

| Métrica | EmoAR | ViT |
|---|---:|---:|
| ECE | 0.0839 | 0.1836 |
| MCE | 0.1477 | 0.3750 |

Apesar da maior accuracy, o ViT apresentou confiança softmax bruta
mais descalibrada no FER2013 privateTest.

### Robustez a perturbações

Esta avaliação utiliza um subconjunto balanceado de 350 imagens.

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Accuracy perturbada | 0.6257 | 0.6868 |
| Pairwise stability | 0.9089 | 0.8764 |
| Fully stable fraction | 0.6886 | 0.5543 |

O ViT permaneceu mais acurado, mas apresentou menor estabilidade de
rótulo diante de algumas pequenas perturbações.

## Figuras preliminares

- [Desempenho classificatório](figures/01_private_test_performance.png)
- [Latência CPU](figures/02_cpu_latency.png)
- [Reliability diagram](figures/03_reliability_diagram.png)
- [Accuracy x coverage](figures/04_accuracy_coverage.png)
- [Robustez a perturbações](figures/05_perturbation_stability.png)

## Documentação

Os protocolos e resultados detalhados estão em `docs/`.

Principais arquivos:

- `docs/emoar_audit.md`
- `docs/emoar_fer2013_baseline_results.md`
- `docs/emoar_vs_vit_comparative_results.md`
- `docs/computational_cost_results.md`
- `docs/confidence_calibration_results.md`
- `docs/perturbation_stability_results.md`

Os resultados brutos e intermediários estão em `results/`.

## Limitações

Os resultados atuais não representam validação clínica e não
estabelecem desempenho para crianças com TEA.

FER2013 contém imagens estáticas de expressões faciais.

Também ainda não foi implementada uma aplicação completa de RA,
avaliação temporal em vídeo ou validação com crianças.

## Estado atual

A parte computacional principal está pronta para revisão da equipe.

Antes de expandir os experimentos, precisamos validar em grupo quais
resultados e análises serão incorporados ao draft da Entrega 3.

## Implementação computacional

Lucas Francisco Alcantara Sales Macedo

# Resumo da contribuição computacional para revisão da equipe

## Contexto

Este documento resume a parte computacional desenvolvida até o momento
para o projeto da disciplina de Realidade Virtual e Aumentada.

O objetivo é permitir que a equipe valide se os experimentos realizados
estão alinhados com o escopo do artigo antes de continuarmos novas
implementações.

## O que foi feito

A contribuição computacional partiu do baseline EmoAR, utilizado como
referência para reconhecimento de expressões faciais.

Foi implementado um pipeline reproduzível para:

1. auditar e reproduzir o baseline EmoAR;
2. avaliar seu desempenho no FER2013;
3. comparar o baseline com um modelo ViT mais recente;
4. avaliar custo computacional e latência;
5. analisar confiança e calibração;
6. avaliar robustez a pequenas perturbações de imagem.

Não foi desenvolvido um novo algoritmo de reconhecimento de emoções.

---

## 1. Reprodução do baseline EmoAR

Foi analisado o código original do EmoAR e identificado o pipeline
utilizado para classificação facial.

Principais elementos reproduzidos:

- arquitetura DenseNet121;
- classificação em 7 categorias FER2013;
- checkpoint original;
- preprocessing histórico utilizado pela aplicação web;
- ordem das classes do modelo.

O objetivo dessa etapa foi garantir que a comparação posterior
utilizasse corretamente o baseline existente.

---

## 2. Comparação EmoAR x ViT

Foi selecionado um modelo baseado em Vision Transformer:

`trpakov/vit-face-expression`

Os dois modelos foram avaliados sobre o mesmo split `privateTest` do
FER2013, contendo 3.589 imagens.

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Accuracy | 0.6651 | 0.7175 |
| Balanced accuracy | 0.6382 | 0.7101 |
| Macro-F1 | 0.6245 | 0.7107 |

### Resultado

No protocolo utilizado, o ViT apresentou melhor desempenho
classificatório.

Isso representa desempenho sobre categorias de expressões faciais no
FER2013 e não deve ser interpretado como reconhecimento do estado
emocional real de uma pessoa.

Também não representa validação específica para crianças com TEA.

---

## 3. Custo computacional e latência

Também foi investigado o custo de utilizar cada modelo.

### Tamanho aproximado

- EmoAR: 6,96 milhões de parâmetros;
- ViT: 85,80 milhões de parâmetros.

O ViT possui aproximadamente 12,3 vezes mais parâmetros.

### Latência CPU

Foi realizado um benchmark contrabalanceado para reduzir a influência
da ordem de execução dos modelos.

| Modelo | Latência média |
|---|---:|
| EmoAR | 70,60 ms |
| ViT | 151,34 ms |

### Resultado

O ViT foi aproximadamente 2,1 vezes mais lento no ambiente CPU
avaliado.

Esse tempo corresponde ao processamento do classificador e não à
latência de uma aplicação de Realidade Aumentada completa.

---

## 4. Confiança e calibração

Foi analisado se a confiança softmax dos modelos acompanha a
probabilidade real de acerto.

| Métrica | EmoAR | ViT |
|---|---:|---:|
| ECE | 0.0839 | 0.1836 |
| MCE | 0.1477 | 0.3750 |

Além disso:

- 8,07% dos erros do EmoAR ocorreram com confiança >= 90%;
- 45,66% dos erros do ViT ocorreram com confiança >= 90%.

### Resultado

Embora o ViT tenha maior accuracy, sua confiança bruta apresentou
maior descalibração no protocolo utilizado.

Isso é relevante caso a confiança do modelo venha a ser usada como
gatilho para uma interação futura em RA.

---

## 5. Robustez a pequenas perturbações

Como o FER2013 contém imagens estáticas, foi realizado um experimento
controlado para verificar se pequenas mudanças na imagem alteram a
categoria prevista.

Foram utilizadas:

- 350 imagens balanceadas;
- 50 imagens por classe;
- 8 perturbações por imagem;
- 2.800 comparações original-perturbada por modelo.

Foram testadas pequenas alterações de:

- rotação;
- translação;
- brilho;
- contraste.

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Accuracy perturbada | 0.6257 | 0.6868 |
| Pairwise stability | 0.9089 | 0.8764 |
| Fully stable fraction | 0.6886 | 0.5543 |

### Resultado

O ViT continuou mais acurado, porém apresentou mais mudanças na
categoria prevista diante de algumas pequenas perturbações,
principalmente rotação e translação.

Este experimento mede robustez controlada e não estabilidade temporal
real em vídeo.

---

## Síntese

Até o momento, os experimentos indicam o seguinte trade-off:

### ViT

- melhor desempenho classificatório;
- maior custo computacional;
- maior latência;
- confiança softmax mais descalibrada;
- menor estabilidade de categoria em algumas perturbações.

### EmoAR

- menor desempenho classificatório;
- menor custo computacional;
- menor latência;
- melhor calibração relativa;
- maior estabilidade agregada nas perturbações testadas.

---

## O que ainda não foi feito

Até o momento não foi realizada:

- implementação de uma aplicação completa de RA;
- avaliação em vídeo temporal real;
- avaliação com crianças;
- avaliação específica com crianças com TEA;
- integração completa com face detection/tracking;
- validação clínica;
- avaliação em dataset infantil externo.

---

## Material disponível no repositório

GitHub:

https://github.com/lukasales/REBECA-AR-Computing

O repositório contém:

- scripts utilizados nos experimentos;
- protocolos experimentais;
- resultados em CSV, JSON e TXT;
- documentação das análises;
- figuras preliminares;
- tabela consolidada de resultados;
- histórico Git de cada etapa.

---

## Pontos que precisamos validar em equipe

Antes de continuar a parte computacional, precisamos decidir:

1. Essa análise está alinhada com o escopo atual do artigo?
2. A comparação EmoAR x ViT deve entrar como parte central ou como
   complemento da análise de aplicabilidade?
3. Quais dimensões queremos levar para a Entrega 3:
   - desempenho;
   - custo/latência;
   - calibração;
   - robustez?
4. Algum desses experimentos deve ser removido ou aprofundado?
5. Precisamos de algum experimento adicional antes do primeiro draft?
6. A equipe considera necessário avançar agora para webcam/RA ou é
   melhor consolidar primeiro os resultados já obtidos?

## Observação

Os resultados e figuras atuais devem ser tratados como preliminares
até a validação da equipe.

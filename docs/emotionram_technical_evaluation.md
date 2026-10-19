# EmotionRAM Faces — avaliação técnica

## Status

O EmotionRAM Faces foi investigado como tecnologia adicional
relacionada à contribuição computacional.

## Modelo

- arquitetura: EfficientNet-B0;
- treinamento declarado pelo repositório: AfeW;
- entrada: RGB 224x224;
- normalização ImageNet;
- classes: Anger, Contempt, Disgust, Fear, Happiness,
  Neutral, Sadness e Surprise;
- número de parâmetros: 4,017,796;
- checkpoint SHA256: `47c1423f3e6f50e3750bf7b0eda7db947c9ce0c2637e1766bf2187eddc652b17`.

## Compatibilidade com o benchmark

Sete das oito classes possuem correspondência direta com o
espaço de classes FER2013 utilizado nos experimentos anteriores.

A classe adicional `Contempt` não existe no protocolo FER2013.

Por isso, esta avaliação não deve ser apresentada como uma
comparação perfeitamente equivalente com EmoAR e ViT.

Ela é uma avaliação cross-dataset/off-the-shelf, pois o
EmotionRAM foi treinado em AfeW e avaliado aqui no FER2013.

## FER2013 privateTest

- imagens: 3589;
- accuracy: 0.5570;
- balanced accuracy: 0.5220;
- macro-F1: 0.5092;
- mean confidence: 0.6468;
- predições Contempt: 52;
- fração Contempt: 1.4489%.

Predições `Contempt` são contabilizadas como incorretas no
protocolo de sete classes.

## Capacidade live

O repositório original contém `webcam.py`.

O pipeline utiliza:

camera -> OpenCV -> MediaPipe face detection -> face ROI ->
EfficientNet-B0 -> classe de expressão -> overlay.

A presença dessa implementação torna o EmotionRAM uma opção
relevante para investigação futura do protótipo live/demo.

A existência do código não significa que a execução da webcam
tenha sido validada neste experimento.

## Interpretação

Os resultados quantitativos do EmotionRAM devem permanecer
separados da comparação principal EmoAR x ViT ou serem
explicitamente identificados como avaliação cross-dataset.

Sua principal relevância adicional para o projeto está na
disponibilidade de um pipeline de webcam já implementado.


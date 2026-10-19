# Status da reprodução live do EmotionRAM

## Pipeline

O código original implementa:

Webcam -> OpenCV -> MediaPipe -> ROI facial -> EfficientNet-B0 ->
classe de expressão -> overlay.

## Compatibilidade encontrada

Para reprodução no ambiente Python 3.11 atual:

- `mediapipe==0.10.21`;
- `opencv-contrib-python==4.11.0.86`;
- `timm==0.9.7`.

O checkpoint completo também requer, com versões novas do PyTorch:

`torch.load(..., weights_only=False)`

Esta alteração deve ser aplicada apenas quando o checkpoint for
obtido de fonte confiável.

## Estado do teste

O modelo e o pipeline foram inicializados até a etapa de captura.

Na máquina utilizada durante esta sessão não foram recebidos frames
físicos da câmera.

Resultado observado:

`[INFO] Skipping due to lack of frames.`

Assim, o status é:

**pipeline preparado; validação visual end-to-end com webcam
pendente.**

O próximo teste será realizado em um notebook com câmera.

## Observação sobre ambientes

O benchmark principal foi originalmente executado com
`numpy==2.4.6`.

O MediaPipe 0.10.21 exige `numpy<2` e instalou `numpy==1.26.4`
durante o teste live.

Para evitar conflito de reprodutibilidade, recomenda-se utilizar
um ambiente virtual separado para a reprodução live do EmotionRAM.


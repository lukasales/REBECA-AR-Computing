# EmoAR Reproducibility Audit

## Reference repository

Repository:
https://github.com/kshntn/EmoAR

Analyzed commit:
33d0c6b46775a830744625ee97f736b3255b8060

Commit description:
33d0c6b - Update README.md

## Historical environment

The web application declares:

- Python 3.7.1
- PyTorch 1.0.0
- torchvision 0.2.1
- NumPy 1.15.4
- Pillow 5.3.0
- Flask 1.0.2
- gunicorn 19.9.0

The declared PyTorch dependency is a Linux CPython 3.7 CPU wheel.

Therefore, the original environment will not be installed directly
into the current Windows/Python 3.11 development environment.

## Model artifact

File:
web_app/classifier.pt

SHA-256:
01DC84D15D836DC8C6552277D7AC0C376086BAA00F1490FD5ACB63CFEF72E5C9

Original Git commit introducing the file:
535c61b8380f1b14214eed738bc940c98b69abfb

File size:
28,334,901 bytes

## Reported model architecture

According to web_app/commons.py:

- torchvision DenseNet121
- ImageNet-pretrained initialization
- final classifier replaced by Linear(1024, 7)
- checkpoint loaded on CPU
- model placed in evaluation mode

The repository README reports training on FER2013.

## Original preprocessing

The web application applies:

1. Resize(255)
2. CenterCrop(224)
3. ToTensor()
4. ImageNet normalization

Mean:
[0.485, 0.456, 0.406]

Standard deviation:
[0.229, 0.224, 0.225]

The web code does not explicitly convert input images to RGB.

## Output classes

FER2013 class order reported by the README:

0 - Angry
1 - Disgust
2 - Fear
3 - Happy
4 - Sad
5 - Surprise
6 - Neutral

## Mapping artifact

The web application contains class_to_idx.json and cat_to_name.json
apparently inherited from a 102-class flower-classification example.

The model itself produces only seven outputs.

The first seven reverse mappings resolve to:

0 -> "1"   -> Angry
1 -> "10"  -> Disgust
2 -> "100" -> Fear
3 -> "101" -> Happy
4 -> "102" -> Sad
5 -> "11"  -> Surprise
6 -> "12"  -> Neutral

This reproduces the FER2013 ordering, but introduces unnecessary
and potentially confusing legacy mappings.

A modern reproduction should replace this mechanism with an
explicit seven-class mapping.

## Reproducibility concerns identified

1. Historical Python/PyTorch dependencies are incompatible with
   the current development environment.

2. load_state_dict() is called with strict=False.

   This may silently accept missing or unexpected checkpoint keys.

   The reproduction must inspect and record all missing and
   unexpected keys rather than ignoring them.

3. The inference code does not expose class probabilities or
   prediction confidence.

4. The inference code does not use torch.no_grad() or
   torch.inference_mode().

5. Input images are not explicitly converted to RGB.

6. The web application does not perform face detection or cropping.
   It assumes an uploaded image is directly suitable for classification.

7. The Android application and web application therefore represent
   different inference pipelines.

8. Legacy flower-classification names remain in code and metadata,
   including the function name get_flower_name().

## Licensing

No LICENSE, COPYING, COPYRIGHT, LICENSE.txt or LICENSE.md file was
identified in the analyzed repository snapshot.

The repository and model will therefore be treated as research
references.

Their source code and model weights will not be redistributed
inside the REBECA-AR Computing repository unless reuse rights are
clarified.

## Next audit question

Before attempting model execution, determine:

- which training architecture generated classifier.pt;
- the exact preprocessing used during training;
- whether grayscale FER2013 images were replicated/converted to RGB;
- whether the saved artifact is a complete state_dict;
- whether DenseNet121 was the selected final web model;
- the training/validation procedure associated with this checkpoint.


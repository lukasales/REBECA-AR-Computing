from pathlib import Path
import csv
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"

TEMP = RESULTS / "temporal_video_summary_aggregate.csv"
PAPER_MD = RESULTS / "paper_summary_table.md"
PAPER_CSV = RESULTS / "paper_summary_table.csv"
COMP_DOC = DOCS / "computational_results_for_paper.md"
TEAM_DOC = DOCS / "team_review_summary.md"
HANDOFF = DOCS / "implementation_handoff.md"
MESSAGE = DOCS / "implementation_handoff_message.txt"
AUDIT = DOCS / "final_implementation_audit.md"

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def pct(x):
    return f"{float(x) * 100:.2f}%"

print("========== LOAD RESULTS ==========")

temporal_rows = read_csv(TEMP)
temporal = {r["model"].lower(): r for r in temporal_rows}

emo = temporal["emoar"]
vit = temporal["vit"]

assert int(emo["n_frames"]) == 1620
assert int(vit["n_frames"]) == 1620

print("EmoAR frames:", emo["n_frames"])
print("ViT frames:", vit["n_frames"])

print("\n========== UPDATE PAPER SUMMARY ==========")

paper_text = PAPER_MD.read_text(encoding="utf-8-sig")

temporal_block = f"""
| Temporal frames | 1620 | 1620 |
| Temporal class switches | {emo["switches"]} | {vit["switches"]} |
| Temporal flickering/change rate | {float(emo["change_rate"]):.4f} | {float(vit["change_rate"]):.4f} |
| Temporal consecutive agreement | {float(emo["consecutive_agreement"]):.4f} | {float(vit["consecutive_agreement"]):.4f} |
| Temporal mean |Δ confidence| | {float(emo["mean_abs_conf_delta"]):.4f} | {float(vit["mean_abs_conf_delta"]):.4f} |
| Temporal P95 |Δ confidence| | {float(emo["p95_abs_conf_delta"]):.4f} | {float(vit["p95_abs_conf_delta"]):.4f} |
| Temporal mean latency (ms) | {float(emo["mean_latency_ms"]):.2f} | {float(vit["mean_latency_ms"]):.2f} |
| Temporal effective classifier FPS | {float(emo["effective_classifier_fps"]):.2f} | {float(vit["effective_classifier_fps"]):.2f} |
""".strip()

if "Temporal frames" not in paper_text:
    paper_text = paper_text.rstrip() + "\n" + temporal_block + "\n"

PAPER_MD.write_text(paper_text, encoding="utf-8")

rows = read_csv(PAPER_CSV)

for row in rows:
    name = row["model"].lower()
    source = emo if "emoar" in name else vit

    row["temporal_frames"] = source["n_frames"]
    row["temporal_switches"] = source["switches"]
    row["temporal_change_rate"] = source["change_rate"]
    row["temporal_consecutive_agreement"] = source["consecutive_agreement"]
    row["temporal_mean_abs_conf_delta"] = source["mean_abs_conf_delta"]
    row["temporal_p95_abs_conf_delta"] = source["p95_abs_conf_delta"]
    row["temporal_mean_latency_ms"] = source["mean_latency_ms"]
    row["temporal_effective_classifier_fps"] = source["effective_classifier_fps"]

fields = list(rows[0].keys())

with PAPER_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("[OK] paper summary updated")

print("\n========== UPDATE COMPUTATIONAL DOC ==========")

doc = COMP_DOC.read_text(encoding="utf-8-sig")

if "- estabilidade temporal em vídeo." not in doc:
    doc = doc.replace(
        "- robustez a pequenas perturbações visuais.",
        "- robustez a pequenas perturbações visuais;\n- estabilidade temporal em vídeo."
    )

temporal_section = f"""
## Avaliação temporal em vídeo

Foi executado um benchmark temporal utilizando 14 vídeos do RAVDESS,
com 2 atores e 7 categorias compatíveis com os classificadores.

Foram efetivamente decodificados 1.620 frames. Cada frame foi
processado pelos dois modelos, totalizando 3.240 previsões.

Foi utilizado crop facial fixo por vídeo, com escala 1.7, sem
temporal smoothing.

| Métrica | EmoAR | ViT |
|---|---:|---:|
| Class switches | {emo["switches"]} | {vit["switches"]} |
| Flickering/change rate | {pct(emo["change_rate"])} | {pct(vit["change_rate"])} |
| Consecutive agreement | {pct(emo["consecutive_agreement"])} | {pct(vit["consecutive_agreement"])} |
| Mean confidence | {float(emo["mean_confidence"]):.4f} | {float(vit["mean_confidence"]):.4f} |
| Mean abs. confidence delta | {float(emo["mean_abs_conf_delta"]):.4f} | {float(vit["mean_abs_conf_delta"]):.4f} |
| P95 abs. confidence delta | {float(emo["p95_abs_conf_delta"]):.4f} | {float(vit["p95_abs_conf_delta"]):.4f} |
| Mean latency | {float(emo["mean_latency_ms"]):.2f} ms | {float(vit["mean_latency_ms"]):.2f} ms |
| Effective classifier FPS | {float(emo["effective_classifier_fps"]):.2f} | {float(vit["effective_classifier_fps"]):.2f} |

O ViT apresentou ligeiramente menos flickering agregado, porém
maior latência e maior variação temporal da confiança.

Os rótulos do RAVDESS são rótulos de clipe e não são utilizados
como ground truth frame a frame.

A avaliação utiliza vídeos previamente gravados. Ela não constitui
uma aplicação de webcam ou RA em tempo real.
""".strip()

if "## Avaliação temporal em vídeo" not in doc:
    marker = "## 9. Síntese do trade-off"

    if marker in doc:
        doc = doc.replace(
            marker,
            temporal_section + "\n\n" + marker
        )
    else:
        doc = doc.rstrip() + "\n\n" + temporal_section + "\n"

doc = doc.replace(
    "- figures/05_perturbation_stability.png.",
    "- figures/05_perturbation_stability.png;\n"
    "- figures/06_temporal_applicability_tradeoff.png."
)

doc = doc.replace(
    "- os experimentos atuais utilizam imagens estáticas;\n",
    ""
)

doc = doc.replace(
    "- a avaliação de perturbações não representa estabilidade temporal real;\n",
    "- a avaliação de perturbações é distinta da avaliação temporal em vídeo;\n"
)

old_gap = """## 13. Próxima lacuna computacional

A principal lacuna experimental atual é a ausência de avaliação em sequências de vídeo.

O próximo experimento deverá investigar estabilidade temporal, mudanças de classe entre frames, variação de confiança e latência em fluxo contínuo."""

new_gap = """## 13. Estado atual e próxima lacuna computacional

A avaliação temporal em vídeos previamente gravados foi concluída.

A implementação atual constitui um benchmark computacional
reproduzível de aplicabilidade.

Ainda não foi implementado um protótipo live com webcam, detecção
ou tracking facial em tempo real, feedback visual ou integração
com uma experiência completa de Realidade Aumentada."""

doc = doc.replace(old_gap, new_gap)

COMP_DOC.write_text(doc, encoding="utf-8")

print("[OK] computational document updated")

print("\n========== CREATE TEAM SUMMARY ==========")

team = f"""# Resumo final da implementação computacional — Entrega 4

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
| Temporal flickering | {pct(emo["change_rate"])} | {pct(vit["change_rate"])} |
| Temporal agreement | {pct(emo["consecutive_agreement"])} | {pct(vit["consecutive_agreement"])} |
| Temporal FPS efetivo | {float(emo["effective_classifier_fps"]):.2f} | {float(vit["effective_classifier_fps"]):.2f} |

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
"""

TEAM_DOC.write_text(team + "\n", encoding="utf-8")

print("[OK] team summary updated")

print("\n========== CREATE HANDOFF ==========")

handoff = f"""# Handoff da implementação computacional — Entrega 4

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
| Class switches | {emo["switches"]} | {vit["switches"]} |
| Flickering | {pct(emo["change_rate"])} | {pct(vit["change_rate"])} |
| Consecutive agreement | {pct(emo["consecutive_agreement"])} | {pct(vit["consecutive_agreement"])} |
| Mean |Δ confidence| | {float(emo["mean_abs_conf_delta"]):.4f} | {float(vit["mean_abs_conf_delta"]):.4f} |
| P95 |Δ confidence| | {float(emo["p95_abs_conf_delta"]):.4f} | {float(vit["p95_abs_conf_delta"]):.4f} |
| Mean latency | {float(emo["mean_latency_ms"]):.2f} ms | {float(vit["mean_latency_ms"]):.2f} ms |
| Effective FPS | {float(emo["effective_classifier_fps"]):.2f} | {float(vit["effective_classifier_fps"]):.2f} |

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
"""

HANDOFF.write_text(handoff + "\n", encoding="utf-8")

print("[OK] implementation handoff created")

message = f"""Oi! Finalizei a parte computacional para a Entrega 4.

Além da comparação EmoAR x ViT no FER2013, finalizei também a
avaliação temporal com 14 vídeos RAVDESS, 1.620 frames e 3.240
previsões.

Principais resultados:
- accuracy: 66,51% EmoAR x 71,75% ViT;
- latência CPU: 70,60 ms x 151,34 ms;
- ECE: 0,0839 x 0,1836;
- estabilidade às perturbações: 90,89% x 87,64%;
- flickering temporal: {pct(emo["change_rate"])} x {pct(vit["change_rate"])};
- FPS temporal: {float(emo["effective_classifier_fps"]):.2f} x {float(vit["effective_classifier_fps"]):.2f}.

Deixei todos os números, limitações e figuras organizados em:
docs/implementation_handoff.md

Detalhes técnicos completos:
docs/computational_results_for_paper.md

Repositório:
https://github.com/lukasales/REBECA-AR-Computing

A implementação atual processa imagens e vídeos gravados frame a
frame. Webcam/RA ao vivo ainda não foi implementada.
"""

MESSAGE.write_text(message, encoding="utf-8")

print("[OK] handoff message created")

print("\n========== FINAL AUDIT ==========")

checks = {
    "handoff": HANDOFF.exists(),
    "message": MESSAGE.exists(),
    "paper summary md": PAPER_MD.exists(),
    "paper summary csv": PAPER_CSV.exists(),
    "computational doc": COMP_DOC.exists(),
    "temporal figure":
        (ROOT / "figures" / "06_temporal_applicability_tradeoff.png").exists(),
    "raw temporal predictions":
        (RESULTS / "temporal_video_predictions.csv").exists(),
}

lines = [
    "# Final implementation audit",
    "",
    "| Check | Status |",
    "|---|---|",
]

for name, ok in checks.items():
    status = "OK" if ok else "FAIL"
    print(f"{status:4s} | {name}")
    lines.append(f"| {name} | {status} |")

if not all(checks.values()):
    raise RuntimeError("Final audit failed.")

lines += [
    "",
    "Experimental benchmark: complete.",
    "",
    "Live webcam prototype: not implemented.",
    "",
    "Full AR application: not implemented.",
]

AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\nFINAL IMPLEMENTATION AUDIT: OK")

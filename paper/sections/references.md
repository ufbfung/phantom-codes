# References

> **Status:** Curated bibliography v0 (2026-05-01). Citation keys map to
> `[Author Year]` callouts in the paper sections. Convert to BibTeX once we
> commit to LaTeX. Annotations summarize *why this paper is cited* — strip
> on conversion.

---

## A. Foundational landmarks (cited for framing, not all recent)

### [PhysioNet 2025]
PhysioNet (2025). *Responsible Use of Large Language Models with PhysioNet
Data.* PhysioNet news, effective 2025-09-24.
[physionet.org/news/post/llm-responsible-use](https://physionet.org/news/post/llm-responsible-use/)

> **Why we cite:** This policy's prohibition on sending credentialed data
> through third-party APIs is the design constraint motivating our
> train-on-MIMIC / evaluate-on-Synthea separation. Cited in the methods
> section's data-flow description and the discussion's
> compliance-by-construction framing.



### [Mullenbach 2018]
Mullenbach, J., Wiegreffe, S., Duke, J., Sun, J., & Eisenstein, J. (2018).
*Explainable Prediction of Medical Codes from Clinical Text* (CAML). NAACL.
[aclanthology.org/N18-1100](https://aclanthology.org/N18-1100/)

> **Why we cite:** Source of the hierarchical-match terminology
> (chapter-/category-level matches). Defacto reference for the deep-learning
> approach to ICD coding from MIMIC discharge summaries that pre-LLM work
> built on.

### [Huang 2022]
Huang, C.-W., Tsai, S.-C., & Chen, Y.-N. (2022). *PLM-ICD: Automatic ICD
Coding with Pretrained Language Models.* Proceedings of the 4th Clinical
NLP Workshop (NAACL 2022). [aclanthology.org/2022.clinicalnlp-1.2](https://aclanthology.org/2022.clinicalnlp-1.2/)

> **Why we cite:** Defacto BERT-baseline reference for ICD coding on MIMIC.
> Reports micro-F1 62.6% (ICD-9) and 58.5% (ICD-10) on MIMIC-IV using
> domain-pretrained encoders. Anchors the "trained classifier" comparison
> arm of our benchmark.

### [Hendrycks 2017]
Hendrycks, D., & Gimpel, K. (2017). *A Baseline for Detecting Misclassified
and Out-of-Distribution Examples in Neural Networks.* ICLR.
[arXiv:1610.02136](https://arxiv.org/abs/1610.02136)

> **Why we cite:** Foundational reference for the OOD framing in our
> `out_of_domain` outcome bucket.

### [Ji 2023]
Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.,
Madotto, A., & Fung, P. (2023). *Survey of Hallucination in Natural Language
Generation.* ACM Computing Surveys, 55(12).
[doi:10.1145/3571730](https://dl.acm.org/doi/10.1145/3571730)

> **Why we cite:** Defacto definition and taxonomy of hallucination in NLG
> systems. Anchors our use of "fabrication" / "hallucination" terminology.

### [Singhal 2023]
Singhal, K., et al. (2023). *Large language models encode clinical
knowledge.* Nature 620, 172–180.
[doi:10.1038/s41586-023-06291-2](https://www.nature.com/articles/s41586-023-06291-2)

> **Why we cite:** Landmark paper establishing that general-purpose LLMs
> can perform medical reasoning at clinically meaningful levels. Cited for
> the broader context that LLMs are being deployed in clinical workflows,
> motivating careful evaluation of their failure modes.

---

## B. Direct LLM-on-medical-coding evaluation (2024-2026)

### [Soroush 2024]
Soroush, A., Glicksberg, B. S., Zimlichman, E., Barash, Y., Freeman, R.,
Charney, A. W., Nadkarni, G. N., & Klang, E. (2024). *Large Language Models
Are Poor Medical Coders — Benchmarking of Medical Code Querying.* NEJM AI.
[doi:10.1056/AIdbp2300040](https://ai.nejm.org/doi/full/10.1056/AIdbp2300040)

> **Why we cite:** THE landmark for direct LLM evaluation on medical code
> querying. GPT-4 achieved 46% exact match on ICD-9, 34% on ICD-10, 50% on
> CPT — with substantial fractions of errors being non-existent codes. Mt
> Sinai group; NEJM AI publication = highest-prestige venue in this area.
> Direct empirical anchor for our hallucination motivation.

### [Yang 2023]
Yang, Z., Wang, S., Rawat, B. P. S., Mitra, A., & Yu, H. (2023).
*Surpassing GPT-4 Medical Coding with a Two-Stage Approach.*
[arXiv:2311.13735](https://arxiv.org/abs/2311.13735)

> **Why we cite:** LLM-codex two-stage architecture (LLM proposer + LSTM
> verifier). Demonstrates that LLM-only approaches can be improved by
> coupling with discriminative components. Counterpoint to our zero-shot
> measurements.

### [Goel 2024]
Goel, A., Schwartz, S., Larochelle, F., Eppes, R., & Klein, D. (2024).
*Can Large Language Models Abstract Medical Coded Language?*
[arXiv:2403.10822](https://arxiv.org/abs/2403.10822)

> **Why we cite:** Tests whether LLMs (GPT, LLaMA-2, Meditron) genuinely
> understand medical code ontologies and can distinguish real from
> fabricated codes. Direct empirical support for the hallucination concern
> we operationalize.

### [Sahaj 2024]
Sahaj, S., et al. (2024). *MedCodER: A Generative AI Assistant for Medical
Coding.* [arXiv:2409.15368](https://arxiv.org/abs/2409.15368)

> **Why we cite:** Extract-retrieve-rerank pipeline for ICD-10 coding —
> represents the modular response to LLM hallucination. Cited as a
> contrasting architecture in the "what's been done" review.

### [Mahmoud 2025]
Mahmoud, A., et al. (2025). *Challenges and Solutions in Applying Large
Language Models to Guideline-Based Management Planning and Automated Medical
Coding in Health Care: Algorithm Development and Validation.* JMIR
Biomedical Engineering, 10:e66691.
[doi:10.2196/66691](https://biomedeng.jmir.org/2025/1/e66691)

> **Why we cite:** Introduces GAVS (Generation-Assisted Vector Search) on
> 958 MIMIC-IV admissions — closest methodological cousin to our
> constrained-mode setup. Inverts standard RAG: generate clinical entities
> first, then match against the coding ontology vector index.

### [Williams 2024]
Williams, K., et al. (2024). *Large Language Models are good medical
coders, if provided with tools.*
[arXiv:2407.12849](https://arxiv.org/abs/2407.12849)

> **Why we cite:** Tool-augmented LLM coding. Companion finding to our
> zero-shot vs. constrained ablation: tools reduce hallucination by
> grounding predictions in real ontology lookups.

### [Almeida 2025]
Almeida, V. A., de Camargo, V., Gómez-Bravo, R., van der Haring, E., van
Boven, K., Finger, M., & Lopez, L. F. (2025). *Large Language Models as
Medical Code Selectors: a benchmark using the International Classification
of Primary Care.* NeurIPS 2025 Workshop on GenAI for Health.
[arXiv:2507.14681](https://arxiv.org/abs/2507.14681)

> **Why we cite:** 33-LLM benchmark with semantic retrieval + LLM selector
> on ICPC-2 codes. Inspirations for our multi-axis evaluation table (F1 +
> cost + latency + format adherence).

### [Bhatti 2025]
Bhatti, U., et al. (2025). *MAX-EVAL-11: A Comprehensive Benchmark for
Evaluating Large Language Models on Full-Spectrum ICD-11 Medical Coding.*
medRxiv. [doi:10.1101/2025.10.30.25339130](https://www.medrxiv.org/content/10.1101/2025.10.30.25339130v1)

> **Why we cite:** MIMIC-III mapped to ICD-11 (10k notes); clinically
> informed weighted scoring by code relevance ranking and diagnostic
> specificity. Inspires the optional weighted-score variant in our
> backlog.

### [Motzfeldt 2025]
Motzfeldt, A., Edin, J., Christensen, C. L., Hardmeier, C., Maaløe, L., &
Rogers, A. (2025). *Code Like Humans: A Multi-Agent Solution for Medical
Coding.* Findings of EMNLP 2025.
[arXiv:2509.05378](https://arxiv.org/abs/2509.05378)

> **Why we cite:** Agentic framework that traverses the ICD-10 alphabetic
> index sequentially. First solution covering full 70k+ ICD-10 labels.
> Strongest existing case for agent-based decomposition as a
> hallucination-mitigation strategy. Will inform our optional "agentic"
> third prompting mode.

### [Hybrid-Code v2 2025]
Authors TBD (2025). *Hybrid-Code v2: Zero-Hallucination Clinical ICD-10
Coding via Neuro-Symbolic Verification and Automated Knowledge Base
Expansion.* [arXiv:2512.23743](https://arxiv.org/html/2512.23743)

> **Why we cite:** Reports 0% Type-I hallucination on MIMIC-III via
> neuro-symbolic verification, vs. 6–18% in neural baselines. Strongest
> claim to date that hallucination can be eliminated via constrained
> generation. Direct comparator if our zero-shot LLMs hallucinate at
> rates within their reported neural-baseline band.

### [Li 2025]
Li, M., Schlegel, V., Mu, T., Oyewusi, W., Kang, K., & Nenadic, G. (2025).
*Evaluation and LLM-Guided Learning of ICD Coding Rationales.*
[arXiv:2508.16777](https://arxiv.org/abs/2508.16777)

> **Why we cite:** Faithfulness/plausibility rationale evaluation on
> MIMIC-IV/ICD-10. Inspires our v2 plan to evaluate per-prediction
> rationales.

---

## C. Hallucination-focused evaluations of LLMs in clinical settings

### [Kim 2025]
Kim, Y., et al. (2025). *Medical Hallucinations in Foundation Models and
Their Impact on Healthcare.* [arXiv:2503.05777](https://arxiv.org/abs/2503.05777)

> **Why we cite:** Defines medical hallucination as model output that is
> "factually incorrect, logically inconsistent, or unsupported by
> authoritative clinical evidence in ways that could alter clinical
> decisions." Evaluates 11 foundation models (7 general, 4 medical) across
> 7 hallucination tasks. Our `hallucination` bucket is a narrow,
> mechanically-checkable instance of their broader definition.

### [Omiye 2025]
Omiye, J. A., et al. (2025). *Multi-model assurance analysis showing large
language models are highly vulnerable to adversarial hallucination attacks
during clinical decision support.* Communications Medicine.
[doi:10.1038/s43856-025-01021-3](https://www.nature.com/articles/s43856-025-01021-3)

> **Why we cite:** Adversarial-prompt study of 6 leading LLMs on 300
> doctor-designed clinical vignettes containing planted fake lab values,
> signs, or diseases. Hallucination elaboration in 50–82% of cases;
> mitigation prompts only halve the rate. Strongest evidence to date that
> LLM hallucination in clinical settings is robust to prompt-level
> mitigation.

### [Hatem 2025]
Hatem, R., et al. (2025). *A framework to assess clinical safety and
hallucination rates of LLMs for medical text summarisation.* npj Digital
Medicine. [doi:10.1038/s41746-025-01670-7](https://www.nature.com/articles/s41746-025-01670-7)

> **Why we cite:** 18-experiment evaluation framework with 450 consultation
> transcript-note pairs (49,590 sentences manually evaluated). Reports
> 1.47% hallucination rate, 3.45% omission rate. Methodologically careful
> baseline for what hallucination evaluation looks like at scale in a
> clinical task.

### [Gershon 2025]
Gershon, A., Soffer, S., Nadkarni, G. N., & Klang, E. (2025). *Automatic
ICD coding using LLMs: a systematic review.* medRxiv.
[doi:10.1101/2025.07.30.25330916](https://www.medrxiv.org/content/10.1101/2025.07.30.25330916v1)

> **Why we cite:** Systematic review of 35 LLM-ICD-coding studies
> identified through January 2025. Best single citation for "the field as
> of mid-2025." Conclusion: LLMs reliably automate common codes but
> degrade on rare diagnoses; external validation is scant; prospective
> multicenter trials needed.

---

## E. Trained-model methodology references (added 2026-05-02 with §01_methodology)

### [Johnson 2023]
Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A.,
Horng, S., Pollard, T. J., Hao, S., Moody, B., Gow, B., Lehman, L. H.,
Celi, L. A., & Mark, R. G. (2023). *MIMIC-IV, a freely accessible
electronic health record dataset.* Scientific Data, 10(1), 1.
[doi:10.1038/s41597-022-01899-x](https://www.nature.com/articles/s41597-022-01899-x)

> **Why we cite:** Canonical citation for the MIMIC-IV dataset. The
> FHIR-formatted v2.1 release (which we use) wraps this same source data.

### [Walonoski 2018]
Walonoski, J., Kramer, M., Nichols, J., Quina, A., Moesel, C., Hall, D.,
Duffett, C., Dube, K., Gallagher, T., & McLachlan, S. (2018). *Synthea:
An approach, method, and software mechanism for generating synthetic
patients and the synthetic electronic health care record.* Journal of
the American Medical Informatics Association, 25(3), 230–238.
[doi:10.1093/jamia/ocx079](https://academic.oup.com/jamia/article/25/3/230/4098271)

> **Why we cite:** Source citation for Synthea, the synthetic FHIR
> generator that produces our headline-evaluation cohort. Synthea data
> is freely redistributable and contains no real patient information,
> which is what enables the LLM-evaluation arm of the comparison.

### [CMS 2026]
Centers for Medicare & Medicaid Services. (2026). *ACCESS Model FHIR
Implementation Guide v0.9.6.*
[dsacms.github.io/cmmi-access-model](https://dsacms.github.io/cmmi-access-model/)

> **Why we cite:** Defines the CKM and eCKM disease-group scope (diabetes,
> ASCVD, CKD-3, hypertension, dyslipidemia, prediabetes, obesity) we
> use to restrict the cohort. The IG bundles ValueSets we consume
> directly (`ACCESSCKMDiagnosisVS`, `ACCESSeCKMDiagnosisVS`).

### [Devlin 2019]
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding.* NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

> **Why we cite:** The architecture and the pre-training-then-fine-tuning
> paradigm we follow. Establishes the canonical [CLS]-token classification
> head and the 2e-5 fine-tuning learning rate.

### [Gu 2021]
Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann,
T., Gao, J., & Poon, H. (2021). *Domain-Specific Language Model
Pretraining for Biomedical Natural Language Processing.* ACM Transactions
on Computing for Healthcare, 3(1), 1–23.
[doi:10.1145/3458754](https://dl.acm.org/doi/10.1145/3458754)

> **Why we cite:** Introduces PubMedBERT (our chosen base encoder) and
> the BLURB benchmark. Establishes that pre-training from scratch on
> biomedical text outperforms continued pre-training of general BERT.

### [Gururangan 2020]
Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I.,
Downey, D., & Smith, N. A. (2020). *Don't Stop Pretraining: Adapt
Language Models to Domains and Tasks.* ACL.
[arXiv:2004.10964](https://arxiv.org/abs/2004.10964)

> **Why we cite:** The general principle that domain-adaptive pre-training
> beats general pre-training for in-domain downstream tasks. Frames why
> PubMedBERT (biomedical pre-train) wins over standard BERT for our task.

### [Alsentzer 2019]
Alsentzer, E., Murphy, J. R., Boag, W., Weng, W.-H., Jin, D., Naumann,
T., & McDermott, M. (2019). *Publicly Available Clinical BERT
Embeddings.* Proceedings of the 2nd Clinical NLP Workshop (NAACL).
[arXiv:1904.03323](https://arxiv.org/abs/1904.03323)

> **Why we cite:** ClinicalBERT — pre-trained on MIMIC-III clinical notes.
> Cited as the alternative we explicitly *reject* due to data
> contamination: using a model that has seen MIMIC-III text to classify
> MIMIC-IV conditions would produce optimistic numbers.

### [Lee 2020]
Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J.
(2020). *BioBERT: a pre-trained biomedical language representation model
for biomedical text mining.* Bioinformatics, 36(4), 1234–1240.
[doi:10.1093/bioinformatics/btz682](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)

> **Why we cite:** BioBERT — alternative biomedical encoder, pre-trained
> on PubMed abstracts only (vs. PubMedBERT's abstracts + full-text).
> Listed as a defensible alternative for v2 sensitivity analysis.

### [Yasunaga 2022]
Yasunaga, M., Leskovec, J., & Liang, P. (2022). *LinkBERT: Pretraining
Language Models with Document Links.* ACL.
[arXiv:2203.15827](https://arxiv.org/abs/2203.15827)

> **Why we cite:** BioLinkBERT — biomedical encoder pre-trained on linked
> documents (PubMed citation graph). Strong on multi-hop reasoning;
> unclear gain for single-hop classification. Listed as v2 ablation
> candidate.

### [Chalkidis 2020]
Chalkidis, I., Fergadiotis, M., Kotitsas, S., Malakasiotis, P., Aletras,
N., & Androutsopoulos, I. (2020). *An Empirical Study on Large-Scale
Multi-Label Text Classification Including Few and Zero-Shot Labels.*
EMNLP. [arXiv:2010.01653](https://arxiv.org/abs/2010.01653)

> **Why we cite:** Establishes the standard trade-off between vocabulary
> coverage and gradient signal density in extreme multi-label
> classification. Justifies our top-50 vocabulary cap for v1.

### [Hu 2022]
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang,
L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language
Models.* ICLR. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

> **Why we cite:** Parameter-efficient fine-tuning method. Cited in v2
> roadmap as the path to fine-tuning larger biomedical encoders on
> hardware similar to ours.

### [Dettmers 2023]
Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).
*QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.
[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

> **Why we cite:** 4-bit quantized LoRA — the practical extension of LoRA
> that enables fine-tuning 7B+ parameter biomedical LLMs on a single
> consumer GPU. v2 roadmap reference.

### [Loshchilov 2019]
Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay
Regularization.* ICLR. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

> **Why we cite:** Introduces AdamW — the optimizer we use. Establishes
> why decoupled weight decay matters for transformer fine-tuning
> (Adam's L2 regularization interacts poorly with adaptive learning
> rates).

### [Liu 2020]
Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., & Han, J.
(2020). *On the Variance of the Adaptive Learning Rate and Beyond.*
ICLR. [arXiv:1908.03265](https://arxiv.org/abs/1908.03265)

> **Why we cite:** Theoretical analysis of why warmup is necessary when
> fine-tuning transformers — without it, adaptive optimizers produce
> high-variance updates in early training that destabilize the
> pre-trained representations.

---

## Notes on coverage

- **Inclusion criteria:** evaluates an LLM (or LLM-augmented system) on a
  clinical concept normalization or coding task; reports at least one
  empirical performance metric; published 2021–2026.
- **Excluded:** general-purpose hallucination benchmarks not in clinical
  settings (HaluEval, etc.); rule-based or pre-LLM coding systems unless
  they are defacto baselines (PLM-ICD, CAML); medical question-answering
  papers where coding is not the task (Med-PaLM is included as
  framing context only).
- **Known gaps in this list:** non-English ICD coding studies; clinical
  trial / structured-data normalization (UMLS metathesaurus mapping);
  SNOMED CT entity linking competition results. Worth a follow-up search
  if reviewers ask for breadth.

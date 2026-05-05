# Architecture

We fine-tune a single transformer encoder with a randomly-initialized
linear classification head over the [CLS] token's final-layer
hidden state. This is the canonical BERT-style classification setup
[@Devlin2019]; we make three deliberate choices on top of it.

## Multi-label rather than single-label

The classification head emits one logit per code (50 logits total),
trained with binary cross-entropy on a sigmoid activation
(`BCEWithLogitsLoss` for numerical stability). While our v1 ground
truth is one code per condition, the multi-label formulation allows
the model to in principle predict multiple codes when comorbidities
are documented in the input text — a property we do not exploit in
v1 but preserve for v2's lab/medication extensions where multi-label
is the norm.

## Top-N vocabulary cap

Restricting the head to the 50 most-frequent codes (rather than all
178 observed codes) trades coverage for classifier signal density.
The long tail of rare codes (≤10 instances in the train split)
cannot be reliably learned regardless of model choice; including
them would dilute the gradient signal on the high-frequency codes
that drive most clinical decisions. This is a standard trade-off in
extreme classification [@Chalkidis2020]; we err toward the
well-supported subset for v1.

## Single linear head, no MLP

A two-layer MLP head with non-linearity is sometimes used to give
the head capacity to learn task-specific combinations of encoder
features. We use a single linear layer for v1 because it is the
standard baseline and because the encoder's [CLS] embedding already
aggregates sentence-level information through 12 transformer
layers. Adding capacity to the head trades against training-set
memorization risk; the cleaner head minimizes confound.

## Choice of base encoder: PubMedBERT

We use **PubMedBERT-base-uncased-abstract-fulltext** [@Gu2021] as
the pre-trained encoder for three reasons:

(a) **Domain-adaptive pre-training.** PubMedBERT is trained from
scratch on PubMed abstracts + full-text articles, producing a
tokenizer that treats biomedical terms as single tokens rather than
fragmented subwords [@Devlin2019; @Gururangan2020].

(b) **Strong empirical performance** on the BLURB biomedical NLP
benchmark [@Gu2021].

(c) **Computational footprint** (~110M parameters) compatible with
local fine-tuning on consumer Apple Silicon — the
compliance-enabling property. Any larger encoder would have forced
us onto cloud GPU infrastructure with corresponding data-residency
review for credentialed MIMIC data.

Alternative encoders were considered and either ruled out or
deferred to a v2 ablation arm:

- **ClinicalBERT** [@Alsentzer2019] — ruled out because its
  pre-training corpus includes MIMIC-III, which contaminates
  MIMIC-IV evaluation.
- **BioBERT** [@Lee2020] and **BioLinkBERT** [@Yasunaga2022] —
  comparable scale to PubMedBERT, deferred to a future ablation
  arm.
- **Larger biomedical models** (BiomedLM, GatorTron, BioMistral,
  Med-PaLM) via parameter-efficient fine-tuning [@Hu2022;
  @Dettmers2023] — deferred; would require either cloud GPU or
  significantly more local memory.

Generative LLMs (Claude, GPT, Gemini) are evaluated separately in
the companion paper [@FungPhantomCodes2026] as zero-shot /
constrained / RAG arms, not as fine-tuned models.

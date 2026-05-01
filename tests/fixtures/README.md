# Test Fixtures

Hand-curated synthetic FHIR `Condition` resources following the [MimicCondition profile](https://kind-lab.github.io/mimic-fhir/StructureDefinition-mimic-condition.html).

**Important:** These are NOT MIMIC data. They are synthetic resources written by hand to exercise the degradation, loader, and model code paths without requiring PhysioNet credentialing. Patient/Encounter references are placeholders (`fixture-patient-NNN`).

10 fixtures, structured to exercise both ACCESS-scope (in scope for v1) and out-of-scope cases.

| ID | Code | System | Group | Notes |
|----|------|--------|-------|-------|
| fixture-001 | E11.9 | ICD-10 | CKM | Type 2 diabetes — uncomplicated |
| fixture-002 | I10 | ICD-10 | eCKM | Essential hypertension |
| fixture-003-out-of-scope | J18.9 | ICD-10 | — | Pneumonia (not ACCESS scope) |
| fixture-004-out-of-scope | N17.9 | ICD-10 | — | Acute kidney failure (no `clinicalStatus`, no `text` — exercises optional-field paths) |
| fixture-005-out-of-scope | I50.9 | ICD-10 | — | Heart failure (`clinicalStatus = resolved` — exercises D4 phrasing) |
| fixture-006-icd9-out-of-scope | 250.00 | ICD-9 | — | Drops because we restrict to ICD-10-CM |
| fixture-007-ckm-cad | I25.10 | ICD-10 | CKM | Atherosclerotic heart disease |
| fixture-008-ckm-ckd3a | N18.31 | ICD-10 | CKM | CKD stage 3a |
| fixture-009-eckm-dyslipidemia | E78.5 | ICD-10 | eCKM | Hyperlipidemia |
| fixture-010-eckm-obesity | E66.9 | ICD-10 | eCKM | Obesity |

Six fixtures land in the ACCESS scope (3 CKM + 3 eCKM); four are dropped by `filter_in_scope`.

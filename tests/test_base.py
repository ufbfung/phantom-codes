"""Smoke tests for the model ABC."""

from typing import Any

from phantom_codes.models.base import ConceptNormalizer, Prediction


class _DummyModel(ConceptNormalizer):
    name = "dummy"

    def predict(
        self,
        *,
        input_fhir: dict[str, Any] | None = None,
        input_text: str | None = None,
        top_k: int = 5,
    ) -> list[Prediction]:
        return [
            Prediction(
                system="http://hl7.org/fhir/sid/icd-10-cm",
                code="E11.9",
                display="Type 2 diabetes mellitus without complications",
                score=0.99,
            )
        ][:top_k]


def test_dummy_model_predict() -> None:
    model = _DummyModel()
    preds = model.predict(input_text="diabetes")
    assert len(preds) == 1
    assert preds[0].code == "E11.9"


def test_predict_batch_default_implementation() -> None:
    model = _DummyModel()
    inputs: list[tuple[dict[str, Any] | None, str | None]] = [
        (None, "a"),
        (None, "b"),
    ]
    batch_results = model.predict_batch(inputs, top_k=3)
    assert len(batch_results) == 2
    assert all(len(r) == 1 for r in batch_results)

from merchant_ai.models import ChatContext
from merchant_ai.services.clarification import ClarificationResolutionService


def test_skill_confirmation_uses_selected_option_as_structured_decision() -> None:
    service = ClarificationResolutionService()
    context = ChatContext(
        pending_clarification_stage="DEEP_ANALYSIS",
        pending_clarification_type="skill_confirm",
        pending_question="original question",
        pending_clarification_options=["accept label", "decline label"],
    )

    accepted = service.resolve_context(context.model_copy(deep=True), "accept label")
    declined = service.resolve_context(context.model_copy(deep=True), "2")

    assert accepted["confirmationDecision"] == "accepted"
    assert accepted["selectedOptionIndex"] == 0
    assert declined["confirmationDecision"] == "declined"
    assert declined["selectedOptionIndex"] == 1


def test_skill_confirmation_rejects_unstructured_free_text() -> None:
    service = ClarificationResolutionService()
    context = ChatContext(
        pending_clarification_stage="DEEP_ANALYSIS",
        pending_clarification_type="skill_confirm",
        pending_question="original question",
        pending_clarification_options=["accept label", "decline label"],
    )

    assert service.resolve_context(context, "ambiguous response") == {}

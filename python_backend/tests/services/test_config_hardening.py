from merchant_ai.config import Settings


def test_unconfigured_merchant_scope_is_fail_closed() -> None:
    settings = Settings(_env_file=None, merchant_id="", allowed_merchant_ids="")

    assert settings.allowed_merchants == set()
    assert settings.merchant_allowed("") is False
    assert settings.merchant_allowed("unconfigured-merchant") is False


def test_configured_merchant_scope_does_not_depend_on_a_code_default() -> None:
    settings = Settings(
        _env_file=None,
        merchant_id="configured-merchant",
        allowed_merchant_ids="",
    )

    assert settings.allowed_merchants == {"configured-merchant"}
    assert settings.merchant_allowed("configured-merchant") is True

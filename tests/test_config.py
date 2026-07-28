from capsule.config import Settings


def test_blank_optional_secrets_are_unset() -> None:
    settings = Settings(ark_api_key="", milvus_token="")

    assert settings.ark_api_key is None
    assert settings.milvus_token is None

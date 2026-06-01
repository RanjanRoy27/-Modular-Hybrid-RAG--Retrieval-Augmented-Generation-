from domains.accounting.preprocessor import AccountingDomain
from domains.base_domain import BaseDomain


def test_accounting_domain_implements_base_interface():
    domain = AccountingDomain()

    assert isinstance(domain, BaseDomain)
    assert domain.get_config()["domain"] == "accounting"
    assert domain.preprocess_document("  text  ") == "text"
    assert "accounting" in domain.get_system_prompt().lower()

from api.main import get_domain
from domains.accounting.preprocessor import AccountingDomain
from domains.bookkeeping.preprocessor import BookkeepingDomain
from domains.legal.preprocessor import LegalDomain


def test_domain_registry_routes_supported_domains():
    assert isinstance(get_domain("accounting"), AccountingDomain)
    assert isinstance(get_domain("legal"), LegalDomain)
    assert isinstance(get_domain("bookkeeping"), BookkeepingDomain)

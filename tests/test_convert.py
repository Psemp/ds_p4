from scripts import convert

# 5% confidence interval to handle decimals, source in sources.txt


def test_convert_kbtusf_to_kw_sqm():
    result = convert.kbtusf_to_kw_sqm(10)
    assert result < 31.54591 * 1.05 and result > 31.54591 * 0.95


def test_convert_ktbu_to_kwh():
    result = convert.ktbu_to_kwh(10)
    assert result < 2.9307107 * 1.05 and result > 2.9307107 * 0.95


def test_convert_sqft_to_sqm():
    result = convert.sqft_to_sqm(10)
    assert result < 0.9290304 * 1.05 and result > 0.9290304 * 0.95

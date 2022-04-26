
def ktbu_to_kwh(kbtu_data):
    """
    - returns a conversion of kbtu_data to kWh
    - kbtu_data must be a number
    """

    one_kwh = 0.2930711
    return kbtu_data * one_kwh


def sqft_to_sqm(sqft_data):
    """
    - returns a conversion of sqft data to metric
    - sqft_data must be a number
    """

    one_sqft = 0.09290304
    return sqft_data * one_sqft


def kbtusf_to_kw_sqm(kbtu_by_sf_data):
    """
    - returns kbtu_by_sf (British Thermal Unit by square foot) to kWh/m2
    - kbtu_by_sf_data must be a number
    """

    kwh_by_ms = 3.2
    return kbtu_by_sf_data * kwh_by_ms


def convert_to_holy_metric(data, col_name: str, hint: str = None):
    """
    - Returns data converted from imperial to metric. Tries to deduct base unit in col_name,
    force correct unit with the hint parameter.
    - Raises an error if no base unit is found or if data is not a number.

    Args:
    - data : raw data needed to convert (must be a number)
    - col_name : the name of the column where the data is from
    - hint : optionnal, default = None. Use to declare explicitly the original unit

    Returns:
    - data converted to metric/international
    """

    try:
        float(data)
    except ValueError:
        raise (ValueError, "data must be a number")

    base_unit = ""

    col_name = col_name.lower

    if hint is not None:
        base_unit = hint
    if hint == "skip":
        base_unit = "skip"
    elif "gfa" in col_name:
        base_unit = "GFA"
    elif "kbtu/sf" in col_name:
        base_unit = "kBtu/sf"
    elif "kbtu" in col_name and "sf" not in col_name:
        base_unit = "kBtu"
    elif "sqft" in col_name or "ft2" in col_name:
        base_unit = "square_foot"
    else:
        pass

    match base_unit:
        case "gfa" | "GFA":
            return sqft_to_sqm(data)
        case "kBtu" | "kbtu":
            return ktbu_to_kwh(data)
        case "kBtu/sf" | "kbtu/sf":
            return kbtusf_to_kw_sqm(data)
        case "square_foot" | "sqft":
            return sqft_to_sqm(data)
        case "skip":
            pass
        case _:
            raise Exception("Base unit not found, use 'hint' parameter to precise it")

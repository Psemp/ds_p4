from pandas import isna
from numpy import nan


def ktbu_to_kwh(kbtu_data):
    """
    - returns a conversion of kbtu_data to kWh
    - kbtu_data must be a number
    """

    one_kbtu_in_therm = 0.2930711
    return kbtu_data * one_kbtu_in_therm


def therms_to_kwh(therm_data):
    """
    - returns a conversion of therm_data to kWh
    - kbtu_data must be a number
    """

    one_therm_in_kwh = 29.307107
    return therm_data * one_therm_in_kwh


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

    kwh_by_ms = 3.15459
    return kbtu_by_sf_data * kwh_by_ms


def kg_cdioxyde_by_sqm(tons_co_two, surface):
    """
    - Divides total CO2 by surface of the place

    Args:
    - tons_co_two : implicit, must be numeric (and metric)
    - surface : surface of an area in square meters

    Returns:
    - kilograms of co2 per square meter
    """

    try:
        float(tons_co_two) and float(surface)
    except ValueError:
        raise (ValueError, "parameters must be numeric")

    kg_co_two = tons_co_two * 1000
    return kg_co_two / surface


def find_unit(var_name: str, hint: str = None, convert: bool = False):
    """
    - Tries to deduct base unit in var_name, returns it - if convert is True, also returns
    the converted international unit as a dict w/ key base_unit & converted_unit

    Args :
    - var_name : variable name, as a string
    - hint : optionnal, default = None. Use to declare explicitly the original unit
    - convert : boolean, default = False, if True, converts the unit and renders it as a key pair in dict

    Returns:
    - The base unit of the variable, if deducted by the function, as a string if convert = False,
    as a two key-value pairs as base_unit: base_unit, converted_unit: converted_unit
    """

    base_unit = None
    var_name = var_name.lower()
    if hint is not None:  # hint = skip if needs to ignore
        base_unit = hint
    elif "gfa" in var_name:
        base_unit = "GFA"
    elif "kbtu/sf" in var_name:
        base_unit = "kBtu/sf"
    elif "kbtu" in var_name and "sf" not in var_name:
        base_unit = "kBtu"
    elif "sqft" in var_name or "ft2" in var_name:
        base_unit = "square_foot"
    elif "therms" in var_name or "therm" in var_name:
        base_unit = "therms"
    else:
        pass

    if not convert:
        return base_unit

    converted_unit = None

    match base_unit:
        case "gfa" | "GFA":
            converted_unit = "Square Metre"
        case "kBtu" | "kbtu":
            converted_unit = "kWh"
        case "kBtu/sf" | "kbtu/sf":
            converted_unit = "kWh/m2"
        case "square_foot" | "sqft":
            converted_unit = "Square Metre"
        case "therms":
            converted_unit = "kWh"
        case "skip":
            pass
        case _:
            raise Exception("Base unit not found, use 'hint' parameter to precise it")

    if convert:
        return {
            "base_unit": base_unit,
            "converted_unit": converted_unit
        }


def convert_to_holy_metric(data, var_name: str, hint: str = None, errors: str = "ignore"):
    """
    - Returns data converted from imperial to metric. Force correct unit with the hint parameter.
    - Returns np.nan if data is not a number.

    Args:
    - data : raw data needed to convert (must be a number), if None or Missing Value (pd.na, not np.nan) -> Skip
    - var_name : the name of the column where the data is from
    - hint : optionnal, default = None. Use to declare explicitly the original unit

    Returns:
    - Data converted to metric/international
    """

    try:
        data = float(data)
    except (ValueError, TypeError) as error:
        if (data is None or isna(data)):
            base_unit = "skip"
        elif errors == "ignore":
            return nan
        else:
            raise error

    var_name = var_name.lower()

    base_unit = find_unit(var_name=var_name, hint=hint)

    match base_unit:
        case "gfa" | "GFA":
            return sqft_to_sqm(data)
        case "kBtu" | "kbtu":
            return ktbu_to_kwh(data)
        case "kBtu/sf" | "kbtu/sf":
            return kbtusf_to_kw_sqm(data)
        case "square_foot" | "sqft":
            return sqft_to_sqm(data)
        case "therms":
            therms_to_kwh(data)
        case "skip":
            pass
        case _:
            raise Exception("Base unit not found, use 'hint' parameter to precise it")

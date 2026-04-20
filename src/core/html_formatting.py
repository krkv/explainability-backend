"""Helpers for consistent numeric formatting in HTML responses."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

HTML_FLOAT_PRECISION = 2


def _quantize_float(value: float, precision: int) -> str:
    quantizer = Decimal("1").scaleb(-precision)
    return str(Decimal(str(value)).quantize(quantizer, rounding=ROUND_HALF_UP))


def format_html_number(value: Any, precision: int = HTML_FLOAT_PRECISION) -> str:
    """Format numeric values for HTML output with a fixed float precision."""
    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return _quantize_float(numeric_value, precision)
        return str(value)

    return str(value)


def dataframe_to_html(frame: pd.DataFrame, *, index: bool = True, precision: int = HTML_FLOAT_PRECISION) -> str:
    """Render a dataframe to HTML using a shared float formatter."""
    return frame.to_html(
        index=index,
        float_format=lambda value: format_html_number(value, precision),
    )


def tabulate_html(table: Any, headers: Any, *, precision: int = HTML_FLOAT_PRECISION, **kwargs: Any) -> str:
    """Render a tabulate HTML table using a shared float formatter."""
    formatted_table = []
    for row in table:
        formatted_row = []
        for value in row:
            if isinstance(value, (float, np.floating)) and not isinstance(value, bool):
                formatted_row.append(format_html_number(value, precision))
            else:
                formatted_row.append(value)
        formatted_table.append(formatted_row)

    return tabulate(
        formatted_table,
        headers,
        tablefmt="html",
        **kwargs,
    )

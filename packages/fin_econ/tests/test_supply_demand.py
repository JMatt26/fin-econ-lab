"""Tests for fin_econ.micro.supply_demand."""

import pytest

from fin_econ.micro.supply_demand import (
    price_elasticity_demand,
    price_elasticity_demand_linear,
)


def test_price_elasticity_demand_linear():
    """Linear demand Q = 10 - 2*P: at P=2, Q=6, slope=-2 -> elasticity = (2)(2)/6 = 2/3."""
    demand = lambda p: 10 - 2 * p
    price = 2.0
    slope = -2.0
    got = price_elasticity_demand_linear(demand, price, slope)
    expected = (2.0 * 2.0) / 6.0  # -slope * price / quantity
    assert got == pytest.approx(expected)


def test_price_elasticity_demand():
    """Same linear demand; with derivative dQ/dP = -2, elasticity = 2*P/Q = 2/3 at P=2."""
    demand = lambda p: 10 - 2 * p
    derivative = lambda p: -2.0
    price = 2.0
    got = price_elasticity_demand(demand, price, derivative_function=derivative)
    expected = (2.0 * 2.0) / 6.0
    assert got == pytest.approx(expected)

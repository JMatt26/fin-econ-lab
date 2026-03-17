"""Tests for fin_econ.micro.supply_demand."""

import math
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

def test_price_elasticity_demand_central_difference():
    """Same linear demand; with derivative dQ/dP = -2, elasticity = 2*P/Q = 2/3 at P=2."""
    demand = lambda p: 10 - 2 * p
    derivative = lambda p: -2.0
    price = 2.0
    got = price_elasticity_demand(demand, price, h=1e-5)
    expected = price_elasticity_demand(demand, price, derivative_function=derivative)
    assert got == pytest.approx(expected)

def test_price_elasticity_demand_quadratic():
    demand = lambda p: 200 - 3*p - 0.5*p**2
    price = 5.0
    derivative = lambda p: -3 - p
    got = price_elasticity_demand(demand, price, derivative_function=derivative)
    expected = 8.0 * 5.0 / 172.5
    assert got == pytest.approx(expected)

def test_price_elasticity_demand_quadratic_central_difference():
    demand = lambda p: 200 - 3*p - 0.5*p**2
    price = 5.0
    derivative = lambda p: -3 - p
    got = price_elasticity_demand(demand, price, h=1e-5)
    expected = price_elasticity_demand(demand, price, derivative_function=derivative)
    assert got == pytest.approx(expected)

def test_price_elasticity_demand_cubic():
    demand = lambda p: 500 - 4*p - 0.5*p**2 - 0.01*p**3
    price = 5.0
    derivative = lambda p: -4 - p - 0.03*p**2
    got = price_elasticity_demand(demand, price, derivative_function=derivative)
    expected = -(-4 - 5 - 0.03*5**2) * 5 / (500 - 4*5 - 0.5*5**2 - 0.01*5**3)
    assert got == pytest.approx(expected)

def test_price_elasticity_demand_quartic():
    demand = lambda p: 800 - 5*p - 0.3 * p**2 - 0.02*p**3 - 0.0001*p**4
    price = 5.0
    derivative = lambda p: -5 - 0.6*p - 0.06*p**2 - 0.0004*p**3
    got = price_elasticity_demand(demand, price, derivative_function=derivative)
    expected = -(-5 - 0.6*5 - 0.06*5**2 - 0.0004*5**3) * 5 / (800 - 5*5 - 0.3*5**2 - 0.02*5**3 - 0.0001*5**4)
    assert got == pytest.approx(expected)

def test_price_elasticity_demand_exponential():
    demand = lambda p: 100 * math.exp(-0.1*p)
    price = 5.0
    derivative = lambda p: -10 * math.exp(-0.1*p)
    got = price_elasticity_demand(demand, price, derivative_function=derivative)
    expected = -(-10 * math.exp(-0.1*5)) * 5 / (100 * math.exp(-0.1*5))
    assert got == pytest.approx(expected)

def test_price_elasticity_demand_exponential_central_difference():
    demand = lambda p: 100 * math.exp(-0.1*p)
    price = 5.0
    derivative = lambda p: -10 * math.exp(-0.1*p)
    got = price_elasticity_demand(demand, price, h=1e-5)
    expected = price_elasticity_demand(demand, price, derivative_function=derivative)
    assert got == pytest.approx(expected)

def test_price_elasticity_demand_logarithmic():
    demand = lambda p: 200 - 40*math.log(p)
    price = 5.0
    derivative = lambda p: -40/p
    got = price_elasticity_demand(demand, price, derivative_function=derivative)
    expected = -(-40/5) * 5 / (200 - 40*math.log(5))
    assert got == pytest.approx(expected)

def test_price_elasticity_demand_logarithmic_central_difference():
    demand = lambda p: 200 - 40*math.log(p)
    price = 5.0
    derivative = lambda p: -40/p
    got = price_elasticity_demand(demand, price, h=1e-5)
    expected = price_elasticity_demand(demand, price, derivative_function=derivative)
    assert got == pytest.approx(expected)

def test_price_elasticity_demand_constant():
    demand = lambda p: 1000*p**(-1.5)
    price = 5.0
    derivative = lambda p: -1500*p**(-2.5)
    got = price_elasticity_demand(demand, price, derivative_function=derivative)
    new_price = 10.0
    got_new = price_elasticity_demand(demand, new_price, derivative_function=derivative)
    assert got == pytest.approx(got_new)

def test_price_elasticity_demand_constant_central_difference():
    demand = lambda p: 1000*p**(-1.5)
    price = 5.0
    got = price_elasticity_demand(demand, price, h=1e-5)
    new_price = 10.0
    got_new = price_elasticity_demand(demand, new_price, h=1e-5)
    assert got == pytest.approx(got_new)
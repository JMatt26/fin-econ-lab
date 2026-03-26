"""
Supply and Demand helpers.

This module provides well-tested utilities for: 
- 
-
-
"""

from typing import Callable, Optional
import math

Number = float
CurveFunction = Callable[[Number], Number]

class EconValueError(ValueError):
    """Raised when an inputnis inavlid for an economics calculation."""

def _validate_finite(name: str, value: Number) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise EconValueError(f"{name} must be finite, got {value}")

def _central_difference(
    fn: CurveFunction,
    x: Number,
    h: Number,
) -> Number:
    return (fn(x + h) - fn(x - h)) / (2.0 * h)

def price_elasticity_demand_linear(
    demand_function: CurveFunction,
    price: Number,
    slope: Number,
) -> Number:
    """
    Compute the price elasticity of demand for a linear demand function.

    Parameters:
    - demand_function: The demand function written as Q(P), where input is price and output is quantity demanded.
    - price: The price at which to compute the elasticity.
    - slope: The slope of the demand function.
    """
    _validate_finite("price", price)
    _validate_finite("slope", slope)

    quantity = demand_function(price)
    _validate_finite("quantity", quantity)

    if quantity == 0:
        raise ZeroDivisionError("Elasticity is undefined for zero quantity.")

    elasticity = -slope * price / quantity
    return elasticity


def price_elasticity_demand(
    demand_function: CurveFunction,
    price: Number,
    derivative_function: Optional[CurveFunction] = None,
    h: Number = 1e-5,
) -> Number:
    """
    Compute the price elasticity of demand

    Parameters:
    - demand_function: The demand function written as Q(P), where input is price and output is quantity demanded.
    - price: The price at which to compute the elasticity.
    - derivative_function: The derivative of the demand function. If not provided, it will be computed numerically using central difference.
    - h: The step size for the numerical derivative.

    Returns:
    - The price elasticity of demand at the given price.
    """

    _validate_finite("price", price)
    _validate_finite("h", h)

    if h <= 0:
        raise EconValueError(f"Step size h must be positive, got {h}")

    quantity = demand_function(price)
    _validate_finite("quantity", quantity)

    if quantity == 0:
        raise ZeroDivisionError("Elasticity is undefined for zero quantity.")

    if derivative_function is not None:
        derivative = derivative_function(price)
        _validate_finite("derivative", derivative) # if not finite, report infinitely elastic
    else:
        derivative = _central_difference(demand_function, price, h) # if 0, report completely inelastic demand
        _validate_finite("derivative", derivative)

    elasticity = -derivative * price / quantity
    return elasticity


def price_elasticity_supply_linear(
    supply_function: CurveFunction,
    price: Number,
    slope: Number,
) -> Number:
    """
    Compute the price elasticity of supply for a linear supply function.
    """
    _validate_finite("price", price)
    _validate_finite("slope", slope)

    quantity = supply_function(price)
    _validate_finite("quantity", quantity)

    if quantity == 0:
        raise ZeroDivisionError("Elasticity is undefined for zero quantity.")

    elasticity = slope * price / quantity
    return elasticity


def price_elasticity_supply(
    supply_function: CurveFunction,
    price: Number,
    derivative_function: Optional[CurveFunction] = None,
    h: Number = 1e-5,
) -> Number:
    """
    Compute the price elasticity of supply
    """
    _validate_finite("price", price)
    _validate_finite("h", h)

    if h <= 0:
        raise EconValueError(f"Step size h must be positive, got {h}")

    quantity = supply_function(price)
    _validate_finite("quantity", quantity)

    if quantity == 0:
        raise ZeroDivisionError("Elasticity is undefined for zero quantity.")

    if derivative_function is not None:
        derivative = derivative_function(price)
        _validate_finite("derivative", derivative)
    else:
        derivative = _central_difference(supply_function, price, h)
        _validate_finite("derivative", derivative)

    elasticity = derivative * price / quantity
    return elasticity

def arc_elasticity(deltaQ, deltaP, averageQ, averageP):
    """
    Compute the arc elasticity of demand or supply
    """
    return -deltaQ / deltaP * averageP / averageQ
#!/usr/bin/env python3
"""cptest: does a coincident peak allocation still hold when a large load is added?

The test compares two ways of dividing a demand-related cost pool between the existing customers and a new load:

  the existing method   the coincident peak method in use (1CP, 4CP, 12CP, or the peaks of named months), which counts
                        every customer group at its average load in the hours that the method names;
  the benchmark         the allocation by hourly loss-of-load risk, which counts every group at its load weighted by
                        the share of each hour in the yearly risk.

The overcharge of the existing customers is  eps = o_a / o_B - 1, where o_a is their share of the pool under the existing
method and o_B their share under the benchmark. The existing method passes if eps is not above a tolerance tau (default 5%).

Two benchmarks are provided, one for each kind of data that a user may hold.

  load   (hourly load only)    The risk of an hour falls exponentially with the margin between capacity and load
                               (Garver's approximation, applied hour by hour). One number is needed, the window m:
                               the megawatts of margin that lower the risk by the factor e. By default it is the
                               window at which the benchmark agrees with the method on today's system, not below a
                               floor, and it grows with the system as m = mu * P * (1 + x) ** gamma.
  fleet  (hourly load and the   The risk of an hour is the probability that the available capacity of the generating
          list of units)        units is below the load, from a capacity outage probability table with the units on
                               planned maintenance removed week by week (the loss-of-load probability method).

For a stated size of new load the tool reports the overcharge, the largest load that the method carries (the failing
size), the smallest share of the pool that has to be allocated on energy, and the megawatts that have to be curtailable
if the method is to stay, with the hours and the energy of curtailment.

If the load file holds the hourly load of the customer classes, the same is reported class by class. The overcharge of
class k is  1 + eps_k = (1 + eps) v_k, where v_k, the class factor, is the part of the class in the demand of the
existing customers under the method divided by its part in their benchmark demand. The verdict is then read for the
class that the new load moves up most, and the remedies are sized on that class.

With both kinds of data nothing has to be assumed. The mode fleet reports, for every size, the window at which the
load-based benchmark gives the same benchmark demand and the exponent with which that window has grown from the window
of today. It also runs the load-based benchmark with its own premises on the same load and reports the floor of the
tolerance: the largest relative difference between the shares that the versions of the benchmark give the existing
customers. A tolerance below that floor would make the verdict depend on the version of the benchmark.

Where a denominator of the theory can vanish, the tool says in words what the case means. The overcharge that a flat
load can cause has a ceiling, kappa_a / lambda - 1. If the tolerance is at or above it, the existing method passes at
any size: there is then no safe size and no failing size to report, and no remedy is needed. README.md lists the cases.

Usage (see README.md for the file formats):

  python cptest.py selftest
  python cptest.py rts-data OUT_DIR
  python cptest.py load  --load load.csv --method 12CP --size 500 1000
  python cptest.py fleet --load load.csv --units units.csv --method 12CP --size 500 1000
  python cptest.py load  --load load.csv --class-columns res_mw com_mw ind_mw --size 500

Requires Python 3.9 or later and numpy. No other package is used.
"""
import argparse
import csv
import datetime as _dt
import json
import math
import os
import sys

import numpy as np

__version__ = "1.3"
HOURS_PER_WEEK = 168
WEEKS = 52
DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


# ======================================================================================================================
# 1. Small helpers
# ======================================================================================================================
def _median(values):
    return float(np.median(np.asarray(values, float)))


def _bisect(f, lo, hi, n=200):
    """Root of a function that rises from negative to positive on [lo, hi]."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def month_of_hour(n_hours):
    """Calendar month (1 to 12) of each hour of a year that starts on 1 January and has no 29 February.
    Hours beyond 8,760 are counted with December; a year of 8,736 hours ends on 30 December."""
    days = np.repeat(np.arange(1, 13), DAYS_IN_MONTH)
    month = np.repeat(days, 24)
    if n_hours <= len(month):
        return month[:n_hours].copy()
    return np.concatenate([month, np.full(n_hours - len(month), 12)])


def week_of_hour(n_hours):
    """Week (0 to 51) of each hour: blocks of 168 hours from the first hour, the last block takes the hours left over."""
    return np.minimum(np.arange(n_hours) // HOURS_PER_WEEK, WEEKS - 1)


# ======================================================================================================================
# 2. The method in use
# ======================================================================================================================
def method_hours(system_load, month, method):
    """Indices of the hours that a coincident peak method names, found on the hourly load of the whole system.

    method   '1CP'              the hour of the annual system peak
             'kCP', k = 2..12   the hours of the k highest monthly system peaks ('12CP': all twelve)
             'months:6,7,8,12'  the hours of the system peak in the named months
    """
    system_load = np.asarray(system_load, float)
    month = np.asarray(month)
    peak_hour = {}
    for mo in range(1, 13):
        idx = np.flatnonzero(month == mo)
        if len(idx):
            peak_hour[mo] = int(idx[np.argmax(system_load[idx])])
    name = method.strip().upper()
    if name == "1CP":
        return [int(np.argmax(system_load))]
    if name.startswith("MONTHS:"):
        return [peak_hour[int(tok)] for tok in name[7:].split(",")]
    if name.endswith("CP") and name[:-2].isdigit():
        k = int(name[:-2])
        if not 2 <= k <= 12:
            raise ValueError("kCP needs k between 1 and 12")
        return sorted(peak_hour.values(), key=lambda i: -system_load[i])[:k]
    raise ValueError(f"unknown method {method!r}: use 1CP, 4CP, 12CP, kCP or months:6,7,8,12")


# ======================================================================================================================
# 3. The load-based benchmark (hourly load only)
# ======================================================================================================================
def exp_weights(profile, window):
    """Hourly weights of the load-based benchmark: w(t) proportional to exp(-(peak - profile(t)) / window).
    They are positive and add up to one. `profile` is the load that the system has to cover in each hour (MW). A window
    of zero is read as its limit: the whole weight on the hour of the peak."""
    profile = np.asarray(profile, float)
    if window <= 0:      # the limit of a vanishing window: the whole weight on the hour, or the hours, of the peak
        w = (profile >= profile.max() - 1e-9 * max(1.0, abs(float(profile.max())))).astype(float)
        return w / w.sum()
    w = np.exp(-(profile.max() - profile) / window)
    return w / w.sum()


def benchmark_demand(load, weights):
    """Benchmark demand of a customer group: its hourly load weighted by the hourly shares of the risk (MW)."""
    return float(np.dot(weights, np.asarray(load, float)))


def equivalent_window(load, demand, profile=None):
    """The window (MW) at which the load-based benchmark counts `load` at `demand`. Zero if the demand is the load at
    the peak hour of the profile or more; infinite if it is the average load or less."""
    load = np.asarray(load, float)
    profile = load if profile is None else np.asarray(profile, float)
    peak = float(profile.max())
    if demand >= float(load[np.argmax(profile)]) - 1e-9:
        return 0.0
    if demand <= float(load.mean()) + 1e-9:
        return float("inf")
    return _bisect(lambda m: demand - benchmark_demand(load, exp_weights(profile, m)), 1e-6 * peak, 1e3 * peak)


def overcharge_ceiling(kappa_a, load_factor):
    """The largest overcharge that a flat load of any size can cause under any benchmark that counts the existing
    customers at no less than their average load: kappa_a / lambda - 1. The bound of the overcharge for a flat load of
    relative size x is this ceiling times x / (kappa_a + x), which rises with x and stays below the ceiling."""
    return kappa_a / load_factor - 1.0


def safe_size(kappa_a, load_factor, tau):
    """The size of flat load, as a share of the existing peak, below which the existing method passes under any
    benchmark that counts the existing customers at no less than their average load:
    x_s = tau kappa_a / (ceiling - tau) = tau kappa_a lambda / (kappa_a - (1 + tau) lambda).
    The denominator is the distance of the tolerance from the ceiling of the overcharge. If the tolerance is at or
    above the ceiling the bound never reaches it, no size fails, and the safe size is infinite."""
    den = kappa_a - (1.0 + tau) * load_factor
    return float("inf") if den <= 0 else tau * kappa_a * load_factor / den


def min_energy_share(o_a, o_b, o_e, tau):
    """Smallest share f of the pool that has to be allocated on energy so that the existing customers' blended share
    (1 - f) o_a + f o_E is not above (1 + tau) o_B. Zero if the existing method passes.

    The expression (o_a - (1 + tau) o_B) / (o_a - o_E) is used only when the method fails. Its numerator is then
    positive, and if an allocation on energy passes, o_E <= (1 + tau) o_B, the denominator is at least as large as the
    numerator: it cannot vanish, and the share lies between zero and one. That is the case under every benchmark that
    counts the existing customers at no less than their average load. Otherwise no share suffices, and the function
    returns infinity."""
    level = (1.0 + tau) * o_b
    if o_a <= level:
        return 0.0
    if o_e > level:
        return float("inf")
    return (o_a - level) / (o_a - o_e)


def tolerance_floor(shares):
    """The floor of the tolerance: the largest relative difference between the shares that several versions of the
    benchmark give the existing customers, max / min - 1. An allocation that reproduced one version exactly would show
    this overcharge when it is tested against another, so a smaller tolerance would turn on the version chosen."""
    shares = [float(v) for v in shares]
    if not shares or min(shares) <= 0:
        raise ValueError("the floor of the tolerance needs positive shares")
    return max(shares) / min(shares) - 1.0


def curtailment(load, curtailable_mw, headroom_mw=0.0):
    """Hours and energy of curtailment under the rule that the curtailable megawatts are served only from the room that
    the firm load leaves below the firm peak (plus an optional headroom). In hour t the cut is
    max(0, X_c - (P - L(t)) - headroom). Returns hours with a cut, energy cut (MWh), the same as hours at full size, and
    as a share of the energy that the curtailable megawatts would draw in the year."""
    load = np.asarray(load, float)
    if curtailable_mw <= 0:
        return {"hours": 0, "energy_mwh": 0.0, "full_size_hours": 0.0, "energy_share": 0.0}
    cut = np.maximum(0.0, curtailable_mw - (load.max() - load) - headroom_mw)
    energy = float(cut.sum())
    return {"hours": int((cut > 0).sum()), "energy_mwh": energy, "full_size_hours": energy / curtailable_mw,
            "energy_share": energy / (len(load) * curtailable_mw)}


# ----------------------------------------------------------------------------------------------------------------------
# A year of data, and one evaluation of method and benchmark for that year
# ----------------------------------------------------------------------------------------------------------------------
def make_year(load, month=None, label="", variable_generation=None, new_load=None, classes=None):
    """A year of hourly data. load: MW of the existing customers. month: 1..12 for each hour (default: a calendar year
    from 1 January). variable_generation: optional MW of wind, solar and other must-take output, which is taken off the
    load before the risk of an hour is formed. new_load: optional hourly profile of a new load that is not flat.
    classes: optional dictionary with the hourly MW of every class of the existing customers; they add up to load."""
    load = np.asarray(load, float)
    if load.ndim != 1 or len(load) < 24 or not np.isfinite(load).all() or load.min() < 0:
        raise ValueError("the hourly load has to be a series of finite values that are not negative")
    year = {"L": load, "month": month_of_hour(len(load)) if month is None else np.asarray(month).astype(int), "label": str(label),
            "P": float(load.max()), "A": float(load.mean())}
    for key, series in (("G", variable_generation), ("N", new_load)):
        if series is not None:
            series = np.asarray(series, float)
            if series.shape != load.shape or not np.isfinite(series).all():
                raise ValueError("every hourly series has to have the length of the load and finite values")
            year[key] = series
    if len(year["month"]) != len(load):
        raise ValueError("the month labels have to have the length of the load")
    if classes:
        year["K"] = {}
        for name, series in classes.items():
            series = np.asarray(series, float)
            if series.shape != load.shape or not np.isfinite(series).all():
                raise ValueError(f"class {name!r}: the hourly series has to have the length of the load and finite values")
            year["K"][str(name)] = series
        if float(np.abs(sum(year["K"].values()) - load).max()) > 1e-6 * max(1.0, year["P"]):
            raise ValueError("the classes do not add up to the load of the existing customers")
    return year


def _new_load(year, size_mw, scale):
    """Hourly megawatts of the new load: flat at size_mw, plus the profile of the year times scale."""
    n = np.full(len(year["L"]), float(size_mw))
    if scale and "N" in year:
        n = n + scale * year["N"]
    return n


def _shares(year, method, new, weights_fn):
    """Demands and shares of the existing customers and of the new load under the method and under a benchmark whose
    hourly weights weights_fn(combined load net of variable generation) returns."""
    L = year["L"]
    total = L + new
    hours = method_hours(total, year["month"], method)
    la, na = float(L[hours].mean()), float(new[hours].mean())
    w = weights_fn(total - year["G"] if "G" in year else total)
    lb, nb = benchmark_demand(L, w), benchmark_demand(new, w)
    o_a = la / (la + na) if la + na > 0 else 1.0
    o_b = lb / (lb + nb) if lb + nb > 0 else 1.0
    energy_new = float(new.mean())
    return {"la": la, "na": na, "lb": lb, "nb": nb, "ne": energy_new, "o_a": o_a, "o_b": o_b, "o_e": year["A"] / (year["A"] + energy_new),
            "eps": o_a / o_b - 1.0, "growth": float(total.max()) / year["P"], "weights": w, "hours": hours}


def load_based(year, method, mu, gamma=1.0, size_mw=0.0, scale=0.0):
    """One year under the load-based benchmark. mu: window of today as a share of the existing peak. gamma: exponent
    with which the window follows the growth of the system peak (1: in proportion; 0.5: as the square root)."""
    new = _new_load(year, size_mw, scale)
    growth = float((year["L"] + new).max()) / year["P"]
    window = mu * year["P"] * growth ** gamma
    out = _shares(year, method, new, lambda profile: exp_weights(profile, window))
    out["window_mw"] = window
    return out


def implied_window_share(years, method):
    """The window, as a share of the existing peak, at which the load-based benchmark agrees with the method on the
    system of today: the median over the years of l_a / l* - 1 is zero. Zero for 1CP. Raises an error if the method
    counts the existing customers below their average load, because no window does that."""
    def gap(mu):
        vals = []
        for y in years:
            s = load_based(y, method, mu)
            vals.append(s["la"] / s["lb"] - 1.0)
        return _median(vals)
    if gap(1e-6) >= -1e-12:
        return 0.0
    if gap(1e3) <= 0:
        raise ValueError("the method counts the existing customers at or below their average load: no window is implied")
    return _bisect(gap, 1e-6, 1e3)


def _reference_peak(years):
    return _median([y["P"] for y in years])


def overcharge_load_based(years, method, mu, size_mw, gamma=1.0, scale=0.0):
    """Median over the years of the overcharge of the existing customers under the load-based benchmark."""
    return _median([load_based(y, method, mu, gamma, size_mw, scale)["eps"] for y in years])


def failing_size_load_based(years, method, mu, tau=0.05, gamma=1.0, upper=1e4):
    """The flat load, as a share of the median existing peak, at which the median overcharge equals the tolerance.
    Infinite if the overcharge stays below the tolerance at any size."""
    ref = _reference_peak(years)
    f = lambda x: overcharge_load_based(years, method, mu, x * ref, gamma) - tau
    return float("inf") if f(upper) < 0 else _bisect(f, 0.0, upper)


# ----------------------------------------------------------------------------------------------------------------------
# Customer classes
# ----------------------------------------------------------------------------------------------------------------------
NEW_LOAD = "new load"


def class_rows(year, case, today, f=0.0):
    """One year, class by class. case: the evaluation of method and benchmark with the new load (the result of
    load_based or fleet_based); today: the evaluation without it; f: share of the pool that is allocated on energy.

    For every class of the existing customers:
      share_method, share_benchmark, share_energy   its shares of the whole pool
      share                                          its share under the allocation, (1 - f) method + f energy
      factor, factor_today   the class factor v: its part in the demand of the existing customers under the method
                             divided by its part in their benchmark demand, with the new load and without it
      overcharge             share / share_benchmark - 1; for f = 0 this is (1 + eps) v - 1
      added                  what the new load adds: (1 + overcharge) / v_today - 1; None for a class that has no
                             load in the hours that the existing method names today (v_today = 0)
    The row NEW_LOAD has the shares and the overcharge of the new load itself."""
    h, w, h0, w0 = case["hours"], case["weights"], today["hours"], today["weights"]
    D, B, A = case["la"] + case["na"], case["lb"] + case["nb"], year["A"] + case["ne"]
    rows = {}
    for name, series in year["K"].items():
        d, b, a = float(series[h].mean()), benchmark_demand(series, w), float(series.mean())
        d0, b0 = float(series[h0].mean()), benchmark_demand(series, w0)
        if min(b, b0) <= 0:
            raise ValueError(f"class {name!r} has no load in any hour that carries risk: it cannot be tested")
        v, v0 = (d / case["la"]) / (b / case["lb"]), (d0 / today["la"]) / (b0 / today["lb"])
        share = (1.0 - f) * d / D + f * a / A
        over = share / (b / B) - 1.0
        # A class without load in the hours that the existing method names today is not counted by it: its factor of
        # today is zero, and what the new load adds cannot be formed as a ratio. It is reported and left out of the verdict.
        rows[name] = {"share_method": d / D, "share_benchmark": b / B, "share_energy": a / A, "share": share, "factor": v,
                      "factor_today": v0, "overcharge": over, "added": ((1.0 + over) / v0 - 1.0) if v0 > 0 else None}
    if case["na"] > 0 and case["nb"] > 0:
        share = (1.0 - f) * case["na"] / D + f * case["ne"] / A
        over = share / (case["nb"] / B) - 1.0
        rows[NEW_LOAD] = {"share_method": case["na"] / D, "share_benchmark": case["nb"] / B, "share_energy": case["ne"] / A, "share": share,
                          "factor": None, "factor_today": None, "overcharge": over, "added": over}
    return rows


def class_table(years, per, today, f=0.0):
    """Medians over the years of class_rows. per, today: one evaluation for each year."""
    rows = [class_rows(y, p, t, f) for y, p, t in zip(years, per, today)]
    return {k: {q: (None if any(r[k][q] is None for r in rows) else _median([r[k][q] for r in rows])) for q in rows[0][k]} for k in rows[0]}


def most_moved(table):
    """The class of the existing customers that the new load moves up most, and its added overcharge. A class that the
    existing method does not count today (added is None) is left out."""
    counted = [k for k in table if k != NEW_LOAD and table[k]["added"] is not None]
    if not counted:
        return None, 0.0
    name = max(counted, key=lambda k: table[k]["added"])
    return name, table[name]["added"]


def class_energy_share(years, per, today, tau):
    """The smallest share of the whole pool on energy at which no class that an energy share relieves is moved up by
    more than the tolerance. Returns the share (None if no share achieves it), the table at that share, and the class
    that stands highest at that share with its added overcharge (an energy share loads the classes with long hours of
    use, and one of them can then be above the tolerance)."""
    base = class_table(years, per, today)
    relieved = [k for k in base if k != NEW_LOAD and base[k]["share_method"] > base[k]["share_energy"]]
    relieved = [k for k in relieved if base[k]["added"] is not None]
    worst = lambda f: (max(class_table(years, per, today, f)[k]["added"] for k in relieved) if relieved else -1.0) - tau
    if worst(0.0) <= 0:
        f = 0.0
    elif worst(1.0) > 0:
        return None, None, None
    else:
        f = _bisect(lambda v: -worst(v), 0.0, 1.0)
    at = class_table(years, per, today, f)
    return f, at, most_moved(at)


def _class_block(years, per, today, tau):
    """The class results of one case: the table, the verdict for the class that is moved most, the energy share."""
    table = class_table(years, per, today)
    name, added = most_moved(table)
    f, at, top = class_energy_share(years, per, today, tau)
    return {"table": table, "most_moved": name, "added_max": added, "passes": bool(added <= tau), "energy_share": f,
            "table_with_energy_share": at, "highest_with_energy_share": None if top is None else {"class": top[0], "added": top[1]}}


def class_failing_size_load_based(years, method, mu, tau=0.05, gamma=1.0):
    """The flat load, as a share of the median existing peak, at which the class that is moved up most reaches the
    tolerance: the first crossing on a coarse grid of sizes, refined by bisection. Infinite if there is none."""
    ref = _reference_peak(years)
    today = [load_based(y, method, mu, gamma) for y in years]

    def f(x):
        per = [load_based(y, method, mu, gamma, x * ref) for y in years]
        return most_moved(class_table(years, per, today))[1] - tau
    lo = 0.0
    for hi in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0):
        if f(hi) > 0:
            return _bisect(f, lo, hi) if lo > 0 else _bisect(f, 1e-9, hi)
        lo = hi
    return float("inf")


def run_load_based(years, method="12CP", sizes_mw=(0.0,), tau=0.05, window="auto", floor=0.04, gammas=(1.0, 0.5),
                   scales=(0.0,), headroom=0.0):
    """The whole test from the hourly load alone. Returns a dictionary with the statistics of the system, the window,
    and for every growth exponent and every size the overcharge and the remedies (medians over the years)."""
    ref = _reference_peak(years)
    base = [load_based(y, method, 1.0) for y in years]      # the method's demand does not depend on the window
    kappa = _median([b["la"] / y["P"] for b, y in zip(base, years)])
    lam = _median([y["A"] / y["P"] for y in years])
    implied = implied_window_share(years, method)
    mu = max(floor, implied) if window == "auto" else float(window)
    res = {"benchmark": "load", "method": method, "tolerance": tau, "years": [y["label"] for y in years], "peak_mw": ref,
           "load_factor": lam, "kappa_a": kappa, "implied_window_share": implied, "window_share": mu,
           "window_rule": "implied by the existing method, not below the floor" if window == "auto" else "set by the user",
           "floor": floor, "overcharge_ceiling": overcharge_ceiling(kappa, lam), "safe_size": safe_size(kappa, lam, tau), "cases": []}
    has_classes = all("K" in y for y in years)
    res["classes"] = sorted(years[0]["K"]) if has_classes else None
    for gamma in gammas:
        fail = failing_size_load_based(years, method, mu, tau, gamma)
        if has_classes:
            today = [load_based(y, method, mu, gamma) for y in years]
            cfail = class_failing_size_load_based(years, method, mu, tau, gamma)
        for size in sizes_mw:
            for scale in scales:
                per = [load_based(y, method, mu, gamma, size, scale) for y in years]
                eps = _median([p["eps"] for p in per])
                fmin = _median([min_energy_share(p["o_a"], p["o_b"], p["o_e"], tau) for p in per])
                case = {"gamma": gamma, "size_mw": size, "scale": scale, "size_share": size / ref,
                        "kappa_star": _median([p["lb"] / y["P"] for p, y in zip(per, years)]),
                        "o_a": _median([p["o_a"] for p in per]), "o_b": _median([p["o_b"] for p in per]),
                        "o_e": _median([p["o_e"] for p in per]), "overcharge": eps, "passes": bool(eps <= tau),
                        "min_energy_share": fmin, "failing_size": fail, "failing_size_mw": fail * ref}
                if scale == 0.0:
                    xc = max(0.0, size - fail * ref) if math.isfinite(fail) else 0.0
                    cur = [curtailment(y["L"], xc, headroom * y["P"]) for y in years]
                    case.update({"curtailable_mw": xc, "curtail_hours": _median([c["hours"] for c in cur]),
                                 "curtail_energy_share": _median([c["energy_share"] for c in cur]),
                                 "curtail_full_size_hours": _median([c["full_size_hours"] for c in cur])})
                if has_classes and (size > 0 or scale > 0):
                    case["by_class"] = _class_block(years, per, today, tau)
                    if scale == 0.0:
                        xcc = max(0.0, size - cfail * ref) if math.isfinite(cfail) else 0.0
                        cur = [curtailment(y["L"], xcc, headroom * y["P"]) for y in years]
                        case["by_class"].update({"failing_size": cfail, "failing_size_mw": cfail * ref, "curtailable_mw": xcc,
                                                 "curtail_hours": _median([c["hours"] for c in cur]), "curtail_energy_share": _median([c["energy_share"] for c in cur])})
                res["cases"].append(case)
    return res


# ======================================================================================================================
# 4. The fleet-based benchmark (hourly load and the list of generating units)
# ======================================================================================================================
def make_unit(capacity_mw, forced_outage_rate, maintenance_weeks=0, start_week=None, name=""):
    """A generating unit with two states: available at its capacity, or on forced outage with the stated probability.
    maintenance_weeks: consecutive weeks of planned outage a year. start_week: 1..52 if the week is fixed; None lets
    the tool place it."""
    if capacity_mw < 0 or not 0.0 <= forced_outage_rate < 1.0 or not 0 <= maintenance_weeks < WEEKS:
        raise ValueError("a unit needs a capacity that is not negative, an outage rate in [0, 1) and 0 to 51 weeks of maintenance")
    return {"mw": float(capacity_mw), "q": float(forced_outage_rate), "weeks": int(maintenance_weeks),
            "start": None if start_week in (None, "") else int(start_week), "name": str(name)}


def outage_table(units, step=1.0):
    """Probability of each level of available capacity of independent two-state units, on a grid of `step` megawatts:
    table[k] = Pr(available capacity = k * step). Built by adding one unit at a time (recursive convolution)."""
    sizes = [int(round(u["mw"] / step)) for u in units]
    table = np.zeros(sum(sizes) + 1)
    table[0] = 1.0
    top = 0
    for k, u in zip(sizes, units):
        new = u["q"] * table
        if k:
            new[k:top + k + 1] += (1.0 - u["q"]) * table[:top + 1]
        else:
            new = table.copy()
        table, top = new, top + k
    return table


def _below(cum, y, step):
    """Pr(available capacity < y) for an array y (MW). cum[j] = Pr(capacity < j * step)."""
    idx = np.ceil(np.asarray(y, float) / step - 1e-9).astype(np.int64)
    return cum[np.clip(idx, 0, len(cum) - 1)]


def _cumulative(table):
    """cum[j] = Pr(available capacity < j * step), j = 0 .. len(table)."""
    return np.concatenate([[0.0], np.cumsum(table)])


def schedule_maintenance(units, weekly_peak):
    """Weeks (0..51) in which each unit is on planned maintenance. Units with a fixed start week keep it. The others
    are placed one at a time, the largest first, each into the run of consecutive weeks in which the highest value of
    weekly peak load plus capacity already on maintenance is lowest (a levelized reserve)."""
    level = np.asarray(weekly_peak, float).copy()
    out = [np.array([], int)] * len(units)
    for i, u in enumerate(units):
        if u["weeks"] and u["start"] is not None:
            out[i] = (u["start"] - 1 + np.arange(u["weeks"])) % WEEKS
            level[out[i]] += u["mw"]
    for i in sorted(range(len(units)), key=lambda j: -units[j]["mw"]):
        u = units[i]
        if not u["weeks"] or u["start"] is not None:
            continue
        best = min(range(WEEKS), key=lambda s: (level[(s + np.arange(u["weeks"])) % WEEKS].max(), s))
        out[i] = (best + np.arange(u["weeks"])) % WEEKS
        level[out[i]] += u["mw"]
    return out


def expand_fleet(units, factor, rule="same-kinds", step=1.0):
    """A fleet for a system whose peak is `factor` times the peak of today.
    'same-kinds'    more units of the kinds already there, in the proportions of today as far as whole units allow:
                    one unit at a time of the kind whose capacity is furthest below `factor` times its capacity of
                    today, until the installed capacity is as close to the target as whole units allow;
    'larger-units'  every unit larger by the factor, the number of units unchanged."""
    if factor <= 1.0 + 1e-12:
        return [dict(u) for u in units]
    if rule == "larger-units":
        return [dict(u, mw=round(u["mw"] * factor / step) * step) for u in units]
    if rule != "same-kinds":
        raise ValueError("the expansion rule is 'same-kinds' or 'larger-units'")
    kinds = {}
    for u in units:
        kinds.setdefault((u["mw"], u["q"], u["weeks"]), []).append(u)
    have = {k: len(v) * k[0] for k, v in kinds.items()}
    want = {k: factor * len(v) * k[0] for k, v in kinds.items()}
    target, total, out = factor * sum(u["mw"] for u in units), sum(u["mw"] for u in units), [dict(u) for u in units]
    while True:
        k = max(kinds, key=lambda kk: (want[kk] - have[kk], kk[0]))
        if abs(total + k[0] - target) > abs(total - target):
            return out
        out.append(dict(kinds[k][0], start=None))
        have[k] += k[0]
        total += k[0]


class Fleet:
    """The risk model of one year: the units, the weeks of maintenance, and an outage table for every week."""

    def __init__(self, units, covered_load, step=1.0):
        """covered_load: the load that the units have to cover in each hour (existing load plus new load, less variable
        generation), used here only to place the maintenance."""
        if len(covered_load) < WEEKS * HOURS_PER_WEEK:
            raise ValueError("the fleet-based benchmark needs a year of at least 8,736 hours")
        self.units, self.step = units, step
        self.week = week_of_hour(len(covered_load))
        weekly_peak = np.array([covered_load[self.week == k].max() for k in range(WEEKS)])
        self.schedule = schedule_maintenance(units, weekly_peak)
        cache, self.cum = {}, []
        for k in range(WEEKS):
            out = frozenset(i for i, wk in enumerate(self.schedule) if k in wk)
            if out not in cache:
                cache[out] = _cumulative(outage_table([u for i, u in enumerate(units) if i not in out], step))
            self.cum.append(cache[out])
        self.installed = sum(u["mw"] for u in units)

    def risk(self, covered_load, firm_block=0.0):
        """Loss-of-load probability of each hour: Pr(available capacity + firm_block < load)."""
        r = np.empty(len(covered_load))
        for k in range(WEEKS):
            sel = self.week == k
            r[sel] = _below(self.cum[k], covered_load[sel] - firm_block, self.step)
        return r

    def firm_block_for(self, covered_load, target_hours):
        """Megawatts of capacity without outages, positive or negative, that bring the yearly risk to the target. The
        yearly risk falls in steps as the block grows, so the target is in general not met exactly: the block returned
        is the smallest one at which the yearly risk is below the target."""
        lo = -(self.installed + float(np.max(covered_load)))
        hi = -lo
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:      # the two sides are neighbouring numbers
                break
            if float(self.risk(covered_load, mid).sum()) < target_hours:
                hi = mid
            else:
                lo = mid
        return hi

    def expected_cut(self, covered_firm, curtailable_mw, firm_block=0.0):
        """Expected hours and energy of curtailment of megawatts that are cut whenever the available capacity is below
        the firm load plus the curtailable megawatts, by the shortfall and at most in full."""
        hours = float(self.risk(covered_firm + curtailable_mw, firm_block).sum())
        energy = 0.0
        for k in range(WEEKS):
            sel = self.week == k
            cum, s = self.cum[k], self.step
            prefix = np.concatenate([[0.0], np.cumsum(cum[1:]) * s])      # integral of Pr(capacity < v) from 0 to j * step

            def integral(v):
                v = np.clip(np.asarray(v, float), 0.0, (len(cum) - 1) * s)
                j = np.minimum(np.floor(v / s + 1e-12).astype(np.int64), len(cum) - 2)
                return prefix[j] + (v - j * s) * cum[j + 1]
            hi = covered_firm[sel] + curtailable_mw - firm_block
            beyond = np.maximum(0.0, hi - (len(cum) - 1) * s) - np.maximum(0.0, hi - curtailable_mw - (len(cum) - 1) * s)
            energy += float((integral(hi) - integral(hi - curtailable_mw) + beyond).sum())
        return hours, energy


def fleet_based(year, units, method, size_mw=0.0, scale=0.0, rule="same-kinds", units_after=None, target_hours=None,
                step=1.0, state=None):
    """One year under the fleet-based benchmark.

    Today: the units as given. If target_hours is None the standard is the yearly risk of the system as it stands;
    otherwise a block of capacity without outages, positive or negative, is added so that the yearly risk of today
    equals target_hours. With the new load: the fleet of `units_after` if given, otherwise the fleet enlarged by `rule`
    in the proportion in which the system peak grows; then a block of capacity without outages, positive or negative,
    closes what is left to the standard. The weights are the shares of the hours in the yearly risk."""
    L = year["L"]
    G = year.get("G", 0.0)
    if state is None:
        today = Fleet(units, L - G, step)
        block0 = 0.0 if target_hours is None else today.firm_block_for(L - G, target_hours)
        r0 = today.risk(L - G, block0)
        state = {"today": today, "block0": block0, "r0": r0, "standard": float(r0.sum()) if target_hours is None else float(target_hours)}
    new = _new_load(year, size_mw, scale)
    total = L + new
    if float(np.abs(new).max()) == 0.0:
        fleet, block = state["today"], state["block0"]
    else:
        growth = float(total.max()) / year["P"]
        fleet = Fleet(units_after if units_after is not None else expand_fleet(units, growth, rule, step), total - G, step)
        block = fleet.firm_block_for(total - G, state["standard"])
    holder = {}

    def weights(profile):
        r = fleet.risk(profile, block)
        holder["risk"] = float(r.sum())
        if holder["risk"] <= 0:
            raise ValueError("the yearly risk of the system is zero: with these units the load is covered in every state of the fleet, so there is no risk "
                             "of shortage to divide among the hours. Give the forced outage rates of the units, or a standard with --target-hours")
        return r / r.sum()
    out = _shares(year, method, new, weights)
    out.update({"standard_hours": state["standard"], "risk_hours": holder["risk"], "installed_mw": fleet.installed,
                "firm_block_mw": block, "units": len(fleet.units), "reserve": (fleet.installed + block) / float(total.max()) - 1.0,
                "fleet": fleet, "state": state})
    return out


def run_fleet_based(years, units, method="12CP", sizes_mw=(0.0,), tau=0.05, rule="same-kinds", units_after=None,
                    target_hours=None, step=1.0, scales=(0.0,), grid=None, headroom=0.0, floor=0.04, gammas=(1.0, 0.5)):
    """The whole test with the fleet-based benchmark. grid: sizes as shares of the existing peak on which the overcharge
    is evaluated to find the failing size (default 0.05, 0.10, ..., 2.00); the failing size is the first crossing of the
    tolerance, interpolated between two points of the grid. floor, gammas: the lower bound of the window and the growth
    exponents of the load-based benchmark that is run beside the fleet-based one for the floor of the tolerance."""
    ref = _reference_peak(years)
    grid = np.round(np.arange(0.05, 2.0001, 0.05), 4) if grid is None else np.asarray(grid, float)
    states = [fleet_based(y, units, method, 0.0, target_hours=target_hours, step=step) for y in years]
    kappa = _median([s["la"] / y["P"] for s, y in zip(states, years)])
    lam = _median([y["A"] / y["P"] for y in years])
    kstar0 = _median([s["lb"] / y["P"] for s, y in zip(states, years)])
    eqw = _median([equivalent_window(y["L"], s["lb"], y["L"] - y["G"] if "G" in y else None) / y["P"] for s, y in zip(states, years)])
    res = {"benchmark": "fleet", "method": method, "tolerance": tau, "years": [y["label"] for y in years], "peak_mw": ref,
           "load_factor": lam, "kappa_a": kappa, "kappa_star_today": kstar0, "equivalent_window_share": eqw,
           "standard_hours": _median([s["standard_hours"] for s in states]), "installed_mw": states[0]["installed_mw"],
           "units": len(units), "reserve_today": _median([s["reserve"] for s in states]), "expansion": "the fleet given for the system with the new load" if units_after is not None else rule,
           "overcharge_ceiling": overcharge_ceiling(kappa, lam), "safe_size": safe_size(kappa, lam, tau), "cases": []}
    res["grid_max"] = float(np.max(grid))

    has_classes = all("K" in y for y in years)
    res["classes"] = sorted(years[0]["K"]) if has_classes else None
    try:      # the window of the load-based benchmark that is run beside the fleet-based one
        mu_load = max(floor, implied_window_share(years, method))
    except ValueError:
        mu_load = None
    res["window_load_based"] = mu_load

    def eps_at(size, scale=0.0):
        return [fleet_based(y, units, method, size, scale, rule, units_after, target_hours, step, s["state"]) for y, s in zip(years, states)]

    def crossing(curve):
        for (x0, e0), (x1, e1) in zip(curve[:-1], curve[1:]):
            if e0 <= tau < e1:
                return x0 + (tau - e0) / (e1 - e0) * (x1 - x0)
        return float("inf")
    cfail = None
    if units_after is None:
        curve, ccurve = [(0.0, 0.0)], [(0.0, 0.0)]
        for x in grid:
            per = eps_at(x * ref)
            curve.append((float(x), _median([p["eps"] for p in per])))
            if has_classes:
                ccurve.append((float(x), most_moved(class_table(years, per, states))[1]))
        fail = crossing(curve)
        res["curve"] = curve
        if has_classes:
            cfail = crossing(ccurve)
            res["class_curve"], res["class_failing_size"] = ccurve, cfail
    else:
        fail = None      # one fleet for one size: no curve
    res["failing_size"] = fail
    for size in sizes_mw:
        for scale in scales:
            per = eps_at(size, scale)
            eps = _median([p["eps"] for p in per])
            case = {"size_mw": size, "scale": scale, "size_share": size / ref,
                    "kappa_star": _median([p["lb"] / y["P"] for p, y in zip(per, years)]), "o_a": _median([p["o_a"] for p in per]),
                    "o_b": _median([p["o_b"] for p in per]), "o_e": _median([p["o_e"] for p in per]), "overcharge": eps,
                    "passes": bool(eps <= tau), "min_energy_share": _median([min_energy_share(p["o_a"], p["o_b"], p["o_e"], tau) for p in per]),
                    "units": per[0]["units"], "installed_mw": per[0]["installed_mw"], "firm_block_mw": _median([p["firm_block_mw"] for p in per]),
                    "reserve": _median([p["reserve"] for p in per]), "risk_hours": _median([p["risk_hours"] for p in per])}
            if scale == 0.0 and size > 0:
                # the window at which the load-based benchmark gives the same benchmark demand, and the exponent with
                # which that window has grown from the window of today
                eq = _median([equivalent_window(y["L"], p["lb"], y["L"] - y["G"] if "G" in y else None) / y["P"] for p, y in zip(per, years)])
                case["equivalent_window_share"] = eq
                case["growth_exponent"] = (math.log(eq / eqw) / math.log(1.0 + size / ref) + 0.0) if 0 < eq < float("inf") and 0 < eqw < float("inf") else None
            if mu_load is not None and gammas and (size > 0 or scale > 0):
                # the versions of the benchmark: the fleet-based one, and the load-based one for each exponent
                case["o_b_load_based"] = {f"{g:g}": _median([load_based(y, method, mu_load, g, size, scale)["o_b"] for y in years]) for g in gammas}
                case["tolerance_floor"] = tolerance_floor([case["o_b"]] + list(case["o_b_load_based"].values()))
            if has_classes and (size > 0 or scale > 0):
                case["by_class"] = _class_block(years, per, states, tau)
                if scale == 0.0 and cfail is not None:
                    xcc = max(0.0, size - cfail * ref) if math.isfinite(cfail) else 0.0
                    cur = [curtailment(y["L"], xcc, headroom * y["P"]) for y in years]
                    case["by_class"].update({"failing_size": cfail, "failing_size_mw": cfail * ref, "curtailable_mw": xcc,
                                             "curtail_hours": _median([c["hours"] for c in cur]), "curtail_energy_share": _median([c["energy_share"] for c in cur])})
            if scale == 0.0 and fail is not None:
                xc = max(0.0, size - fail * ref) if math.isfinite(fail) else 0.0
                case["curtailable_mw"] = xc
                cur = [curtailment(y["L"], xc, headroom * y["P"]) for y in years]
                case.update({"curtail_hours": _median([c["hours"] for c in cur]), "curtail_energy_share": _median([c["energy_share"] for c in cur])})
                if xc > 0:
                    firm = eps_at(size - xc)
                    exp = [f["fleet"].expected_cut(y["L"] + (size - xc) - (y["G"] if "G" in y else 0.0), xc, f["firm_block_mw"]) for f, y in zip(firm, years)]
                    case.update({"expected_curtail_hours": _median([e[0] for e in exp]),
                                 "expected_curtail_energy_share": _median([e[1] / (len(y["L"]) * xc) for e, y in zip(exp, years)])})
            res["cases"].append(case)
    return res


# ======================================================================================================================
# 5. The IEEE Reliability Test System (data as printed in Billinton and Allan, 1996, Appendix 2)
# ======================================================================================================================
# Table A2.1: capacity (MW), forced outage rate, weeks of scheduled maintenance a year, number of units
RTS_UNITS = ((12, 0.02, 2, 5), (20, 0.10, 2, 4), (50, 0.01, 2, 6), (76, 0.02, 3, 4), (100, 0.04, 3, 3), (155, 0.04, 4, 4),
             (197, 0.05, 4, 3), (350, 0.08, 5, 1), (400, 0.12, 6, 2))
# Table A2.2: weekly peak load in percent of the annual peak, weeks 1 to 52
RTS_WEEK = (86.2, 90.0, 87.8, 83.4, 88.0, 84.1, 83.2, 80.6, 74.0, 73.7, 71.5, 72.7, 70.4, 75.0, 72.1, 80.0, 75.4, 83.7, 87.0, 88.0,
            85.6, 81.1, 90.0, 88.7, 89.6, 86.1, 75.5, 81.6, 80.1, 88.0, 72.2, 77.6, 80.0, 72.9, 72.6, 70.5, 78.0, 69.5, 72.4, 72.4,
            74.3, 74.4, 80.0, 88.1, 88.5, 90.9, 94.0, 89.0, 94.2, 97.0, 100.0, 95.2)
# Table A2.3: daily peak load in percent of the weekly peak, Monday to Sunday
RTS_DAY = (93, 100, 98, 96, 94, 77, 75)
# Table A2.4: hourly load in percent of the daily peak, hours 1 to 24: (weekday, weekend) for winter weeks (1-8 and 44-52),
# summer weeks (18-30) and spring and fall weeks (9-17 and 31-43)
RTS_HOUR = {
    "winter": ((67, 63, 60, 59, 59, 60, 74, 86, 95, 96, 96, 95, 95, 95, 93, 94, 99, 100, 100, 96, 91, 83, 73, 63),
               (78, 72, 68, 66, 64, 65, 66, 70, 80, 88, 90, 91, 90, 88, 87, 87, 91, 100, 99, 97, 94, 92, 87, 81)),
    "summer": ((64, 60, 58, 56, 56, 58, 64, 76, 87, 95, 99, 100, 99, 100, 100, 97, 96, 96, 93, 92, 92, 93, 87, 72),
               (74, 70, 66, 65, 64, 62, 62, 66, 81, 86, 91, 93, 93, 92, 91, 91, 92, 94, 95, 95, 100, 93, 88, 80)),
    "spring": ((63, 62, 60, 58, 59, 65, 72, 85, 95, 99, 100, 99, 93, 92, 90, 88, 90, 92, 96, 98, 96, 90, 80, 70),
               (75, 73, 69, 66, 65, 65, 68, 74, 83, 89, 92, 94, 91, 90, 90, 86, 85, 88, 92, 100, 97, 95, 90, 85))}
RTS_PEAK = 2850.0
RTS_STEADY_MW = 600.0      # megawatts of the steady class in the illustration of the class columns (not part of the published system)
# Published results without maintenance (Billinton and Allan, 1996): loss-of-load expectation with the 364 daily peaks
# (days a year) and with the 8,736 hourly loads (hours a year), and Table A2.9: days a year for other annual peaks
RTS_LOLE_DAYS, RTS_LOLE_HOURS = 1.36886, 9.39418
# the exact energy indices of the basic system: energy not served (GWh a year) and energy index of reliability
RTS_LOEE_GWH, RTS_EIR = 1.176, 0.999923
# Table A2.5: capacity on outage (MW) and the cumulative probability of that outage or more
RTS_TABLE_A25 = {100: 0.54760141, 200: 0.38132840, 265: 0.33556693, 400: 0.26187364, 556: 0.08457820, 600: 0.06211297,
                 950: 0.00749197, 1200: 0.00079125, 1500: 0.00004043}
RTS_TABLE_A29 = {1.10: 6.68051, 1.06: 3.77860, 1.04: 2.67126, 1.00: 1.36886, 0.96: 0.65219, 0.92: 0.29734, 0.88: 0.12174, 0.84: 0.04756}


def rts_units(maintenance=True):
    """The 32 units of the test system. maintenance=False sets the weeks of scheduled maintenance to zero."""
    return [make_unit(mw, q, wk if maintenance else 0, name=f"{mw} MW unit {i + 1}") for mw, q, wk, n in RTS_UNITS for i in range(n)]


def _rts_season(week):
    return "winter" if (week <= 8 or week >= 44) else "summer" if 18 <= week <= 30 else "spring"


def rts_load(peak=RTS_PEAK):
    """Hourly load (8,736 hours, week 1 starts on a Monday) and the 364 daily peaks of the test system (MW)."""
    hourly, daily = [], []
    for w in range(1, 53):
        for d in range(7):
            day_peak = peak * RTS_WEEK[w - 1] / 100.0 * RTS_DAY[d] / 100.0
            daily.append(day_peak)
            hourly.extend(day_peak * h / 100.0 for h in RTS_HOUR[_rts_season(w)][0 if d < 5 else 1])
    return np.array(hourly), np.array(daily)


def rts_year(peak=RTS_PEAK):
    hourly, _ = rts_load(peak)
    return make_year(hourly, month_of_hour(len(hourly)), "IEEE RTS")


# ======================================================================================================================
# 6. Files
# ======================================================================================================================
def _number(text, what, row):
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ValueError(f"row {row}: {what} is not a number: {text!r}") from None


def read_load_csv(path, hour_ending=False, new_load_column=None, class_columns=None):
    """Hourly load from a CSV file with a header line. Columns (others are ignored):
      load_mw                  required, unless class columns are named: the load is then their sum
      <class columns>          optional: the hourly MW of each customer class. If load_mw is given as well, what the
                               classes leave of it is carried as a class of its own, called "other"
      timestamp                optional: 2025-07-14 16:00 or 2025-07-14T16:00; gives the year and the month of each hour
      year, month              optional: used if there is no timestamp
      variable_generation_mw   optional: wind, solar and other must-take output, taken off the load for the risk
      <new_load_column>        optional: hourly profile of a new load that is not flat
    Without timestamp, year and month the file is taken as one year that starts on 1 January.
    hour_ending=True: the time stamp marks the end of the hour, so the hour is labelled by the time one hour earlier.
    Hours of 29 February are dropped. Returns a list of years."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    class_columns = list(class_columns or [])
    if not rows or ("load_mw" not in rows[0] and not class_columns):
        raise ValueError("the load file needs a header line with a column load_mw")
    cols = rows[0].keys()
    for name in class_columns + ([new_load_column] if new_load_column else []):
        if name not in cols:
            raise ValueError(f"the load file has no column {name!r}")
    if len(set(class_columns)) != len(class_columns) or "other" in class_columns:
        raise ValueError("the class columns have to have different names, and none may be called 'other'")
    recs = []
    for i, row in enumerate(rows, start=2):
        yr, mo = row.get("year") or "", row.get("month") or ""
        if "timestamp" in cols and row["timestamp"]:
            t = _dt.datetime.fromisoformat(row["timestamp"].strip().replace("T", " ").replace("Z", ""))
            t = t - _dt.timedelta(hours=1) if hour_ending else t
            if t.month == 2 and t.day == 29:
                continue
            yr, mo = t.year, t.month
        rec = {"year": str(yr), "month": int(float(mo)) if mo != "" else None}
        if class_columns:
            rec["K"] = [_number(row[name], name, i) for name in class_columns]
        rec["L"] = _number(row["load_mw"], "load_mw", i) if "load_mw" in cols else sum(rec["K"])
        if "variable_generation_mw" in cols:
            rec["G"] = _number(row["variable_generation_mw"] or 0.0, "variable_generation_mw", i)
        if new_load_column:
            rec["N"] = _number(row[new_load_column], new_load_column, i)
        recs.append(rec)
    years, order = {}, []
    for rec in recs:
        if rec["year"] not in years:
            years[rec["year"]] = []
            order.append(rec["year"])
        years[rec["year"]].append(rec)
    out = []
    for key in order:
        part = years[key]
        if len(part) < 8000:
            print(f"note: year {key or '(no label)'} has only {len(part)} hours and is left out", file=sys.stderr)
            continue
        month = [r["month"] for r in part] if part[0]["month"] is not None else None
        classes = None
        if class_columns:
            classes = {name: np.array([r["K"][j] for r in part]) for j, name in enumerate(class_columns)}
            load = np.array([r["L"] for r in part])
            rest = load - sum(classes.values())
            if float(rest.min()) < -0.01 * float(load.max()):
                raise ValueError(f"year {key or '(no label)'}: the classes add up to more than the load")
            if float(np.abs(rest).max()) > 1e-6 * float(load.max()):
                classes["other"] = rest
        out.append(make_year([r["L"] for r in part], month, key, [r["G"] for r in part] if "G" in part[0] else None,
                             [r["N"] for r in part] if "N" in part[0] else None, classes))
    if not out:
        raise ValueError("no complete year of hourly load in the file")
    return out


def read_units_csv(path):
    """Generating units from a CSV file with a header line. Columns: capacity_mw and forced_outage_rate (required);
    maintenance_weeks, maintenance_start_week (1..52), name and count (optional; count repeats the row)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "capacity_mw" not in rows[0] or "forced_outage_rate" not in rows[0]:
        raise ValueError("the units file needs a header line with the columns capacity_mw and forced_outage_rate")
    units = []
    for i, row in enumerate(rows, start=2):
        n = int(float(row.get("count") or 1))
        for _ in range(n):
            units.append(make_unit(_number(row["capacity_mw"], "capacity_mw", i), _number(row["forced_outage_rate"], "forced_outage_rate", i),
                                   int(float(row.get("maintenance_weeks") or 0)), row.get("maintenance_start_week") or None, row.get("name") or ""))
    return units


def write_rts_files(out_dir):
    """Writes the test system as the two input files of the tool: rts_load.csv and rts_units.csv."""
    os.makedirs(out_dir, exist_ok=True)
    hourly, _ = rts_load()
    month = month_of_hour(len(hourly))
    with open(os.path.join(out_dir, "rts_load.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["hour", "month", "load_mw"])
        wr.writerows([i + 1, int(month[i]), f"{hourly[i]:.4f}"] for i in range(len(hourly)))
    with open(os.path.join(out_dir, "rts_units.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["name", "capacity_mw", "forced_outage_rate", "maintenance_weeks", "count"])
        wr.writerows([f"{mw} MW", mw, q, wk, n] for mw, q, wk, n in RTS_UNITS)
    # an illustration of the class columns: the same load divided into a steady class and the rest. The test system
    # has no customer classes; the division is made up to show the mechanics and is not part of the published system.
    with open(os.path.join(out_dir, "rts_classes.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["hour", "month", "steady_mw", "varying_mw"])
        wr.writerows([i + 1, int(month[i]), f"{RTS_STEADY_MW:.4f}", f"{hourly[i] - RTS_STEADY_MW:.4f}"] for i in range(len(hourly)))
    return out_dir


# ======================================================================================================================
# 7. Report
# ======================================================================================================================
def _pct(v, digits=1):
    return "none" if v is None else "no limit" if not math.isfinite(v) else f"{100 * v:.{digits}f}%"


def _mw(v):
    return "no limit" if not math.isfinite(v) else f"{v:,.0f} MW"


def report(res):
    """The result of run_load_based or run_fleet_based as plain text."""
    tau, lines = res["tolerance"], []
    add = lines.append
    add(f"Coincident peak test, {'load-based' if res['benchmark'] == 'load' else 'fleet-based'} benchmark (cptest {__version__})")
    add(f"  method {res['method']}; tolerance {_pct(tau, 0)}; years of load: {len(res['years'])}" + (f" ({', '.join(y for y in res['years'] if y)})" if any(res["years"]) else ""))
    add(f"  existing peak P = {res['peak_mw']:,.0f} MW (median of the years); load factor lambda = {res['load_factor']:.3f}")
    add(f"  the existing method counts the existing load at kappa_a = {res['kappa_a']:.3f} of its peak")
    add(f"  ceiling of the overcharge (the most that a flat load of any size can shift to the existing customers): {_pct(res['overcharge_ceiling'])}")
    if math.isfinite(res["safe_size"]):
        add(f"  safe size (passes whatever the benchmark, if it counts the existing load at no less than its average): {_pct(res['safe_size'])} of the peak")
    else:
        add("  safe size: none is needed. The tolerance is not below the ceiling, so the existing method passes for a flat load of any size")
    if res["benchmark"] == "load":
        add(f"  window implied by the existing method today: {_pct(res['implied_window_share'])} of the peak; window used: {_pct(res['window_share'])} ({res['window_rule']})")
        if res["window_share"] <= 0:
            add("  note: a window of zero stays zero as the system grows. The benchmark is then the hour of the peak at every size, and no load can fail. Set a floor above zero")
    else:
        add(f"  fleet today: {res['units']} units, {res['installed_mw']:,.0f} MW installed, reserve {_pct(res['reserve_today'])}; standard held: {res['standard_hours']:.2f} hours of loss of load a year")
        add(f"  the benchmark counts the existing load today at kappa* = {res['kappa_star_today']:.3f}; the load-based benchmark gives the same with a window of {_pct(res['equivalent_window_share'])} of the peak")
        add(f"  expansion of the fleet with the new load: {res['expansion']}")
        if res.get("failing_size") is not None:
            if math.isfinite(res["failing_size"]):
                add(f"  failing size (first crossing of the tolerance on the grid of sizes): {_pct(res['failing_size'], 0)} of the peak")
            else:
                add(f"  failing size: none on the grid of sizes, up to {_pct(res['grid_max'], 0)} of the peak. The overcharge stays within the tolerance there")
    if res.get("classes"):
        add(f"  customer classes: {', '.join(res['classes'])}")
    head = None
    for c in res["cases"]:
        if res["benchmark"] == "load" and c["gamma"] != head:
            head = c["gamma"]
            add("")
            if math.isfinite(c["failing_size"]):
                add(f"  Window grows with the system peak with exponent gamma = {head:g}: failing size {_pct(c['failing_size'], 0)} of the peak ({_mw(c['failing_size_mw'])})")
            else:
                add(f"  Window grows with the system peak with exponent gamma = {head:g}: no failing size. The overcharge stays within the tolerance at every size")
        elif res["benchmark"] == "fleet" and head is None:
            head = True
            add("")
        what = f"flat load {c['size_mw']:,.0f} MW" + (f" plus {c['scale']:g} times the profile" if c["scale"] else "")
        add(f"    {what} ({_pct(c['size_share'], 0)} of the peak): share of the existing customers under the existing method {_pct(c['o_a'])}, under the benchmark {_pct(c['o_b'])}; "
            f"overcharge {_pct(c['overcharge'], 2)} -> {'passes' if c['passes'] else 'FAILS'}")
        if not c["passes"]:
            if math.isfinite(c["min_energy_share"]):
                extra = f"      remedy 1: at least {_pct(c['min_energy_share'])} of the pool on energy"
            else:
                extra = "      remedy 1: no share of the pool on energy suffices, because this benchmark counts the existing customers below their share of the energy"
            if "curtailable_mw" in c:
                extra += f"; remedy 2: {c['curtailable_mw']:,.0f} MW curtailable, cut in {c['curtail_hours']:,.0f} hours a year for {_pct(c['curtail_energy_share'])} of their energy"
                if "expected_curtail_hours" in c:
                    extra += f" (expected with the outage table: {c['expected_curtail_hours']:,.0f} hours, {_pct(c['expected_curtail_energy_share'], 2)})"
            add(extra)
        if c.get("equivalent_window_share") is not None:
            grown = "" if c.get("growth_exponent") is None else f", grown from the window of today with the exponent {c['growth_exponent']:.2f}"
            add(f"      the load-based benchmark gives the same benchmark demand with a window of {_pct(c['equivalent_window_share'])} of the peak{grown}")
        if c.get("tolerance_floor") is not None:
            others = ", ".join(f"{_pct(v)} for the exponent {g}" for g, v in c["o_b_load_based"].items())
            add(f"      floor of the tolerance: {_pct(c['tolerance_floor'])} (benchmark share of the existing customers: {_pct(c['o_b'])} fleet-based; load-based from the load alone {others})")
        if "by_class" in c:
            lines.extend(_class_report(c["by_class"], tau))
    return "\n".join(lines)


def _signed(v, digits=1):
    """A share as a signed percentage; a value that rounds to zero is printed without a sign."""
    t = f"{100 * v:+.{digits}f}%"
    return t[1:] if float(t[:-1]) == 0 else t


def _class_report(b, tau):
    """The class results of one case as lines of text."""
    width = max(len(k) for k in b["table"]) + 2
    out = ["      by class: share of the pool under the existing method and under the benchmark; gap of today; what the new load adds"]
    for k, r in b["table"].items():
        if k != NEW_LOAD and r["added"] is None:      # the class has no load in the hours that the existing method names
            out.append(f"        {k:<{width}s}{_pct(r['share_method']):>7s} {_pct(r['share_benchmark']):>7s}   not counted by the existing method today; left out of the verdict")
            continue
        today = "        " if r["factor_today"] is None else f"{_signed(r['factor_today'] - 1.0):>8s}"
        out.append(f"        {k:<{width}s}{_pct(r['share_method']):>7s} {_pct(r['share_benchmark']):>7s} {today} {_signed(r['added']):>8s}")
    if b["most_moved"] is None:
        return out + ["      no class is counted by the existing method today, so no verdict by class can be given"]
    line = f"      the class that is moved up most: {b['most_moved']}, {_signed(b['added_max'])} -> {'passes' if b['passes'] else 'FAILS'}"
    if "failing_size" in b:
        line += f"; failing size by class {_pct(b['failing_size'], 0)} of the peak"
    out.append(line)
    if not b["passes"]:
        if b["energy_share"] is None:
            out.append("      remedy 1 by class: no share of the pool on energy brings every class within the tolerance")
        else:
            extra = f"      remedy 1 by class: at least {_pct(b['energy_share'])} of the pool on energy"
            top = b["highest_with_energy_share"]
            if top["added"] > tau + 1e-9:
                extra += f" (at that share the class {top['class']} is at {_signed(top['added'])})"
            out.append(extra)
        if "curtailable_mw" in b:
            out.append(f"      remedy 2 by class: {b['curtailable_mw']:,.0f} MW curtailable, cut in {b['curtail_hours']:,.0f} hours a year for {_pct(b['curtail_energy_share'])} of their energy")
    return out


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items() if k not in ("weights", "fleet", "state", "hours")}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ======================================================================================================================
# 8. Self-test
# ======================================================================================================================
def selftest(verbose=True):
    """Checks the two benchmarks against published values and against identities of the theory. Returns True if all
    checks hold."""
    ok, say = [], (print if verbose else (lambda *a, **k: None))

    def check(name, got, want, tol):
        good = abs(got - want) <= tol
        ok.append(good)
        say(f"  {'ok  ' if good else 'FAIL'} {name}: {got:.6g} (expected {want:.6g})")

    say("1. The outage table against the published results of the IEEE Reliability Test System (no maintenance)")
    units, (hourly, daily) = rts_units(False), rts_load()
    cum = _cumulative(outage_table(units))
    check("installed capacity, MW", sum(u["mw"] for u in units), 3405, 0)
    check("loss-of-load expectation with daily peaks, days a year", float(_below(cum, daily, 1.0).sum()), RTS_LOLE_DAYS, 5e-6)
    check("loss-of-load expectation with hourly loads, hours a year", float(_below(cum, hourly, 1.0).sum()), RTS_LOLE_HOURS, 5e-6)
    check("probability that no unit is on outage (Table A2.5, to six decimals)", float(outage_table(units)[3405]), 0.23639495, 5e-7)
    for out_mw, v in RTS_TABLE_A25.items():
        check(f"probability of {out_mw} MW or more on outage (Table A2.5, to six decimals)", float(cum[3405 - out_mw + 1]), v, 5e-7)
    for k, v in RTS_TABLE_A29.items():
        check(f"annual peak at {k:.2f} of 2,850 MW, days a year", float(_below(cum, rts_load(RTS_PEAK * k)[1], 1.0).sum()), v, 5e-6)

    say("2. The load-based benchmark on a system of three hours (5,000, 4,500 and 4,000 MW; window 500 MW)")
    three = np.array([5000.0, 4500.0, 4000.0])
    w = exp_weights(three, 500.0)
    for got, want in zip(w, (0.665241, 0.244728, 0.090031)):
        check("weight", float(got), want, 5e-7)
    check("benchmark demand, MW", benchmark_demand(three, w), 4787.605, 5e-4)

    say("3. Identities of the theory, on the hourly load of the test system")
    year = rts_year()
    L, P, A = year["L"], year["P"], year["A"]
    mu = implied_window_share([year], "12CP")
    s0 = load_based(year, "12CP", mu)
    check("at the implied window the benchmark demand equals the method's demand (ratio)", s0["lb"] / s0["la"], 1.0, 1e-9)
    X = 0.8 * P
    s = load_based(year, "12CP", mu, 1.0, X)
    check("a flat load is counted at its megawatts by the benchmark (ratio)", s["nb"] / X, 1.0, 1e-12)
    check("overcharge = share of the new load times the gap", s["eps"], X / (s["la"] + X) * (s["la"] / s["lb"] - 1.0), 1e-12)
    kap, kst, lam, x = s["la"] / P, s["lb"] / P, A / P, X / P
    check("overcharge in dimensionless form", s["eps"], x / (kap + x) * (kap / kst - 1.0), 1e-12)
    tau = 0.02
    f = min_energy_share(s["o_a"], s["o_b"], s["o_e"], tau)
    check("at the smallest energy share the overcharge equals the tolerance", ((1 - f) * s["o_a"] + f * s["o_e"]) / s["o_b"] - 1.0, tau, 1e-12)
    check("smallest energy share in dimensionless form", f, (s["eps"] - tau) / (1 + s["eps"]) * kap * (lam + x) / (x * (kap - lam)), 1e-12)
    xs = safe_size(kap, lam, 0.05)
    check("at the safe size an energy benchmark gives exactly the tolerance", xs / (kap + xs) * (kap / lam - 1.0), 0.05, 1e-12)
    grown = make_year(L, year["month"], "", new_load=L)
    check("a new load with the shape of the existing load: overcharge", load_based(grown, "12CP", mu, 1.0, 0.0, 0.7)["eps"], 0.0, 1e-12)
    cut = curtailment(L, 0.3 * P)
    depth = np.sort(P - L)
    check("energy of curtailment = area under the count of hours within a distance of the peak (ratio)",
          cut["energy_mwh"] / float(np.maximum(0.0, 0.3 * P - depth).sum()), 1.0, 1e-12)

    say("4. The fleet-based benchmark")
    fb = fleet_based(year, units, "12CP")
    check("yearly risk of the test system as it stands, hours", fb["risk_hours"], RTS_LOLE_HOURS, 5e-6)
    check("weights add up to one", float(fb["weights"].sum()), 1.0, 1e-12)
    with_g = make_year(L, year["month"], "", variable_generation=np.full(len(L), 100.0))
    plus = units + [make_unit(100.0, 0.0)]
    a, b = fleet_based(with_g, units, "12CP"), fleet_based(year, plus, "12CP")
    check("constant variable generation of 100 MW = a unit of 100 MW without outages (benchmark demand, ratio)", a["lb"] / b["lb"], 1.0, 1e-12)
    doubled = fleet_based(year, units, "12CP", size_mw=P, rule="same-kinds")
    check("a flat load equal to the peak, same kinds: the fleet has twice the units", doubled["units"], 64, 0)
    check("with the new load the yearly risk is back at the standard (ratio)", doubled["risk_hours"] / doubled["standard_hours"], 1.0, 2e-3)
    check("and it is not above the standard", float(doubled["risk_hours"] < doubled["standard_hours"]), 1.0, 0)
    hrs, mwh = fb["fleet"].expected_cut(L - 3000.0, 3000.0)
    check("expected hours with a shortfall = yearly risk", hrs, RTS_LOLE_HOURS, 5e-6)
    check("expected energy not served of the test system, GWh a year (published: 1.176)", mwh / 1000.0, RTS_LOEE_GWH, 5e-4)
    check("energy index of reliability (published: 0.999923)", 1.0 - mwh / float(L.sum()), RTS_EIR, 5e-7)
    rng = np.random.default_rng(1)
    small = [make_unit(40, 0.1), make_unit(25, 0.2), make_unit(25, 0.05)]
    fl = Fleet(small, np.full(WEEKS * HOURS_PER_WEEK, 60.0))
    draws = sum(u["mw"] * (rng.random(400000) > u["q"]) for u in small)
    check("expected energy of curtailment against a simulation (MWh per hour)", fl.expected_cut(np.full(WEEKS * HOURS_PER_WEEK, 60.0), 30.0)[1] / (WEEKS * HOURS_PER_WEEK),
          float(np.minimum(30.0, np.maximum(0.0, 90.0 - draws)).mean()), 0.05)

    say("5. Both kinds of data: the window and the exponent that the fleet implies, and the floor of the tolerance")
    f0, f1 = fleet_based(year, rts_units(True), "12CP"), fleet_based(year, rts_units(True), "12CP", X)
    w0, w1 = equivalent_window(L, f0["lb"]) / P, equivalent_window(L, f1["lb"]) / P
    g_f = math.log(w1 / w0) / math.log(1.0 + X / P)
    check("with the window and the exponent that the fleet implies, the load-based benchmark gives the fleet-based benchmark demand (MW)",
          load_based(year, "12CP", w0, g_f, X)["lb"], f1["lb"], 1e-6)
    check("and the window of today gives the fleet-based benchmark demand of today (MW)", load_based(year, "12CP", w0)["lb"], f0["lb"], 1e-6)
    versions = [f1, load_based(year, "12CP", mu, 1.0, X), load_based(year, "12CP", mu, 0.5, X)]
    e_ = [v["eps"] for v in versions]
    check("floor of the tolerance from the benchmark shares = (1 + highest overcharge) / (1 + lowest overcharge) - 1",
          tolerance_floor([v["o_b"] for v in versions]), (1.0 + max(e_)) / (1.0 + min(e_)) - 1.0, 1e-12)
    check("versions of the benchmark that agree have a floor of zero", tolerance_floor([f1["o_b"], load_based(year, "12CP", w0, g_f, X)["o_b"]]), 0.0, 1e-8)
    rf = run_fleet_based([year], rts_units(True), "12CP", [X], grid=[1.0])
    check("the run with generating units reports that floor", rf["cases"][0]["tolerance_floor"], tolerance_floor([v["o_b"] for v in versions]), 1e-12)

    say("6. Customer classes (the test system divided into a steady class and the rest, as an illustration)")
    steady = np.full(len(L), RTS_STEADY_MW)
    yk = make_year(L, year["month"], "", classes={"steady": steady, "varying": L - steady})
    t0, t1 = load_based(yk, "12CP", mu), load_based(yk, "12CP", mu, 1.0, X)
    rows = class_rows(yk, t1, t0)
    check("weighted by the benchmark shares, the overcharges of all classes add up to zero", sum(r["share_benchmark"] * r["overcharge"] for r in rows.values()), 0.0, 1e-12)
    check("class overcharge = (1 + group overcharge) times the class factor - 1", rows["varying"]["overcharge"], (1.0 + t1["eps"]) * rows["varying"]["factor"] - 1.0, 1e-12)
    check("weighted by the parts in the benchmark demand, the class factors average to one",
          sum(r["factor"] * r["share_benchmark"] for k, r in rows.items() if k != NEW_LOAD) / t1["o_b"], 1.0, 1e-12)
    check("a steady class is overcharged exactly as the new flat load", rows["steady"]["overcharge"], rows[NEW_LOAD]["overcharge"], 1e-12)
    ys = make_year(L, year["month"], "", classes={"a": 0.3 * L, "b": 0.7 * L})
    same = class_rows(ys, load_based(ys, "12CP", mu, 1.0, X), load_based(ys, "12CP", mu))
    check("a class with the shape of the system has the class factor one", same["a"]["factor"], 1.0, 1e-12)
    check("and its overcharge is that of the existing customers as one group", same["b"]["overcharge"], t1["eps"], 1e-12)
    f_k, at, top = class_energy_share([yk], [t1], [t0], 0.02)
    check("at the energy share sized on the classes, the class that it relieves is moved by the tolerance", at["varying"]["added"], 0.02, 1e-9)
    xk = class_failing_size_load_based([yk], "12CP", mu, 0.05, 1.0)
    per = load_based(yk, "12CP", mu, 1.0, xk * P)
    check("at the failing size by class, the class that is moved most is at the tolerance", most_moved(class_table([yk], [per], [t0]))[1], 0.05, 1e-9)
    check("the failing size by class is not above that of the group", float(xk <= failing_size_load_based([yk], "12CP", mu, 0.05, 1.0)), 1.0, 0)
    fk = run_fleet_based([yk], rts_units(True), "12CP", [P], grid=[0.5, 1.0])
    bc = fk["cases"][0]["by_class"]["table"]
    check("fleet-based benchmark: a steady class is overcharged exactly as the new flat load", bc["steady"]["overcharge"], bc[NEW_LOAD]["overcharge"], 1e-12)

    say("7. Limits: the cases in which a denominator of the theory can vanish")
    ceil = overcharge_ceiling(kap, lam)
    big = load_based(year, "12CP", mu, 1.0, 1000.0 * P)
    check("a flat load of a thousand times the peak: the overcharge is still below the ceiling", float(big["eps"] < ceil), 1.0, 0)
    check("and close to it (ratio)", big["eps"] / ceil, 1.0, 0.02)
    check("safe size at a tolerance just below the ceiling: finite and above a thousand times the peak", float(1e3 < safe_size(kap, lam, ceil - 1e-6) < float("inf")), 1.0, 0)
    check("safe size at a tolerance equal to the ceiling: infinite, the existing method passes at any size", float(math.isinf(safe_size(kap, lam, ceil))), 1.0, 0)
    check("safe size at a tolerance above the ceiling: infinite as well", float(math.isinf(safe_size(kap, lam, ceil + 0.1))), 1.0, 0)
    check("with a tolerance above the ceiling there is no failing size", float(math.isinf(failing_size_load_based([year], "12CP", mu, ceil + 0.01, 1.0))), 1.0, 0)
    rng, inside = np.random.default_rng(7), True
    for _ in range(2000):      # whenever the method fails and the benchmark share is not below the energy share
        o_e_ = rng.uniform(0.05, 0.85)
        o_b_ = rng.uniform(o_e_, 0.9)
        t_ = rng.uniform(0.0, 0.1)
        o_a_ = rng.uniform((1.0 + t_) * o_b_, 1.0)
        f_ = min_energy_share(o_a_, o_b_, o_e_, t_)
        inside = inside and (0.0 <= f_ < 1.0) and abs(((1 - f_) * o_a_ + f_ * o_e_) - (1.0 + t_) * o_b_) < 1e-12
    check("the smallest energy share lies between zero and one and meets the tolerance in 2,000 random cases in which the method fails", float(inside), 1.0, 0)
    check("it is zero when the method passes, also when the shares under the method and on energy are equal", min_energy_share(0.6, 0.6, 0.6, 0.05), 0.0, 0)
    check("it is infinite when even an allocation on energy does not pass", float(math.isinf(min_energy_share(0.6, 0.5, 0.58, 0.05))), 1.0, 0)
    check("for a very large load it approaches 1 - tolerance / ceiling", min_energy_share(big["o_a"], big["o_b"], big["o_e"], 0.05), 1.0 - 0.05 / ceil, 0.01)
    check("a window of zero puts the whole weight on the hour of the peak (benchmark demand over peak)", benchmark_demand(L, exp_weights(L, 0.0)) / P, 1.0, 1e-12)
    flat = make_year(np.full(len(L), 1000.0), year["month"])
    check("a constant existing load: the ceiling is zero", overcharge_ceiling(1.0, flat["A"] / flat["P"]), 0.0, 1e-12)
    check("and no flat load is overcharged against", load_based(flat, "12CP", 0.04, 1.0, 5000.0)["eps"], 0.0, 1e-12)
    try:
        fleet_based(year, [make_unit(400.0, 0.0) for _ in range(10)], "12CP")
        refused = 0.0
    except ValueError:
        refused = 1.0
    check("a fleet that can never be short is refused with a message", refused, 1.0, 0)
    night = np.where(np.arange(len(L)) % 24 < 5, 50.0, 0.0)
    yn = make_year(L, year["month"], "", classes={"night": night, "rest": L - night})
    tn0, tn1 = load_based(yn, "12CP", mu), load_based(yn, "12CP", mu, 1.0, X)
    rn = class_rows(yn, tn1, tn0)
    check("a class without load in the named hours is reported as not counted", float(rn["night"]["added"] is None and rn["night"]["share_method"] == 0.0), 1.0, 0)
    check("and the class that is moved up most is found among the others", float(most_moved(class_table([yn], [tn1], [tn0]))[0] == "rest"), 1.0, 0)
    say("all checks hold" if all(ok) else "A CHECK FAILED")
    return all(ok)


# ======================================================================================================================
# 9. Command line
# ======================================================================================================================
def main(argv=None):
    """The command line. A fault in an input file ends with one line that says what is wrong."""
    try:
        return _main(argv)
    except (ValueError, OSError) as err:
        print(f"cptest: {err}", file=sys.stderr)
        return 2


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="cptest", description="Test of a coincident peak allocation against hourly loss-of-load risk when a large load is added.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="check the tool against published values and identities")
    p = sub.add_parser("rts-data", help="write the IEEE Reliability Test System as input files")
    p.add_argument("out_dir")
    for name, text in (("load", "test from the hourly load alone"), ("fleet", "test from the hourly load and the list of generating units")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--load", required=True, help="CSV file with the hourly load of the existing customers (column load_mw, or class columns)")
        p.add_argument("--method", default="12CP", help="1CP, 4CP, 12CP, kCP or months:6,7,8,12 (default 12CP)")
        p.add_argument("--size", type=float, nargs="+", default=[0.0], help="megawatts of flat new load to test (one or more)")
        p.add_argument("--tolerance", type=float, default=0.05, help="tolerance for the overcharge (default 0.05)")
        p.add_argument("--hour-ending", action="store_true", help="time stamps mark the end of the hour")
        p.add_argument("--new-load-column", help="column of the load file with the hourly profile of a new load that is not flat")
        p.add_argument("--class-columns", nargs="+", default=None, help="columns of the load file with the hourly MW of each customer class")
        p.add_argument("--scale", type=float, nargs="+", default=None, help="multiples of that profile to test (default 1)")
        p.add_argument("--headroom", type=float, default=0.0, help="room above the firm peak that curtailable megawatts may use, as a share of the existing peak (default 0)")
        p.add_argument("--json", help="write the results to this file as well")
        p.add_argument("--floor", type=float, default=0.04, help="lowest window share of the load-based benchmark when the window is set from the method (default 0.04)")
        p.add_argument("--growth", type=float, nargs="+", default=[1.0, 0.5], help="exponents with which the window of the load-based benchmark follows the system peak (default 1 and 0.5)")
        if name == "load":
            p.add_argument("--window", default="auto", help="'auto' (implied by the method, not below the floor) or a share of the existing peak such as 0.05")
        else:
            p.add_argument("--units", required=True, help="CSV file with the generating units of today")
            p.add_argument("--units-after", help="CSV file with the generating units planned for the system with the new load")
            p.add_argument("--expansion", default="same-kinds", choices=["same-kinds", "larger-units"], help="how the fleet grows if --units-after is not given")
            p.add_argument("--target-hours", type=float, default=None, help="standard in hours of loss of load a year (default: the risk of the system as it stands)")
            p.add_argument("--step", type=float, default=1.0, help="grid of the outage table in MW (default 1)")
    a = ap.parse_args(argv)
    if a.command == "selftest":
        return 0 if selftest() else 1
    if a.command == "rts-data":
        print("written:", write_rts_files(a.out_dir))
        return 0
    years = read_load_csv(a.load, a.hour_ending, a.new_load_column, a.class_columns)
    scales = (a.scale or [1.0]) if a.new_load_column else [0.0]
    if a.command == "load":
        res = run_load_based(years, a.method, a.size, a.tolerance, a.window, a.floor, a.growth, scales, a.headroom)
    else:
        after = read_units_csv(a.units_after) if a.units_after else None
        if after is not None and (len(a.size) > 1 or len(scales) > 1):
            ap.error("with --units-after give one size: the planned fleet belongs to one size of new load")
        res = run_fleet_based(years, read_units_csv(a.units), a.method, a.size, a.tolerance, a.expansion, after, a.target_hours, a.step, scales, None, a.headroom, a.floor, a.growth)
    print(report(res))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(_json_ready(res), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

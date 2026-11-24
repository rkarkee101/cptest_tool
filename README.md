# cptest: does a coincident peak allocation still hold when a large load is added?

This Python code belongs to the paper "When does a coincident peak allocation need an energy component for a large flat load?". It computes the test of that paper for any system, from the data that the user holds, and it is ready to use: one file, no installation beyond numpy. Whoever has the hourly load of a system uses the mode `load`; every utility records that load, and a commission or an intervenor can request it. A utility that also has the list of its generating units uses the mode `fleet`. If the load file holds the hourly load of the customer classes, both modes give the result class by class. The symbols and the equation numbers below are those of the paper. This is version 1.3. The paper and its Supplementary Information call this code "the tool", and so does this file.

The tool is one file, `cptest.py`. It needs Python 3.9 or later and the package numpy, and nothing else.

## What the test does

The existing method is the coincident peak method that the commission has approved for the system: 1CP, 4CP, 12CP, or the peaks of named months. It counts every group of customers at its average load in the hours that it names. The benchmark is the allocation by hourly loss-of-load probability: it counts every group at its hourly load weighted by the share of each hour in the yearly risk. The benchmark is a yardstick; it divides no cost. The overcharge of the existing customers is the relative excess of their share of the cost pool under the existing method over their share under the benchmark, equation (9). The existing method passes for a new load if the overcharge is not above a tolerance, which is 5% unless another value is given.

For each size of new load the tool reports the overcharge, the failing size (the largest flat load that the existing method carries, equation (25)), the smallest share of the pool that has to be allocated on energy (equation (30)), and the megawatts that have to be curtailable if the existing method is kept (equation (32)), with the hours and the energy of curtailment (equation (33)). It also reports the ceiling of the overcharge and the safe size of equation (12), below which the existing method passes whatever the benchmark.

The ceiling of the overcharge is kappa_a / lambda - 1: the count of the existing customers under the existing method, divided by their load factor, less one. No flat load of any size can shift more than this to the existing customers, because the benchmark cannot move further than an allocation on energy. If the tolerance is at the ceiling or above it, the existing method passes for a flat load of any size, and the tool says so and prints no safe size and no failing size. If the tolerance is below the ceiling, the safe size is finite; its denominator is the distance of the tolerance from the ceiling, so it grows without limit as the tolerance approaches the ceiling. The section "Limits of the expressions" below lists every such case.

The energy share replaces a part of the existing method. It is not a charge on top of it, and the pool keeps its size: with a share f, every customer, the new load included, bears 1 - f of its share under the existing method plus f of its share of the energy. A customer with steady load, such as the new flat load, pays more than before, and a class with peaked load pays less.

A user who holds both kinds of data, the hourly load and the list of generating units, needs neither premise of the load-based benchmark. The mode `fleet` then reports for each size of new load the window at which the load-based benchmark gives the count of the fleet-based one, equation (26), and the exponent with which that window has grown from the window of today, equation (27). With these two numbers the load-based benchmark has no free parameter left. It also reports the floor of the tolerance, equation (28): the largest relative difference between the shares that several versions of the benchmark give the existing customers. A tolerance below that floor would fail an allocation that agrees with one version of the benchmark, only because it is measured against another.

The existing customers are tested as one group. With class columns in the load file the tool also applies the class factor of equation (36): the overcharge of a class is that of the group times the part of the class among the existing customers under the existing method, divided by its part under the benchmark. What the new load adds to the overcharge of a class, equation (37), is compared with the tolerance, and the verdict and the remedies are read for the class that the new load moves up most.

## Quick start

The commands below check the installation, write the IEEE Reliability Test System as input files, repeat the two tests of the paper for that system, and show the class columns.

```
python cptest.py selftest
python cptest.py rts-data examples
python cptest.py load  --load examples/rts_load.csv --method 12CP --size 1425 2850 4275
python cptest.py fleet --load examples/rts_load.csv --units examples/rts_units.csv --method 12CP --size 1425 2850 4275
python cptest.py load  --load examples/rts_classes.csv --class-columns steady_mw varying_mw --method 12CP --size 1425 2850
```

The folder `examples` holds the input files `rts_load.csv`, `rts_units.csv` and `rts_classes.csv`, and the output of the last three commands, `output_load.txt`, `output_fleet.txt` and `output_classes.txt`. The output of the mode `fleet` is the calculation that Table S3 of the Supplementary Information follows step by step. The test system has no customer classes. The file `rts_classes.csv` divides its load into a steady class of 600 MW and the rest, only to show how class columns work; this division is not part of the published system and no result of the paper rests on it.

## Input files

Both files are text files with comma-separated values and a header line. Columns that the tool does not know are ignored.

The load file has one row for every hour.

| Column | Required | Meaning |
|---|---|---|
| load_mw | yes, unless class columns are named | Load of the existing customers in that hour, in megawatts. This is the load without the new load that is to be tested. |
| columns of your choice | no | Hourly megawatts of each customer class. Name them with `--class-columns`. Without a column load_mw the load is the sum of the classes. With it, what the classes leave of the load is carried as a further class called `other`; the classes may not add up to more than the load. |
| timestamp | no | Date and time of the hour, such as 2025-07-14 16:00. It gives the year and the month of the hour. With the option `--hour-ending` the time stamp marks the end of the hour. Hours of 29 February are dropped. |
| year, month | no | Used when there is no time stamp. The month is a number from 1 to 12. |
| variable_generation_mw | no | Output of wind, solar and other plant that is not dispatched, in megawatts. It is taken off the load before the risk of an hour is formed. The hours that a method names are always found on the full load. |
| a column of your choice | no | Hourly megawatts of a new load that is not flat. Name the column with `--new-load-column`. |

Without time stamp, year and month, the file is read as one year that starts on 1 January and has no 29 February. Several years can be given in one file. The window is then set once for all years, and every result is the median over the years, as in the paper. A year with fewer than 8,000 hours is left out. The mode `fleet` needs at least 8,736 hours in every year, which is 52 weeks of 168 hours. A fault in an input file ends the run with one line that says what is wrong.

The units file, for the mode `fleet`, has one row for every generating unit or for every group of equal units.

| Column | Required | Meaning |
|---|---|---|
| capacity_mw | yes | Capacity of the unit in megawatts. |
| forced_outage_rate | yes | Probability that the unit is unavailable when it is called, between 0 and 1. |
| maintenance_weeks | no | Weeks of planned maintenance a year, in one run of consecutive weeks. Zero if the column is missing. |
| maintenance_start_week | no | Week (1 to 52) in which the maintenance of the unit starts. If it is empty the tool places the maintenance. |
| count | no | Number of equal units that the row stands for. One if the column is missing. |
| name | no | A label. |

## The mode `load`

```
python cptest.py load --load load.csv --method 12CP --size 500 1000
```

| Option | Default | Meaning |
|---|---|---|
| --method | 12CP | The existing method: 1CP, 4CP, 12CP, any kCP with k from 2 to 12 (the k highest monthly peaks), or months:6,7,8,12 for the peaks of named months. |
| --size | 0 | Megawatts of flat new load. Several values can be given. For flat loads that arrive one after the other, give the total that has joined since the existing method was approved. |
| --tolerance | 0.05 | The tolerance for the overcharge. |
| --window | auto | `auto` sets the window of the risk curve from the existing method itself: the window at which the benchmark agrees with it on the system of today, equation (21), but not below the floor. A number such as 0.05 sets the window as a share of the existing peak. Use a number when a reliability study or the mode `fleet` has given one. |
| --floor | 0.04 | Lowest window share that `auto` uses, equation (22). |
| --growth | 1 0.5 | Exponents with which the window follows the growth of the system peak, equation (23). The paper reports 1 and 0.5 as the two reference cases, and the result should be read as the range between them. |
| --new-load-column | none | Column of the load file with the hourly megawatts of a new load that is not flat. The flat megawatts of `--size` are added to it. |
| --class-columns | none | Columns of the load file with the hourly megawatts of the customer classes. |
| --scale | 1 | Multiples of that hourly profile to test. |
| --headroom | 0 | Room above the firm peak that the curtailable megawatts may use, as a share of the existing peak. Zero is the rule of equation (33). |
| --hour-ending | off | The time stamps mark the end of the hour. |
| --json | none | A file to which the results are written as well. |

## The mode `fleet`

```
python cptest.py fleet --load load.csv --units units.csv --method 12CP --size 500 1000
```

The options `--method`, `--size`, `--tolerance`, `--new-load-column`, `--class-columns`, `--scale`, `--headroom`, `--hour-ending` and `--json` are as above. The options `--floor` and `--growth` are accepted as well: they set the window and the exponents of the load-based benchmark from the load alone, which is one of the versions that enter the floor of the tolerance. The others are these.

| Option | Default | Meaning |
|---|---|---|
| --units | required | The generating units of today. |
| --units-after | none | The generating units that are planned for the system with the new load. This is the option for a utility with a resource plan. It belongs to one size of new load, so give one value of `--size`. Without it the tool enlarges the fleet by one of the two rules below. |
| --expansion | same-kinds | `same-kinds` adds units of the kinds already there. `larger-units` makes every unit larger. |
| --target-hours | none | The standard in hours of loss of load a year, for example 2.4. Without it the standard is the yearly risk of the system as it stands. |
| --step | 1 | Grid of the outage table in megawatts. A larger step makes large systems faster. |

The outage table is the capacity outage probability table of reliability analysis: the probability distribution of the capacity that is out of service, as a list of every amount of capacity that can be out together, each with its probability. The tool computes it from the units file.

## What the output says

The report states first how the system and the existing method stand today: the existing peak, the load factor, the share of the peak at which the existing method counts the existing load (kappa_a), the ceiling of the overcharge, and the safe size. In the mode `load` it gives the window that the existing method implies and the window that is used. In the mode `fleet` it gives the yearly risk that is held as the standard, the share of the peak at which the benchmark counts the existing load today (kappa*), and the window at which the mode `load` would give the same count. That window can be passed to the mode `load` with `--window`.

For each size it then gives the share of the pool that the existing customers bear under the existing method and under the benchmark, the overcharge, and whether the existing method passes. If it fails, the first remedy is the smallest energy share and the second is the curtailable megawatts. The hours and the energy share that follow are those of the rule of equation (33): the curtailable megawatts are served only from the room that the firm load leaves below the firm peak. The mode `fleet` adds the expected hours and the expected share of the energy that is cut, from the outage table: the megawatts are cut whenever the available capacity is below the firm load plus the curtailable megawatts.

A negative overcharge means that the existing method charges the new load more than the benchmark does.

In the mode `fleet` every size of new load is followed by a line with the window at which the mode `load` gives the same benchmark demand, and with the exponent with which that window has grown from the window of today. These are the two numbers that the mode `load` has to assume, the window of equation (22) and the exponent of equation (23). A utility that publishes them with its result lets others repeat the test with `--window` and `--growth` from the hourly load alone.

The next line gives the floor of the tolerance. Three versions of the benchmark enter it: the fleet-based benchmark as it was computed, and the load-based benchmark from the load alone, with the window that the existing method implies and each of the exponents of `--growth`. The line states the share of the pool that each version gives the existing customers, and the floor is the largest of these shares divided by the smallest, less one. If the tolerance is below the floor, the verdict can depend on the version of the benchmark, and the version should be settled first. The floor is written to the file of `--json` as `tolerance_floor`. A utility that has several resource plans can run the mode `fleet` once for each and compare the shares in the same way.

With class columns a table follows for every size. It has one row for every class and one for the new load, and four columns: the share of the pool under the existing method, the share under the benchmark, the gap of today (the class factor of equation (36) on the system without the new load, minus one), and what the new load adds, equation (37). The line below the table names the class that is moved up most and says whether it passes, and gives the failing size by class, which is the flat load at which that class reaches the tolerance. If the existing method fails by class, the first remedy is the smallest energy share of the whole pool at which no class that an energy share relieves is moved up by more than the tolerance. An energy share relieves the classes whose share of the energy is below their share under the existing method, and it loads the others; if one of those then stands above the tolerance, the line says which. The second remedy is the megawatts above the failing size by class, with the hours and the energy of curtailment under the rule of equation (33). The class results are written to the file of `--json` under `by_class`.

To test the growth of an existing class, name its column as the new load as well: `--class-columns res_mw com_mw ind_mw --new-load-column ind_mw --scale 0.5` tests a growth of the class `ind_mw` by one half. The megawatts that it adds are then a new load with the shape of the class.

## Limits of the expressions

Several expressions of the paper have a denominator that can become zero. None of them is used at such a point: the existing method then passes, or the quantity is not needed, or another form of the same expression applies. Section S3 of the Supplementary Information goes through all of them. The tool reports each case in words and prints no number for it.

| Case | What the tool reports |
|---|---|
| The tolerance is at the ceiling of the overcharge or above it, so that the denominator of the safe size, equation (12), is zero or negative | "safe size: none is needed", with the statement that the existing method passes for a flat load of any size. The line of the failing size says that there is none, and no remedy is printed. In the file of `--json` the safe size and the failing size are then empty (null). |
| The overcharge stays within the tolerance at every size, because the benchmark does not move far enough | "no failing size", or in the mode `fleet` "failing size: none on the grid of sizes", with the largest size that was computed. |
| The existing method passes, which includes a new load of zero megawatts and an existing method that counts the existing customers at their average load, the two cases in which the denominator of the energy share, equation (30), is zero | No energy share is printed, and `min_energy_share` returns zero. When the existing method fails, the denominator exceeds the numerator, and the share lies between zero and one. |
| The benchmark counts the existing customers below their share of the energy by more than the tolerance | "remedy 1: no share of the pool on energy suffices", and `min_energy_share` returns infinity. No case of the paper is of this kind. |
| A window of zero, which is the implied window of 1CP | The whole weight is put on the hour of the peak. With `--window auto` the floor of `--floor` applies. A window of zero that is set by hand stays zero at every size; the benchmark is then the hour of the peak at every size, no flat load can fail, and the report says so. |
| A fleet that can never be short of the load | The run ends with one line: the yearly risk of the system is zero, so there is no risk to divide. |
| A window of the fleet that is zero or infinite, equation (26), or a new load of zero megawatts, for which the logarithm in the denominator of equation (27) is zero | No exponent is printed. |
| A class with no load in the hours that the existing method names today | The class is listed as "not counted by the existing method today" and is left out of the verdict, because what the new load adds to it, equation (37), is a ratio to zero. |
| A class with no load in any hour that carries risk | The run ends with one line that names the class. |

## Rules that are built in

The months of a year decide which hours a method names, so the months in the load file should be calendar months in local time.

In the mode `fleet` every unit has two states, available or on forced outage, and the outages of the units are independent. The distribution of the available capacity is built by adding one unit at a time on the grid of `--step` megawatts. A load within a billionth of a megawatt above a grid point counts as that grid point. A week is a block of 168 hours from the first hour of the file, and the last week takes the hours that are left over. Maintenance that has no start week is placed one unit at a time, the largest unit first and equal units in the order of the file, each into the run of consecutive weeks in which the highest value of weekly peak load plus capacity already on maintenance is lowest; of several such runs the earliest is taken.

With a new load the fleet is enlarged for a peak that is larger by the factor by which the system peak grows. Under `same-kinds`, units are added one at a time, each time a unit of the kind whose capacity is furthest below that factor times its capacity of today, until one more unit would take the installed capacity further from the target than it is. Under `larger-units` the capacity of every unit is multiplied by the factor and rounded to the grid, with halves rounded to the even grid point. In both cases, and also with `--units-after`, a block of capacity without outages, positive or negative, is then added so that the yearly risk returns to the standard. The yearly risk falls in steps as the block grows, so the standard is in general not met exactly: the block is the smallest one at which the yearly risk is below the standard. Its size is written for every size of new load to the file of `--json`, as `firm_block_mw`. A large block means that the fleet that was given does not meet the standard by itself.

The failing size of the mode `fleet` is found on a grid of sizes from 5% to 200% of the existing peak in steps of 5%, as the first crossing of the tolerance, interpolated between two points of the grid. The failing size of the mode `load` is searched up to ten thousand times the existing peak.

## Use from Python

The file can be imported. The functions that the scripts of the paper use are these.

| Function | What it returns |
|---|---|
| `make_year(load, month, label, variable_generation, new_load, classes)` | A year of hourly data. `classes` is a dictionary with the hourly load of every class. |
| `read_load_csv(path, hour_ending, new_load_column, class_columns)` and `read_units_csv(path)` | The years of a load file, and the units of a units file. |
| `run_load_based(years, method, sizes_mw, tau, window, floor, gammas, scales, headroom)` | The whole test from the hourly load. |
| `run_fleet_based(years, units, method, sizes_mw, tau, rule, units_after, target_hours, step, scales, grid, headroom, floor, gammas)` | The whole test with the list of units, with the window, the exponent and the floor of the tolerance for each size. |
| `report(result)` | The result as text. |
| `load_based(year, method, mu, gamma, size_mw, scale)` and `fleet_based(year, units, method, size_mw, scale, rule, ...)` | Demands, shares and overcharge of one year for one new load. |
| `implied_window_share(years, method)`, `failing_size_load_based(...)`, `overcharge_ceiling(kappa_a, load_factor)`, `safe_size(kappa_a, load_factor, tau)`, `min_energy_share(o_a, o_b, o_e, tau)`, `curtailment(load, curtailable_mw, headroom_mw)`, `equivalent_window(load, demand)`, `tolerance_floor(shares)` | The single quantities of the paper. |
| `class_table(years, cases, today, f)`, `most_moved(table)`, `class_energy_share(years, cases, today, tau)`, `class_failing_size_load_based(years, method, mu, tau, gamma)` | The test by class: the table of the classes for the cases of `load_based` or `fleet_based` with and without the new load and an energy share `f`, the class that is moved up most, the energy share sized on the classes, and the failing size by class. |
| `outage_table(units, step)`, `schedule_maintenance(units, weekly_peak)`, `expand_fleet(units, factor, rule, step)` | The parts of the fleet-based benchmark. |
| `rts_units(maintenance)`, `rts_load(peak)`, `rts_year(peak)` | The IEEE Reliability Test System. |

## The self-test

`python cptest.py selftest` runs 75 checks and ends with the line "all checks hold". The first group compares the outage table with the values that Billinton and Allan (1996, Appendix 2) print for the IEEE Reliability Test System without maintenance: the loss-of-load expectation of 1.36886 days a year with daily peaks and of 9.39418 hours a year with hourly loads, the eight values of their Table A2.9, the nine cumulative probabilities of their Table A2.5 to six decimals, and the energy indices of 1.176 GWh a year and 0.999923. The second group checks a system of three hours that can be followed by hand. The third and fourth groups check the identities of the paper, among them that the overcharge at the smallest energy share equals the tolerance and that a new load with the shape of the existing load causes no overcharge, and compare the expected energy of curtailment with a simulation. The fifth group checks what a user with both kinds of data gets: with the window and the exponent that the fleet shows, the load-based benchmark returns the benchmark demand of the fleet-based one, and the floor of the tolerance is the same whether it is formed from the shares or from the overcharges. The sixth group checks the test by class on a system of two classes: the overcharges of all classes add up to zero when weighted by the benchmark shares, a class with the shape of the system has the class factor one, and a steady class is overcharged exactly as the new flat load, under both benchmarks. The seventh group checks the limits of the expressions: a flat load of a thousand times the peak stays below the ceiling of the overcharge; the safe size is finite just below the ceiling and infinite at it and above it; with a tolerance above the ceiling there is no failing size; the smallest energy share lies between zero and one and meets the tolerance in 2,000 random cases in which the method fails, and is zero when it passes; a window of zero puts the whole weight on the hour of the peak; a constant existing load has a ceiling of zero; a fleet that can never be short is refused; and a class without load in the named hours is reported and left out of the verdict.

## Limits

The load-based benchmark rests on two premises that the paper states: the window is set from the existing method, and it grows with the system by a stated exponent. Its results are a range, not a point. The fleet-based benchmark models units with two states and independent outages. It has no derated states, no limits on energy, no storage, no imports and no network. Variable generation enters only as an hourly series that is taken off the load. The two rules for the growth of the fleet are simple brackets, and a utility should give its own planned fleet with `--units-after`. The tolerance is a judgment that belongs to the commission; the floor that the tool reports is the distance between the versions of the benchmark that it computed, not a recommendation. The test by class reads the verdict on what the new load adds and reports the gap of today beside it; a small class with unusual hours can show a large gap of today, and a class with no load in the hours that the existing method names is reported and left out of the verdict. The test divides a shared cost pool and says nothing about plant that is built for one customer.



## Reference

Billinton, R., Allan, R.N., 1996. Reliability Evaluation of Power Systems, second ed. Plenum Press, New York. https://doi.org/10.1007/978-1-4899-1860-4

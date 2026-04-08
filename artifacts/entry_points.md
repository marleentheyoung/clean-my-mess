# Entry Points and Call Chains

## 1. Main Solve Path: `solve_epsilons.py:main()` (line 48)

The `main()` function spans lines 48-2725, but approximately 95% is dead code.
Lines 48-107 are live; everything from line 108 onward is either inside a
triple-quoted block comment (lines 108-1039), followed by more dead plotting
code (lines 1039-1040 is another `"""` pair wrapping lines 1040-1039), and then
the remaining ~1600 lines reference undefined variables (`dPi_L`, `iNj`,
`results_vector`, `dPi_S`, `vt_buy_c`, `vt_buy_nc`, `a_index`, `alpha`, `err`,
`Line2D`) that would crash at runtime.

**Live code (lines 48-107):**

```
solve_epsilons.py:main()                                       # line 48
  |-- misc.construct_jitclass(parfile.par_dict)                # line 67 -> misc_functions.py:construct_jitclass()
  |-- tauch.tauchen(...)                                       # line 76 -> tauchen.py:tauchen()
  |-- grid_creation.create(par)                                # line 77 -> grid_creation.py:create()
  |     |-- tauch.tauchen(...)                                 #   -> tauchen.py:tauchen()
  |     |-- tauch.initial_dist(...)                            #   -> tauchen.py:initial_dist()
  |     |-- tauch.weight_matrix(...)                           #   -> tauchen.py:weight_matrix()
  |     |-- tauch.lifecycle(...)                               #   -> tauchen.py:lifecycle()
  |     |-- tauch.median_inc(...)                              #   -> tauchen.py:median_inc()
  |     |-- grid.nonlinspace_jit(...)                          #   -> grids.py:nonlinspace_jit()
  |     |-- misc.construct_jitclass(grids_dict)                #   -> misc_functions.py:construct_jitclass()
  |-- welfare_stats.find_expenditure_equiv(...)                # line 103 -> proper_welfare_debug.py:find_expenditure_equiv()
  |     |-- household_problem.solve_ss(...)                    #   -> household_problem_epsilons_nolearning.py:solve_ss()
  |     |-- sim.stat_dist_finder(...)                          #   -> simulation.py:stat_dist_finder()
  |     |-- household_problem.solve(...)                       #   -> household_problem_epsilons_nolearning.py:solve()
  |     |-- equil.generate_pricepath(...)                      #   -> equilibrium.py:generate_pricepath()
  |-- print(tax_equiv_newborns)                                # lines 104-107
```

**Dead code starts at line 108** with a `"""` opening the first triple-quoted
block comment, which closes at line 1039. Line 1040 opens a second `"""` block
that closes on line 2725. Together, lines 108-2725 are two consecutive
triple-quoted string literals containing ~2,617 lines of dead code that Python
treats as no-op expressions.

Note: line 76 calls `tauch.tauchen()` but its return value `mMarkov` is
immediately overwritten by `grid_creation.create()` on line 77, making the line
76 call redundant.


## 2. Calibration Path: `full_calibration.py:f()` (line 41)

```
full_calibration.py:f(x, grad)                                 # line 41
  |-- misc.construct_jitclass(par_dict)                        # line 134 -> misc_functions.py:construct_jitclass()
  |-- tauch.tauchen(...)                                       # line 135 -> tauchen.py:tauchen()
  |-- grid_creation.create(par)                                # line 136 -> grid_creation.py:create()
  |     |-- (same subtree as above)
  |-- equil.initialise_coefficients_initial(...)               # line 138 -> equilibrium.py [NOTE: this function name does not exist; only initialise_coefficients_ss() exists -- will crash at runtime]
  |     |-- household_problem.solve_ss(...)                    #   -> household_problem_epsilons_nolearning.py:solve_ss()
  |     |     |-- continuation_value.solve_last_period_*()     #     -> continuation_value_nolearning.py
  |     |     |-- stayer_problem.solve(...)                    #     -> stayer_problem.py:solve()
  |     |     |-- stayer_problem_renter.solve(...)             #     -> stayer_problem_renter.py:solve()
  |     |     |-- buyer_problem_epsilons.solve(...)            #     -> buyer_problem_epsilons.py:solve()
  |     |-- sim.stat_dist_finder(...)                          #   -> simulation.py:stat_dist_finder()
  |     |     |-- sim.update_dist_continuous(...)              #     -> simulation.py:update_dist_continuous()
  |     |     |     |-- mortgage_sim.solve(...)                #       -> mortgage_choice_simulation.py:solve()
  |     |     |     |-- mortgage_sim_exc.solve(...)            #       -> mortgage_choice_simulation_exc.py:solve()
  |     |     |     |-- buy_sim.solve(...)                     #       -> buyer_problem_simulation.py:solve()
  |     |     |     |-- sim.continuous_decide(...)             #       -> simulation.py:continuous_decide()
  |     |     |     |-- sim.simulate_stay(...)                 #       -> simulation.py:simulate_stay()
  |     |     |     |-- sim.simulate_buy_outer(...)            #       -> simulation.py:simulate_buy_outer()
  |     |     |     |-- sim.simulate_rent_outer(...)           #       -> simulation.py:simulate_rent_outer()
  |     |     |     |-- initial_joint_sim.initial_joint(...)   #       -> simulate_initial_joint.py:initial_joint()
  |     |-- equil.house_prices_algorithm(...)                  #   -> equilibrium.py:house_prices_algorithm()
  |     |     |-- equil.precompute_market_data(...)            #     -> equilibrium.py:precompute_market_data()
  |     |     |-- sim.excess_demand_continuous(...)            #     -> simulation.py:excess_demand_continuous()
  |     |     |-- equil.secant_method_system_2d(...)           #     -> equilibrium.py:secant_method_system_2d()
  |-- household_problem.solve_initial(...)                     # line 139 -> [NOTE: solve_initial does not exist in household_problem_epsilons_nolearning.py; only solve() and solve_ss()]
  |-- sim.stat_dist_finder(...)                                # line 140 -> simulation.py:stat_dist_finder()
  |-- lom.LoM_C(...)                                           # line 141 -> LoM_epsilons.py:LoM_C()
  |-- lom.LoM_NC(...)                                          # line 142 -> LoM_epsilons.py:LoM_NC()
  |-- find_moments.calc_moments(...)                           # line 145 -> moments.py:calc_moments()
  |-- (compute squared residuals and return)                   # lines 146-176
```

`full_calibration.py:main()` (line 178) wraps `f()` in an NLopt optimizer:

```
full_calibration.py:main()                                     # line 178
  |-- nlopt.opt(nlopt.G_MLSL_LDS, 7)                          # line 188
  |-- opt.optimize([...]) -> calls f(x, grad) repeatedly      # line 202
```


## 3. Experiment Path: `experiments.py`

```
experiments.py:full_information_experiment(...)                 # line 44
  |-- grid_creation.create(par)                                # line 46 -> grid_creation.py:create()
  |-- gen_distribution_now(...)                                # line 48 -> experiments.py:gen_distribution_now() (line 33)
  |     |-- household_problem.solve_ss(...)                    #   -> household_problem_epsilons_nolearning.py:solve_ss()
  |     |-- sim.stat_dist_finder(...)                          #   -> simulation.py:stat_dist_finder()
  |     |-- equil.generate_pricepath(...)                      #   -> equilibrium.py:generate_pricepath()
  |     |     |-- household_problem.solve(...)                 #     -> household_problem_epsilons_nolearning.py:solve()
  |     |     |-- equil.house_prices_algorithm(...)            #     -> equilibrium.py:house_prices_algorithm()
  |     |     |-- sim.update_dist_continuous(...)              #     -> simulation.py:update_dist_continuous()
  |-- grid_creation.create(par, experiment=True)               # line 49 -> grid_creation.py:create()
  |-- full_information_shock(...)                              # line 50 -> experiments.py:full_information_shock() (line 13)
        |-- equil.find_coefficients(...)                       #   -> equilibrium.py:find_coefficients()
              |-- equil.generate_pricepath(...)                #     -> equilibrium.py:generate_pricepath()
              |-- equil.coeff_updater(...)                     #     -> equilibrium.py:coeff_updater()
              |     |-- misc.ols_numba(...)                    #       -> misc_functions.py:ols_numba()
```

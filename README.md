# Mini_codes

Short pricing scripts that do not deserve a repository on their own.

The option pricer grew into its own repository:
[option-pricing-lab](https://github.com/Luca-CS/option-pricing-lab) (American, Asian and barrier options
under Heston, Dupire local volatility and a calibrated stochastic-local-volatility model, with a
[browser demo](https://luca-cs.github.io/option-pricing-lab/)).

| Script | What it does |
|---|---|
| [`Python/montecarlo_pricer_v1.py`](Python/montecarlo_pricer_v1.py) | European option priced by Monte Carlo under Black-Scholes, with two variance-reduction methods: antithetic variates, and a control variate on the terminal price S_T (known mean and variance, coefficient estimated on pilot paths). Prints each estimate with its 95 % interval next to the closed-form Black-Scholes price, and plots the paths and the distribution of S_T. |
| [`Python/Hull_and_white_rate_options_pricer.py`](Python/Hull_and_white_rate_options_pricer.py) | Tkinter app that prices interest-rate options (European or Bermudan, call or put, optional barrier) by backward induction on a short-rate tree. **Prototype**, see the limitations below. |

## Running

```bash
pip install numpy matplotlib pandas-datareader
python Python/montecarlo_pricer_v1.py
python Python/Hull_and_white_rate_options_pricer.py
```

With 200,000 paths, both Monte Carlo estimators land within one standard error of the Black-Scholes price
(call: 0.3544, put: 10.0797 for S = 100, K = 110, r = 3 %, σ = 25 %, T = 1 month).

## Limitations of the short-rate prototype

- The tree is a symmetric binomial tree on the short rate with probability 1/2. The mean-reversion speed `a` is
  read but not used yet, and the tree is not fitted to the initial term structure. A real Hull-White
  implementation needs a trinomial tree with mean reversion, calibrated to the zero curve.
- Knock-in barriers are checked node by node, which does not capture the path dependence of a knock-in.
  Knock-out barriers are monitored only at the tree dates.
- The "US LIBOR" rate comes from the FRED series `USDONTD156N`. LIBOR was discontinued in 2023, so the script
  falls back to a default rate when the download fails.

## License

[MIT](LICENSE)

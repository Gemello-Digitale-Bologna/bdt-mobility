# TEST TS_SOLVE

## Test 1: compare deterministic and adaptive models (`test_ts_solve_alternatives`)

> _"Beware of bugs in the above code; I have only proved it correct, not tried it."_ - Donald Knuth

We claim that the two models give the same result, up to the `dwell_time` ($\textit{dwt}$) parameter.

More precisely, the **deterministic** model computes function:

$$ s[t] = \sum_{\delta \ge 0} \textit{ts}[t-i] e^{-i/\tau_D}$$

where $\textit{ts}$ is the input time series, $s$ is the computed time series, and 
$\tau_D = \textit{dwt}_D \times \textit{frequency}$ is the dwell time adjusted for the frequency.

The **adaptive** model computes function:

$$ s[t] = \sum_{\delta \ge 0} \textit{ts}[t-i] \left(\frac{\tau_A-1}{\tau_A}\right)^i$$

where $\textit{ts}$ is the input time series, $s$ is the computed time series, and 
$\tau_A = \textit{dwt}_A \times \textit{frequency}$ is the dwell time adjusted for the frequency.
   dw_d = 1/(frequency * math.log(dwell_time*frequency/(dwell_time*frequency-1)))

Now, if we set 
$$\textit{dwt}_D = \left(\log\frac{\textit{dwt}_A}{\textit{dwt}_A-1}\right)^{-1}$$ 
we get the same result.

Note the more complex formula in the code, where we take into account the frequency:
````    
dwt_d = 1/(frequency * math.log(dwt_a*frequency/(dwt_a*frequency-1)))
````

Also note that the cose set cutoff / termination parameters to very conservative values, 
i.e., lower performance and higher precision.

**The results of Test 1 confirm our claim.**

## Test 2: evaluate the parameters of the deterministic model (`test_ts_solve_deterministic_parameters`)

This test compares the output of the deterministic model with different cutoff probabilities (`prob_threshold`). 

The result is that the output improves, in terms of similarity to the 0-cutoff case, 
when the cutoff probability is decreased. Good outputs are produced for `prob_threshold` values below 0.0001.

## Test 3: evaluate the performance of the deterministic model (`test_ts_solve_deterministic_performance`)

This test compares the performance of the deterministic model with different cutoff probabilities (`prob_threshold`), 
including the 0-cutoff case.

No significant performance difference is observed for the different cutoff probabilities.

**Given the results of Test 2 and 3, we can conclude that the 0-cutoff case is the best one.**

## Test 4: evaluate the parameters of the adaptive model (`test_ts_solve_adaptive_parameters`)

This test compares the output of the adaptive model with different iterations (`max_iter`) 
and minimum difference cutoffs (`min_diff`),

The results show that quality of the output improves when `max_iter` is increased, but this improvement is (obviously) 
limited by the minimum difference cutoff.

Good outputs are produced for `min_diff` values below 0.0001, which requires at 60 iterations to be reached.

**Given these results, we can conclude that `mid_diff` = 0.0001 and `max_iter` = 100 (or no `max_iter`) are the 
best parameters.**

## Test 5: evaluate the performance of the adaptive model (`test_ts_solve_adaptive_performance`)

This test compares the performance of the adaptive model with different iterations (`max_iter`) and 
minimum difference cutoffs (`min_diff`).

Times are very low, increasing with the number of executed iterations. This test offers no relevant insights
on the best parameters.

## Test 6: compare the performance of the deterministic and adaptive models (`test_ts_compare_performance`)

This test compares the performance of the deterministic and adaptive models, with the best parameters 
(`prob_threshold` = 0.0 for the deterministic model; `max_iter` = 100 and `min_diff` = 0.0001 for the adaptive model).

**Given the results of Test 6, we can conclude that the adaptive model is the best one.**
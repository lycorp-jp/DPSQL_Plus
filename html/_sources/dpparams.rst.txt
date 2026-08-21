.. currentmodule:: dpsql

DPParams
======================

The DPParams class provides a set of configurable parameters for Differential Privacy (DP) mechanisms in DPSQL+. 
This includes setting the noise level, which determines the privacy parameters epsilon and delta.

The DPParams class provides two options for determining the amount of noise added to query results:

1. **sigmas, tau, and sigma_for_thresholding**: Initializing with these parameters allows you to directly set the standard deviation of the noise added to each aggregation function and the parameters for tau-thresholding. **This option is deprecated but still supported for debugging purposes (e.g., removing noise by setting sigmas to 0). We recommend using epsilon and delta instead.**

2. **epsilon and delta**: Setting epsilon and delta allows DPSQL+ to automatically calculate the appropriate noise parameters. The privacy budget (epsilon and delta) is distributed among the aggregation functions and tau-thresholding based on the selected accountant mechanism.

Regardless of which option you choose, you need to set the following parameters:

1. **contribution_bound**: This parameter limits the maximum number of rows that a single user can contribute to the aggregation. This is essential for ensuring differential privacy by bounding the influence of any individual user on the query results.

2. **clipping_thresholds**: This parameter specifies the range [L, U] for clipping input data for each aggregation function. This is important for bounding the sensitivity of the query. ``None`` is allowed only for ``COUNT`` and ``COUNT_DISTINCT`` aggregations.

3. **min_frequency**: This parameter sets the minimum number of users required for a query result to be released. This parameter works in conjunction with tau-thresholding to ensure that results are only shown when they represent a sufficient number of individuals, providing an additional layer of privacy protection beyond differential privacy.

4. **accountant_class** (Optional): This parameter specifies which privacy accountant to use for splitting noise parameters. Available options include:
   
   - **None** (default): Uses basic composition and allocates ``epsilon/(n+1)`` to each aggregation function and tau-thresholding, where ``n`` is the number of aggregation functions.
   - **RenyiAccountant**: Uses Renyi DP composition for tighter privacy budget allocation.
   - **PLDAccountant**: Uses Privacy Loss Distribution for tighter privacy budget allocation.

   Using advanced accountants like RenyiAccountant or PLDAccountant can significantly reduce the amount of noise needed for the same privacy guarantees, improving utility.


When using epsilon and delta instead of directly specifying noise parameters, DPSQL+ calculates the values based on the parameters in DPParams and the types of aggregation functions before executing the query.
The calculation process is as follows:

1. The privacy budget (epsilon, delta) is divided equally among the aggregation functions and tau-thresholding.
   If an accountant_class is specified, it is used to optimize privacy budget allocation for better utility while maintaining the same privacy guarantees.

2. For each aggregation function, the sensitivity is calculated based on the contribution_bound and clipping_thresholds. 
   The sensitivity represents the maximum impact a single user can have on the query result.

3. Using the allocated privacy budget and the sensitivity, the appropriate sigma for each aggregation function is calculated using the analytic Gaussian mechanism. 
   This ensures that the noise added to each aggregation function satisfies the differential privacy guarantee.

4. For tau-thresholding, the parameters tau and sigma_for_thresholding are calculated based on the allocated privacy budget, contribution_bound, and min_frequency. 
   Tau-thresholding ensures that query results are only released when they are based on a sufficient number of users, further enhancing privacy protection.

The DPParams class provides methods to perform these calculations:

- **get_noise_parameters(sensitivities)**: Calculates and returns all noise parameters (sigmas, tau, sigma_for_thresholding) based on the sensitivities of the aggregation functions. This method uses the specified accountant_class (if provided) to optimize the noise parameters for better utility while maintaining the same privacy guarantees.

This method ensures that the privacy budget is properly allocated and that the noise added to the query results satisfies the differential privacy guarantee.

.. automodule:: dpsql.dp_params
   :members:
   :undoc-members:

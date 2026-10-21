.. currentmodule:: dpsql

Accountant
======================
The DPSQL+ Accountant is responsible for managing the privacy budget across multiple queries.

We provide two implementations of the Accountant class: `RenyiAccountant` and `PLDAccountant`.
`PLDAccountant` provides a tighter bound on privacy budget consumption than `RenyiAccountant`.
However, it is slower and does not support fully adaptive settings, where users can determine privacy parameters based on the results of previous queries.

We also provide `BasicAccountant` for testing and debugging, but it is not recommended for production because `RenyiAccountant` provides tighter privacy guarantees at similar speed.

The properties of the Accountant classes are summarized as follows:

.. list-table::

   * - Accountant
     - Composition
     - Speed
     - Fully adaptive composition [3]_
   * - RenyiAccountant
     - Rényi DP composition [1]_
     - Fast
     - ✔️
   * - PLDAccountant
     - PLD composition [2]_
     - Slow
     - ✖️
  
.. automodule:: dpsql.accountant
   :members:
   :undoc-members:


.. [1] Bun, Mark, and Thomas Steinke. "Concentrated differential privacy: Simplifications, extensions, and lower bounds." Theory of cryptography conference. Berlin, Heidelberg: Springer Berlin Heidelberg, 2016.
.. [2] Google Differential Privacy Team. "Privacy Loss Distribution." 2024, https://github.com/google/differential-privacy/blob/main/common_docs/Privacy_Loss_Distributions.pdf
.. [3] Desfontaines, Damien. "Open problem(s) - How generic can composition results be?" blog post, 2023, https://differentialprivacy.org/open-problems-how-generic-can-composition-be/
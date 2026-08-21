.. DPSQL documentation master file, created by
   sphinx-quickstart on Tue Nov 19 15:49:30 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DPSQL+
===================

DPSQL+ is a library that provides a SQL-like interface for querying sensitive data, along with privacy budget management based on differential privacy.
It is designed to provide a high level of privacy protection without requiring users to have a deep understanding of differential privacy.

Key features
------------

- DPSQL+ applies user-level differential privacy under the add-remove model and enforces a minimum frequency rule requiring that every released group contain at least :math:`k` distinct users.

- The :doc:`./accountant` module supports sequential and adaptive SQL queries. It provides RDP-based accounting for adaptive composition and PLD-based numerical accounting for sequential composition.

- DPSQL+ targets :math:`(\varepsilon, \delta)`-DP and therefore uses the Gaussian mechanism. This choice often improves the utility-privacy trade-off for sequential and adaptive queries.

For details, see our paper, *DPSQL+: A Differentially Private SQL Library with a Minimum Frequency Rule* (https://arxiv.org/abs/2602.22699).

Contents
------------

.. toctree:: 
   :maxdepth: 2

   overview

   installation
   quickstart

   accountant
   backend
   engine
   validator
   dpparams

# Changelog

## 0.2.0 (2020-07-15)
- Unify language for keys and variables to use the term "variable".
- Re-factor beliefpropogation code into a computation and construction module.
- Use Shafer-Shenoy updates for belief propogation in junction tree.
- Remove usage of networkx library in triangulation tests.
- Remove usage of list concatenation via sum() function.

## 0.1.2 (2018-09-15)
- Fix support for duplicate factors (#2).
- Assert that factors are given as a list of lists.

## 0.1.1 (2018-02-12)
- Support factors without edges in triangulation.
- Add `attrs` to dependencies.

## 0.1.0 (2018-02-11)
- Add triangulation and Junction tree construction for factor graphs.
- Add Hugin algorithm to propagate potentials by using sum-product distributive
  law.

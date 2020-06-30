# Changelog

## Dev
- FIXME

## 0.1.3 (2020-06-29)
- Unified language for keys and variables to use the term "variable".
- Re-factored beliefpropogation code into a computation and construction module.
- Using Shafer-Shenoy updates for belief propogation in junction tree.
- Removed usage of networkx library in triangulation tests.
- Removed usage of list concatenation via sum() function.

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

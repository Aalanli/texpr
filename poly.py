# %%
import islpy as isl


s1 = isl.UnionSet("[n] -> { [i, j] : 0 <= i <= 10 and i < n and 0 <= j <= 10 and exists k : 2k = j }")
s2 = isl.UnionSet("[n] -> { [i, j] : 0 <= i <= 10 and i < n and 0 <= j <= 10}")

v = isl.make_zero_and_vars("i,j,k", "n")
myset = (
    v[0].le_set(v["i"] + v["j"])
    &
    (v["i"] + v["j"]).lt_set(v["n"])
    &
    (v[0].le_set(v["i"]))
    &
    (v["i"].le_set(13 + v["n"]))
    )
# %%

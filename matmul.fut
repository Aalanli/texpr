def matmul [n][m][p] 'a
           (add: a -> a -> a) (mul: a -> a -> a) (zero: a)
           (A: [n][m]a) (B: [m][p]a) : [n][p]a =
  map (\A_row ->
         map (\B_col ->
                reduce add zero (map2 mul A_row B_col))
             (transpose B))
      A

def matmul_f32 = matmul (+) (*) 0f32

entry main [n][m][p] (a: [n][m]f32) (b: [m][p]f32): [n][p]f32 =
  matmul_f32 a b

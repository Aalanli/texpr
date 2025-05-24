module TExpr3 where

import Util
import IExpr
import Data.Sequence
import Data.Map

{-
informal semantics of tstmts:

--- // all threads assumed to sync upon entrance into a par block
par N i {
    --- // there is absolutely no guaranteed ordering for different instances of e(i) (this is for thread virtualization)
    --- // assume serial execution within each scope
    e(i)
}
--- // all threads assumed to sync upon leaving a par block
--- // all memory accesses are fully visible to later threads within the same par scope / nesting level

-}

data AutoSym = AutoSym ID | Concrete Int
data Mem = Mem { memId::ID, level::Maybe Int, size::AutoSym }

data TStmt s = 
    Par Int Index (Seq s)               -- par N i [s]
    | Seq Int Index (Seq s)             -- seq N i [s]
    | IfGuard (Seq (Seq ICond)) (Seq s) -- DNF form (or (and e11 ...) (and e21 ...) ...)
    | Bind Mem ITerm TETerm             -- Bind %v I E
    | NewMem Mem                        -- NewMem %v I E
    | Anchor TETerm
type TSTerm = TFix ID TStmt

data ICond = Le0 ITerm | Eq0 ITerm 

data TExpr e = View Mem ITerm | Operation OpType (Seq e)
data OpType = ArithConst Double | Add | Mul -- obviously add more

type TETerm = TFix ID TExpr

type ThreadFactors = Map ID Int
-- 1. get ThreadFactors
-- 2. memory promotion
-- 3. codegen

{-
example 1: matmul

A: f32[M * K]
B: f32[K * N]
C: f32[M * N]

par (M/Tm) im {
    par (N/Tn) in {
        tA = new f32[Tm * Tk]
        tB = new f32[Tk * Tn]
        tC = new f32[Tm * Tn]
        // init tC 
        // {0 -> tC[i] : 0 <= i < Tm * Tn }
        par (Tm * Tn) i {
            tC[i] = 0
        }

        seq (K/Tk) ik {
            // load into buffers
            // { A[(im * Tm + i) * K + (j + ik * Tk)] -> tA[i * Tk + j] : 0 <= i < Tm, 0 <= j < Tk }
            par Tm i {
                par Tk j {
                    tA[i*Tk + j] = A[(im * Tm + i) * K + (j + ik * Tk)]
                }
            }

            // { B[(in * Tn + i) + (j + ik * Tk) * N] -> tB[i + j * Tk] : 0 <= i < Tn, 0 <= j < Tk }
            par Tn i {
                par Tk j {
                    tB[i + j * Tk] = B[(in * Tn + i) + (j + ik * Tk) * N]
                }
            }
            // mma
            // { (tA[i * Tk + k], tB[k * Tn + j]) -> tC[i * Tn + j] : 0 <= i < Tm, 0 <= j < Tn }
            par Tm i {
                seq Tk k {
                    par Tn j {
                        tC[i * Tn + j] += tA[i * Tk + k] * tB[k * Tn + j]
                    }
                }
            }
        }
        // write back
        // { tC[i * Tn + j] -> C[(im + i) * M + (in + j)] : 0 <= j < Tn, 0 <= i < Tm }
        par Tn j {
            par Tm i {
                C[(im + i) * M + (in + j)] = tC[i * Tn + j]
            }
        }
    }
}

which is equivalent to (but less idiomatic?):
par (M/Tm) im {
    par (N/Tn) in {
        tA = new f32[Tm * Tk]
        tB = new f32[Tk * Tn]
        par (Tm * Tn) tid {
            ci = 0
            i = tid / Tn
            j = tid % Tn

            seq (K cdiv Tk) ik {
                seq (Tm * Tn cdiv (Tm * Tk)) i {
                    tA[i * Tm * Tk + tid] = A[...]
                }
                seq (...) i {
                    tB[...] = B[...]
                }
                sync()    

                seq Tk k {
                    ci += tA[i * Tk + k] * tB[k * Tn + j]
                }
            }

            C[(im + i) * M + (in + j)] = ci

        }
    }
}

-}
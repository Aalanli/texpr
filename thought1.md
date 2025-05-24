## The case for another tensor expression dsl

Currently there's no tensor DSL (that I'm aware of) that offers the combination of these features:
1. Relatively advanced optimizations (fusion, tiling, shared+reg promotion, etc.)
2. Lower level escape hatch, and good interoperability with the optimization engine
3. completely self-contained code gen with no runtime jit


There are different levels of dynamicism that such a DSL might support, each with it's own interface.

At the strictist level, all shapes are static and hence we desire such an interface:
```c++
struct Program {
    int getWorkspaceBytes();
    void launch_program(void **in, void **out, char* workspace);
};
```
Implied in this is no memory allocation and only kernel launches.

At the next level, input shapes can be dynamic, and hence the output shapes
can only be an expression of the input shapes.
```c++
struct Program {
    int getWorkspaceBytes(int *shapes);
    void getOutputShapes(int *shapes, int *outShapes);
    void launch_program(void *shapes, void **in, void **out, char* workspace);
};
```
Implied in this is no memory allocation and only kernel launches.

At the next level, the shapes can be data dependent as well.
```c++
struct MemAllocator {
    char* alloc(int N);
    void free(char* ptr);
};

struct Program {
    void launch_program(MemAllocator *alloc, int *inShapes, void **in, void **out, int *outShapes);
};
```

In this instance the program must manage the memory somehow.


In case 1, all kernel scheduling and selection can be done offline and no jit 
compilation is necessary.

In case 2, kernel compilation can be done offline with heuristics, with some optional runtime tuning. Case 2 could also be handled via quantization of case 1, but this will either require manually specified bounds or jit compilation.

Case 3 can be handled via graph partitioning of case 2.


## TExpr 1

```

tfunction = (pub)? fn NAME args [tstmt]

tstmt s = par N |i| [s]
        | seq N |i| [s]
        | let A N {memLvl}
        | bind A I e
texpr e = view A I
        | elemwise {opcode} [e]
```

example program 1
```
pub fn matmul(a: [N, K], b: [K, M]): [N, M] {
    let c [N, M]
    par (N, M) |i, j| {
        let c1 1
        seq K |k| {
            c1[0] += A[i, k] * B[k, j];
        }
        c[i, j] = c1[0];
    }
}

pub fn matmul2(a: [N, K], b: [K, M]): [N, M] {
    par (N/Tn, M/Tm) |it, jt| {
        let at [Tn, Tk]
        let bt [Tk, Tm]
        let ct [Tn, Tm] = 0;
        seq (K/Tk) |k| {
            par (Tn, Tk) |i, j| {
                at[i, j] = a[i + it * Tn, j + k * Tk];
            }
            par (Tk, Tm) |i, j| {
                bt[i, j] = b[i + k * Tk, j + jt * Tm];
            }

            par (Tn/Rn, Tm/Rm) |i, j| {
                let ra [Rn, Rk]
                let rb [Rk, Rm]
                let rc [Rn, Rm] = 0
                seq (Tk/Rk) |rk| {
                    par (Rn, Rk) |ri, rj| { ra[ri, rj] = ra[ri + i * Rn, rj + rk * Rk]; }
                    par (Rk, Rm) |ri, rj| { rb[rk, rj] = rb[ri + rk * Rk, rj + j * Rm]; }

                    par (Rn, Rm) |ri, rj| {
                        seq (Rk) |sk| {
                            rc[ri, rj] += ra[ri, rk] * rb[rk, rj];
                        }
                    }
                }
                par (Rn, Rm) |ri, rj| {
                    ct[ri + Rn * i, rj + Rm * j] += rc[ri, rj];    
                }
            }
        }
        par (Tn, Tm) |i, j| {
            c[i + it * Tn, j + jt * Tm] = ct[i, j];
        }
    }
}

pub fn matmul3(a: [N, K], b: [K, M]): [N, M] {
    par (N/Tn, M/Tm) |it, jt| {
        let at [Tn, Tk]
        let bt [Tk, Tm]
        let ct [Tn, Tm] = 0;
        seq (K/Tk) |k| {
            par (Tn, Tk) |i, j| {
                at[i, j] = a[i + it * Tn, j + k * Tk];
            }
            par (Tk, Tm) |i, j| {
                bt[i, j] = b[i + k * Tk, j + jt * Tm];
            }

            ct += matmul(at, bt); // TODO: function calls
        }
        par (Tn, Tm) |i, j| {
            c[i + it * Tn, j + jt * Tm] = ct[i, j];
        }
    }
}

```



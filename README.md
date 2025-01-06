# texpr

The TExpr abstract machine

```
te = 
   | Compute n=Int |i| te
   | Reduce  n=Int |i| te
   | Let v = te in te
   | Index v i
   | Const Double n=Int
   | Elemwise (e,)*
```

Each level of Compute corresponds to a level of the 
memory hierarchy and SIMD parallelism. 

```
Compute N |i| // Level L2 (grid)
    Compute M |j| // Level L1 (block)
        // (thread)
        Compute K |r| // Level L0 
            ...
```

Let in the respective level corresponds to memory allocation at that level

```
Let a = ... // (global)
Compute N |i| // Level L2
    Let b = ... // (shared)
    Compute M |j| // Level L1
        Let c = ... // (registers)
        ...
```

All Compute index operations can access every element of all higher
levels, but can only read. 
Compute definitions corresponds to parallel logical work elements,
and Reduce at the level of the Compute definition it is nested within 
(or the global one), and can be implemented in terms of Compute. 

Due to nested Computes, Compute definitions can return vector values.
There are only vectors with associated length in this model, while tensors can be implemented on top, this is due to redundancies in representing layouts. 
The data-layout of Compute with vector valued nested expression
is the concatenation of inner instances. Eg.
```
Compute 3 |_|
    [0,1,2]
== [0,1,2,0,1,2,0,1,2]
```
One can optionally specify an index map to permute this
```
Compute 3 {i -> (i%3)*3 + i/3} |_| 
    [0,1,2]
== [0,0,0,1,1,1,2,2,2]
== let a = Compute 3 |_|
    [0,1,2] in 
    Compute 9 |i|
        a[(i%3)*3 + i/3]
```

TExpr is responsible for partitioning work as to maximize data
reuse/locality within a nested memory hierarchy. This includes
fusion, tiling, etc.

TExpr2 is responsible for assigning physical work elements onto
logical ones, and thus exposes layout information. Vectors at level
Ln can be a flat buffer allocated at Ln, or nested compositions of 
memory from L{n-1} to L0. 
Still undecided on how to represent these layouts, but the basic
premise is a mapping from physical worker ids to logical elements,
or vice-versa.
I wonder if this is even necessary, since this adds complexity
and breaks the earlier abstraction. Also, the whole point of materializing tensors in higher levels of memory is to share data between parallel compute elements, so Computes that stay local can be fused. 


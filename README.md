# texpr

TExpr is a research compiler prototype aiming to make GPU kernel code more composable and expressive, and thus make it easier for CUDA programmers. It grew out of a frustration with existing tools that tries to replace CUDA programming altogether but fails at (relatively common) edgecases, and tools that don't have good support for ahead-of-time inference (being too focused on JIT in python).

TExpr leverages techniques from polyhedral compilation to address the problem of composability - that is, how to performantly compose parallel code fragments in a non-leaky abstraction (unlike regular `__device__` annoted procedures in CUDA).  
My hope is that my work on this project can be integrated further up the stack of ML compilers - eventually helping the lowering of tensor level IRs into efficient device code.

You can read more about it [here](https://aalanli.github.io/), but from a high-level perspective, TExpr makes composition of parallel programs more composable by providing non-leaky function and parallel decomposition abstractions. CUDA and AMD's HIP `__device__` procedures are leaky because the caller/callee must make implicit assumptions about the state of the program. For example: should all threads call this procedure, should threads by coalesced, what state does this procedure leave the caller, etc. Furthermore, with the introduction of more levels in the parallel hierarchy (from what's there already: threads, warps, blocks, gpus), and more sophisticated fusions, scheduling becomes more challenging.

TExpr introduces the `par` control flow block that abstracts the notion of parallel execution. When a programmer writes 
```
par N |i| {
    e(i)
}
```
it signifies that this is a parallel loop, and the statements in it are eligible to executed in parallel. The loop is and must be semantically identical to 
```
for i in 0..N {
    e(i)
}
```
However, unlike CUDA threads and like CUDA blocks, the instances in this block are not guaranteed to be executed fully in parallel; the compiler is free to balance instruction level-parallelism with thread parallelism by scheduling some iterations on the same thread. For example, the first example may be transformed to:
```
par N/2 |i'| {
    for j in 0..2 {
        e(i' * 2 + j)
    }
}
```
The difficulty arises when you have multiple nested `par` and regular `for` loops, TExpr uses techniques from polyhedral compilation, namely representing the loop structure as a quasi-affine polyhedron, and performing mixed linear programming on top. Suffice it to say that its not easy and theoretically, at worst, such programs are equivalent to mixed integer quadratic optimization problems, so NP hard (haha).

TExpr makes the guarantee that all threads entering the `par` block are converged and all memory effects fully materialized, and all threads exiting the `par` block is fully converged, so explicit synchronization is not necessary. This guarantee is also true for functions, and so functions are no longer 'colored'; you can call them anywhere, and there shouldn't be a separate between device and host procedures. 

You might be wondering what about some algorithms that makes the assumption that all threads/units of parallelism are fully resident? TExpr provides escape mechanisms by which you can signal to the compiler that this block must be executed on the thread/warp/block level, thereby enabling inline asm as well. I go into more detail in the link above.

Like a silicon valley startup founder I am sharing my vision of what TExpr could be, not what it is right now. Quite unlike a founder, however, I am not getting paid for this work. While I believe the theoretical foundations are there, I should sign off with the important caveat that something unexpected could happen. Right now I have moved all development into a private repository due to the project's immature status.

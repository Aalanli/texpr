# texpr

TExpr is a research compiler prototype aiming to make GPU kernel code more composable and expressive. It grew out of a frustration with existing tools that tries to replace CUDA programming altogether but fails at (relatively common) edgecases, and tools that don't have good support for ahead-of-time inference (being too focused on JIT in python).
TExpr leverages techniques from polyhedral compilation to address the problem of composability - that is, how to performantly compose parallel code fragments in a non-leaky abstraction (unlike regular `__device__` annoted procedures in CUDA).  You can read more about it [here](https://aalanli.github.io/). 
My hope is that my work on this project can be integrated further up the stack of ML compilers - eventually helping the lowering of tensor level IRs into efficient device code.

I have moved all work into a private repository for now due to the project's recency. 

% What is functional programming, and why do we care?
% Conal Elliott
% July 2012

 <!-- References -->

 <!-- -->

# What is functional programming?

*   High-level, parallel-friendly programming.
*   Strong foundation for correctness and optimization.

# Why not Verilog?

# What makes a language good for parallelism?

# What makes a language *bad* for parallelism?

*   That is, what gives most programming languages their sequential bias?
*   (C, C++, Java, Python, etc.)
    *   Basic primitive: assignment (state change)
    *   Basic composition: *sequential* execution
*   *Over-linearizes* algorithms.
*   Parallelizing compilers must distinguish accidental sequentiality from essential.

# A complex computation involves zillions of small simple computations.

*   Question: Why not compute them all at once?
*   Answer: data dependencies.
*   Conclusion: Minimize dependencies.

# Dependencies

*   Three kinds:

    1.   Inherent in problem/question
    2.   Resulting from algorithm
    3.   Resulting from language

*   Goals: eliminate #3 and reduce #2.

# Explicit and implicit dependencies

*   Imperative code:

    ~~~~{.C}
    a = foo(...);
    b = bar(...);
    return baz(a,b);
    ~~~~

*   Fewer assignments:

    ~~~~{.C}
    return baz(foo(...),bar(...));
    ~~~~

    Still problematic. How?

# Idea: remove *all* state/mutation

*   Only explicit data dependencies remain.
*   Now programming is like math/calculation.
*   Precise & tractable reasoning (algebra).
*   Allows optimization/transformation.

# Functional Programming

*   More aptly, "value-oriented programming".
*   Like arithmetic/algebra/trig (etc), but beyond small values (numbers) to big ones (strings, sequences, streams, trees, images, geometry, functions).
*   Simple and useful algebraic properties.
*   Capture *high-level* computation patterns, often in parallel-friendly ways.
    Examples: `map`, `fold`/`unfold`, `scan`.
    For `fold` & `scan`, associativity enables parallelism, while `map` is always parallel-friendly.
*   Great for code reuse.
    (Code is a liability.)

# Static typing

*   Automate consistency checking.
*   Catch bugs much earlier.
*   Ideal: if the code type-checks, it's correct.
    Often achieved.
*   Ideal: eliminate all incorrect programs, but keep all correct ones.
*   Helps evolution.
    Guide the transition through inconsistent intermediate code states.
*   We'll need rich/sophisticated/fancy types.
    The functional discipline helps as well.
    Harder with sequential.

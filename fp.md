% Functional programming
% Conal Elliott
% July 10, 2012

# Why & What

## Why functional programming?

*   \wow{Parallelism}
    \note{Parallel-friendly}
*   Correctness
    \note{Practical \& precise reasoning.}
*   Productivity
    \note{Captures high-level programming patterns formally for reuse.
    Code is a liability.}

## What is functional programming?

*   Value-oriented
    \note{as opposed to action-oriented}
*   Like arithmetic on big values
    \note{strings, sequences, streams, trees, images, geometry, functions.}

## Finishes a shift that Fortran began

*   Machine/assembly: statements only
    \note{Built up by sequencing.}
*   Fortran etc: statements + expressions
    \note{Expressions on RHSs. Nestable!
    Mainstream (imperative) languages are all Fortran variations.}
*   Functional: expressions only
    \note{With expressions like these, who needs statements?}

# Parallelism

## What makes a language good for parallelism?

...

## What makes a language *bad* for parallelism?

*   Sequential bias
    *   Primitive: assignment (state change)
    *   Composition: *sequential* execution
    *   "Von Neumann" languages (Fortran, C, Java, Python, ...)
*   *Over-linearizes* algorithms.
*   Hard to isolate accidental sequentiality.

## Applications perform squillions of simple computations.

*   Compute all at once?
*   Oops -- data dependencies.
*   Minimize dependencies!

## Dependencies

*   Three sources:
    1.   Problem
    2.   Algorithm
    3.   Language
*   Goals: eliminate #3, and reduce #2.

## Dependency in imperative languages

*   Built into sequencing: $A\, ; B$
*   Semantics: $B$ begins where $A$ ends.

## Idea: remove *all* state

*   And, with it, remove
    *   mutation,
    *   sequencing,
    *   statements.
*   Expression dependencies are specific & explicit.
*   Remainder can be parallel.
*   Programming is calculation/math:
    *   Precise & tractable reasoning (algebra),
    *   ... including optimization/transformation.
*   No loss of expressiveness!

# Examples

## Sequential sum

C:

~~~{.C}
int sumArr(int arr[], int n) {
    int sum = 0;
    for (int i=0; i<n; i++)
      sum += arr[i];
    return sum;
}
~~~

Haskell:

> sum l = foldl (+) 0 l

where

> foldl (+) 0 [a,b,...,z] == (...((0 + a) + b) ...) + z

## Parallel sum -- how?

Left-associated sum: $$(\ldots ((0 + a) + b) \ldots) + z$$

How to parallelize?

\pause

*   Divide and conquer
    *   Sum each part independently
    *   Then sum results
*   Generalize beyond +,0.
\pause
*   *When valid?*

## Associative folds

*Monoid*: type with associative operator & identity.

> fold :: Monoid a => [a] -> a

Not just lists:

> fold :: (Foldable f, Monoid a) => f a -> a

Balanced data structures lead to balanced parallelism.

## Balance

Contrast:

> data List a = Nil | Cons a (List a)

with

> data Tree a = L a | B (Tree a) (Tree a)

Can enforce tree balance with fancier types.

## Trickier algorithm: prefix sums

C:

~~~{.C}
int prefixSums(int arr[], int n){
    int sum = 0;
    for (int i=0; i<n; i++) {
        int next = arr[i];
        arr[i] = sum;
        sum += next;
    }
    return sum;
}
~~~

Haskell:

> prefixSums l = scanl (+) 0 l

## Prefix sums on trees

> prefixSums t = scanl (+) 0 t
>
> scanl op acc (L a)   = (mempty, acc `op` a)
> scanl op acc (B u v) = (B u' v', uvTot)
>  where
>    (u', uTot) = scanl op acc  u
>    (v',uvTot) = scanl op uTot v

*   Still very sequential.
*   Does associativity help as with `fold`?

## Parallel prefix sums

General version:

> scan :: (Traversable f, Monoid a) => f a -> (f a, a)

On trees,

> scan (L a)   = (mempty, a)
> scan (B u v) = (B u' v'', uTot `mappend` vTot)
>  where
>    (u',uTot) = scan u
>    (v',vTot) = scan v
>    v'' = map (uTot `mappend`) v'

If balanced, dependency depth $O (\log n)$, work $O (n \log n)$.

Can reduce work to $O (n)$.

# Misc

## 1977 Turing Award -- John Backus

\ \ \ \ \ ![1977 Turing Award](../BackusTuringPaperHighlight.png)

## Static typing

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

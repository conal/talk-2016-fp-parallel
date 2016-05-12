%% -*- latex -*-

%% %let atwork = True

%% % Presentation
%% \documentclass{beamer}

% Printed
\documentclass[serif,handout]{beamer}

%% % 2-up
%% \usepackage{pgfpages}
%% \pgfpagesuselayout{2 on 1}[border shrink=1mm]

%% % Printed, 4-up
%% \documentclass[serif,handout,landscape]{beamer}
%% \usepackage{pgfpages}
%% \pgfpagesuselayout{4 on 1}[border shrink=1mm]

\usefonttheme{serif}

\usepackage{beamerthemesplit}

%% % http://www.latex-community.org/forum/viewtopic.php?f=44&t=16603
%% \makeatletter
%% \def\verbatim{\small\@verbatim \frenchspacing\@vobeyspaces \@xverbatim}
%% \makeatother

\usepackage{graphicx}
\usepackage{color}
\DeclareGraphicsExtensions{.pdf,.png,.jpg}

\usepackage{wasysym}

\useinnertheme[shadow]{rounded}
% \useoutertheme{default}
\useoutertheme{shadow}
\useoutertheme{infolines}
% Suppress navigation arrows
\setbeamertemplate{navigation symbols}{}

\input{macros}

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include mine.fmt

\title{Functional programming and parallelism} % 
\author{\href{http://conal.net}{Conal Elliott}}
\date{July 2012 / May 2016}

\setlength{\itemsep}{2ex}
\setlength{\parskip}{1ex}

\setlength{\blanklineskip}{1.5ex}

\nc\pitem{\pause \item}

%%%%

% \setbeameroption{show notes} % un-comment to see the notes

\begin{document}

\frame{\titlepage}

\title{FP and parallelism}
\date{May 2016}

%if False
\framet{Why functional programming?}{

\begin{itemize}\itemsep 3ex
\item
  Correctness \note{Practical \& precise reasoning.}
\item
  Productivity \note{Captures high-level programming patterns formally for reuse.
  Code is a liability.}
\item
  \wow{Parallelism} \note{Parallel-friendly}
\end{itemize}
}

\framet{What is functional programming?}{

\begin{itemize}\itemsep 3ex
\item
  Value-oriented \note{as opposed to action-oriented}
\item
  Like arithmetic on big values \note{strings, sequences, streams, trees, images, geometry, functions.}
\end{itemize}
}

\framet{Finishes a shift that Fortran began}{

\begin{itemize}\itemsep 3ex
\item
  Machine/assembly: statements only \note{Built up by sequencing.}
\pitem
  Fortran etc: statements + expressions \note{Expressions on RHSs. Nestable!
  Mainstream (imperative) languages are all Fortran variations.}
\pitem
  Functional: expressions only \note{With expressions like these, who needs statements?}
\end{itemize}
}

%endif

\framet{What makes a language good for parallelism?}{

\ldots{}
}

\framet{What makes a language \emph{bad} for parallelism?}{
\begin{itemize}\itemsep 3ex
\pitem
  Sequential bias\vspace{2ex}
  \begin{itemize}\itemsep 2ex
    \item
    Primitive: assignment (state change)
  \item
    Composition: \emph{sequential} execution
  \item
    ``Von Neumann'' languages (Fortran, C, Java, Python, \ldots{})
  \end{itemize}\vspace{-1ex}
\pitem
  \emph{Over-linearizes} algorithms.
\item
  Hard to isolate accidental sequentiality.
\end{itemize}
}

\framet{Can we fix sequential languages?}{

\begin{itemize}\itemsep 3ex
\item
  Throw in parallel composition
\pitem
  Oops:\vspace{0.5ex}

\begin{minipage}[c]{0.5\textwidth}
  \begin{itemize}\itemsep 2ex
    \item Nondeterminism
    \item Deadlock
    \item Intractable reasoning
  \end{itemize}
\end{minipage}
\pause
\begin{minipage}[c]{0.3\textwidth}
\includegraphics{yuk.png}
\end{minipage}
\end{itemize}
}

\framet{Can we \emph{un-break} sequential languages?}{

\vspace{6ex}

\pause

\begin{quote}
\emph{Perfection is achieved not when there is nothing left to add, but when there is nothing left to take away.}
\end{quote}

\begin{flushright}
Antoine de Saint-Exup\'ery
\end{flushright}
}

\framet{Applications perform zillions of simple computations.}{

\begin{itemize}\itemsep 3ex
\item
  Compute all at once?
\item
  Oops -- dependencies.
\item
  Minimize dependencies!
\end{itemize}
}

\framet{Dependencies}{

\begin{itemize}\itemsep 3ex
\item
  Three sources:\vspace{2ex}
  \begin{enumerate}\itemsep 2.5ex
  \def\labelenumi{\arabic{enumi}.}
    \item
    Problem
  \item
    Algorithm
  \item
    Language
  \end{enumerate}
\item
  Goals: eliminate \#3, and reduce \#2.
\end{itemize}
}

\framet{Dependency in imperative languages}{

\begin{itemize}\itemsep 5ex
\item
  Built into sequencing: \(A\, ; B\)
\item
  Semantics: \(B\) begins where \(A\) ends.
\end{itemize}
}

\framet{Idea: remove \emph{all} state}{

\begin{itemize}\itemsep 3ex
\item
  And, with it,\vspace{1.5ex}
  \begin{itemize}\itemsep 1.5ex
    \item
    mutation,
  \item
    sequencing,
  \item
    statements.
  \end{itemize}\vspace{-1ex}
\item
  Expression dependencies are specific \& explicit.
\item
  Remainder can be parallel.
\end{itemize}

~

\begin{itemize}\itemsep 3ex
\pitem
  Contrast: ``\(A\, ; B\)'' vs ``\(A + B\)'' vs ``\((A + B) \times C\)''.
\end{itemize}
}

\framet{Stateless programming}{

\begin{itemize}\itemsep 5ex
\item
  Programming is calculation/math:\vspace{1.5ex}
  \begin{itemize}\itemsep 2ex
    \item
    Precise \& tractable reasoning (algebra),
  \item
    \ldots{} including optimization/transformation.
  \end{itemize}
\item
  No loss of expressiveness!
\item
  ``Functional programming'' (value-oriented) \note{as opposed to action-oriented}
\item
  Like arithmetic on big values \note{strings, sequences, streams, trees, images, geometry, functions.}
\end{itemize}
}

\framet{Sequential sum}{

\textbf{C:}
\vspace{-2ex}
\begin{verbatim}
   int sum(int arr[], int n) {
       int acc = 0;
       for (int i=0; i<n; i++)
         acc += arr[i];
       return acc;
   }
\end{verbatim}

\vspace{1ex}\pause

\textbf{Haskell:}
\begin{code}
sum = sumAcc 0
  where
    sumAcc acc []      = acc
    sumAcc acc (a:as)  = sumAcc (acc + a) as
\end{code}
}

\framet{Refactoring}{

\begin{code}
sum = foldl (+) 0
\end{code}

where

\begin{code}
foldl op acc []      = acc
foldl op acc (a:as)  = foldl op (acc `op` a) as
\end{code}
}

\framet{Right alternative}{

\begin{code}
sum = foldr (+) 0
\end{code}

where

\begin{code}
foldr op e []      = e
foldr op e (a:as)  = a `op` foldr op e as
\end{code}
}

\framet{Sequential sum --- left}{
\wfig{4.75in}{figures/sum-lvec6-no-opt}
}

\framet{Sequential sum --- right}{
\wfig{4.75in}{figures/sum-rvec6-no-opt}
}

\framet{Parallel sum -- how?}{

Left-associated sum:

\begin{code}
sum [a,b,...,z] == (...((0 + a) + b) ...) + z
\end{code}

\vspace{5ex}

How to parallelize?

Divide and conquer?
}

\framet{Balanced data}{

\begin{code}
data Tree a = L a | B (Tree a) (Tree a)
\end{code}

Sequential:

\begin{code}
sum = sumAcc 0
  where
   sumAcc acc (L a)    = acc + a
   sumAcc acc (B s t)  = sumAcc (sumAcc acc s) t
\end{code}

Again, |sum = foldl (+) 0|.

\pause

Parallel:

\begin{code}
sum (L a)    = a
sum (B s t)  = sum s + sum t
\end{code}

Equivalent? Why?
}

\framet{Balanced tree sum --- depth 4}{
\vspace{-4ex}
\wfig{4.75in}{figures/sum-rt4-no-opt}
}

\framet{Balanced computation}{

\begin{itemize}\itemsep 5ex
\item
  Generalize beyond +, 0.
\pitem
  When valid?
\end{itemize}
}

\framet{Associative folds}{

\emph{Monoid}: type with associative operator \& identity.

\begin{code}
fold :: Monoid a => [a] -> a
\end{code}

\vspace{2ex}

Not just lists:

\begin{code}
fold :: (Foldable f, Monoid a) => f a -> a
\end{code}

\vspace{5ex}

Balanced data structures lead to balanced parallelism.
}

\framet{Two associative folds}{

\pause

%% On lists:

\begin{code}
fold :: Monoid a => [a] -> a
fold []      = mempty
fold (a:as)  = a `mappend` fold as
\end{code}

~

%% On trees:

\begin{code}
fold :: Monoid a => Tree a -> a
fold (L a)    = a
fold (B s t)  = fold s `mappend` fold t
\end{code}

~

Derivable automatically from types.
}

\framet{Trickier algorithm: prefix sums}{

\textbf{C:}
\vspace{-2ex}
\begin{verbatim}
   int prefixSums(int arr[], int n) {
       int sum = 0;
       for (int i=0; i<n; i++) {
           int next = arr[i];
           arr[i] = sum;
           sum += next;
       }
       return sum;
   }
\end{verbatim}

\textbf{Haskell:}
\begin{code}
prefixSums = scanl (+) 0
\end{code}
}

\framet{Sequence prefix sum}{
\vspace{4ex}
\wfig{4.7in}{figures/lsumsT-lvec6-no-opt}
}

\framet{Sequential prefix sums on trees}{
\begin{code}
prefixSums = scanl (+) 0

scanl op acc (L a)    = (L acc, acc `op` a)
scanl op acc (B u v)  = (B u' v', uvTot)
  where
    (u', uTot)  = scanl op acc  u
    (v',uvTot)  = scanl op uTot v
\end{code}
}

\framet{Sequential prefix sums on trees --- depth 2}{
\vspace{4ex}
\wfig{4.7in}{figures/lsumsT-rt2-no-opt}
}

\framet{Sequential prefix sums on trees --- depth 3}{
\vspace{4ex}
\wfig{4.7in}{figures/lsumsT-rt3-no-opt}
}

%if False
\framet{Sequential prefix sums on trees --- depth 4}{
\vspace{2ex}
\wfig{4.7in}{figures/lsumsT-rt4-no-opt}
}
%endif

\framet{Sequential prefix sums on trees}{

\begin{code}
prefixSums = scanl (+) 0

scanl op acc (L a)    = (L acc, acc `op` a)
scanl op acc (B u v)  = (B u' v', vTot)
  where
    (u',uTot)  = scanl op acc  u
    (v',vTot)  = scanl op uTot v
 \end{code}

\begin{itemize}\itemsep 3ex
\pitem
  Still very sequential.
\item
  Does associativity help as with |fold|?
\end{itemize}
}

\framet{Parallel prefix sums}{

On trees:

\begin{code}
scan (L a)    = (L mempty, a)
scan (B u v)  = (B u' (fmap adjust v'), adjust vTot)
  where
    (u',uTot)  = scan u
    (v',vTot)  = scan v
    adjust x   = uTot `mappend` x
\end{code}

\begin{itemize}\itemsep 2ex
\item
  If balanced, dependency depth \(O (\log n)\), work \(O (n \log n)\).
\item
  Can reduce work to \(O (n)\).
  \emph{(\href{https://github.com/conal/talk-2013-understanding-parallel-scan}{Understanding efficient parallel scan}).}
\pitem
  Generalizes from trees.
\item
  \emph{Automatic from type.}
\end{itemize}
}

\framet{Balanced parallel prefix sums --- depth 2, unoptimized}{
\vspace{-1ex}
\wfig{4.75in}{figures/lsums-rt2-no-opt}
}
\framet{Balanced parallel prefix sums --- depth 3, unoptimized}{
\vspace{-1.5ex}
\wfig{4.5in}{figures/lsums-rt3-no-opt}
}
\framet{Balanced parallel prefix sums --- depth 4, unoptimized}{
\vspace{-2ex}
\wfig{4.25in}{figures/lsums-rt4-no-opt}
}

\framet{Balanced parallel prefix sums --- depth 2, optimized}{
\vspace{-2ex}
\wfig{4.75in}{figures/lsums-rt2}
}
\framet{Balanced parallel prefix sums --- depth 3, optimized}{
\vspace{-2ex}
\wfig{4.7in}{figures/lsums-rt3}
}
\framet{Balanced parallel prefix sums --- depth 4, optimized}{
\vspace{-3ex}
\wfig{4.75in}{figures/lsums-rt4}
}

%if False

\framet{CUDA parallel prefix sum}{

\begin{verbatim}
__global__ void scan(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int offset = 1;
    temp[2*thid] = g_idata[2*thid];
    temp[2*thid+1] = g_idata[2*thid+1];
    for (int d = n>>1; d > 0; d >>= 1) { 
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    ...
\end{verbatim}
}

\framet{CUDA parallel prefix sum (cont)}{

\begin{verbatim}
    if (thid == 0) { temp[n - 1] = 0; }
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t; 
        }
    }
    __syncthreads();
    g_odata[2*thid] = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1];
}
\end{verbatim}
}

%endif

\framet{Manual vs automatic placement}{

\begin{itemize}\itemsep 3ex
\item
  Programmers used to explicitly place computations in space.
\pitem
  Mainstream programming \emph{still} manually places in time.
\pitem
  Sequential composition: crude placement tool.
\item
  Threads: notationally clumsy \& hard to manage correctly.
\item
  If we relinquish control, automation can do better.
\end{itemize}
}

%if False
\framet{1977 Turing Award -- John Backus}{
~~~~~\includegraphics{BackusTuringPaperHighlight.png}
}
%endif

\end{document}

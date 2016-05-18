%% -*- latex -*-

%% % Presentation
%% \documentclass{beamer}

% Printed
\documentclass[handout]{beamer}

%% % 2-up
%% \usepackage{pgfpages}
%% \pgfpagesuselayout{2 on 1}[border shrink=1mm]

%% % Printed, 4-up
%% \documentclass[serif,handout,landscape]{beamer}
%% \usepackage{pgfpages}
%% \pgfpagesuselayout{4 on 1}[border shrink=1mm]

\usefonttheme{serif}

% \usepackage{beamerthemesplit}

%% % http://www.latex-community.org/forum/viewtopic.php?f=44&t=16603
%% \makeatletter
%% \def\verbatim{\small\@verbatim \frenchspacing\@vobeyspaces \@xverbatim}
%% \makeatother

\usepackage{graphicx}
\usepackage{color}
\DeclareGraphicsExtensions{.pdf,.png,.jpg}

%% \usepackage{wasysym}

\useinnertheme[shadow]{rounded}
\useoutertheme{default}
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
\date{May 2016} % original: July 2012

\setlength{\itemsep}{2ex}
\setlength{\parskip}{1ex}

\setlength{\blanklineskip}{1.5ex}

\nc\pitem{\pause \item}

%%%%

% \setbeameroption{show notes} % un-comment to see the notes

\begin{document}

%% \frame{\titlepage}

{ % local tweak
\setbeamertemplate{footline}{}
\frame{\titlepage}
}

%% % Short form to fit
%% \title{FP and parallelism}
%% \date{May 2016}

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

\begin{itemize}\itemsep 5ex
\item
\begin{minipage}[c]{0.45\textwidth}
  Throw in parallel composition.
\end{minipage}
\begin{minipage}[c]{0.45\textwidth}
\wfig{1.7in}{figures/hindenburg-over-new-york-1937}
\end{minipage}

\vspace{2ex}

\begin{minipage}[c]{0.45\textwidth}
\pitem
  Oops:\vspace{0.5ex}
 
  \begin{itemize}\itemsep 2ex
    \item Nondeterminism
    \item Deadlock
    \item Intractable reasoning
  \end{itemize}
\end{minipage}
\begin{minipage}[c]{0.45\textwidth}
\wfig{1.4in}{figures/hindenburg-disaster-1937}
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

\begin{itemize}\itemsep 5ex
\item
  Compute all at once?
\pitem
  Oops --- dependencies.
\item
  Minimize dependencies!
\end{itemize}
}

\framet{Dependencies}{

\begin{itemize}\itemsep 5ex
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
  % accept \#1
\end{itemize}
}

\framet{Dependency in sequential languages}{

\begin{itemize}\itemsep 5ex
\item
  Built into sequencing: $A\, ; B$
\item
  Semantics: $B$ begins where $A$ ends.
\item
  Why sequence?
\end{itemize}
}

\framet{Idea: remove \emph{all} state}{

\begin{itemize}\itemsep 3ex
\item
  And, with it,\vspace{1.5ex}
  \begin{itemize}\itemsep 1.5ex
    \item
    mutation (assignment),
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
  Contrast: ``$A\, ; B$'' vs ``$A + B$'' vs ``$(A + B) \times C$''.
\end{itemize}
}

\framet{Programming without state}{

\begin{itemize}\itemsep 3ex
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
\vspace{2ex}
\pitem
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
\vspace{1ex}
\wfig{4.75in}{figures/sum-rvec6-no-opt}
}

\framet{Parallel sum --- how?}{

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
scanl op acc (B u v)  = (B u' v', vTot)
  where
    (u',uTot)  = scanl op acc   u
    (v',vTot)  = scanl op uTot  v
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
    (u',uTot)  = scanl op acc   u
    (v',vTot)  = scanl op uTot  v
 \end{code}

\begin{itemize}\itemsep 3ex
\pitem
  Still very sequential.
\item
  Does associativity help as with |fold|?
\end{itemize}
}

\framet{Parallel prefix sums on trees}{

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
  If balanced, dependency depth $O (\log n)$, work $O (n \log n)$.
\item
  Can reduce work to $O (n)$.
  \emph{(\href{https://github.com/conal/talk-2013-understanding-parallel-scan}{Understanding efficient parallel scan}).}
\pitem
  Generalizes from trees.
\item
  \emph{Automatic from type.}
\end{itemize}
}

\framet{Balanced parallel prefix sums --- depth 2}{
\vspace{-5ex}
\wfig{4.75in}{figures/lsumsp-rt2-no-hash-no-opt}
}
\framet{Balanced parallel prefix sums --- depth 3}{
\vspace{-4ex}
\wfig{4.75in}{figures/lsumsp-rt3-no-hash-no-opt}
}
\framet{Balanced parallel prefix sums --- depth 4}{
\vspace{-4ex}
\wfig{4.75in}{figures/lsumsp-rt4-no-hash-no-opt}
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

\framet{Why functional programming?}{
\pause
\begin{itemize}\itemsep 5ex
\item
  \wow{Parallelism} \note{Parallel-friendly}
\item
  Correctness \note{Practical \& precise reasoning.}
\item
  Productivity \note{Captures high-level programming patterns formally for reuse.
  Code is a liability.}
\end{itemize}
}

%if False
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

%if False
\framet{1977 Turing Award --- John Backus}{
~~~~~\includegraphics{BackusTuringPaperHighlight.png}
}
%endif

\framet{R\&D agenda: elegant, massively parallel FP}{
\pause
\begin{itemize}\itemsep 5ex
\item
  Algorithm design:\vspace{1.5ex}
  \begin{itemize}\itemsep 2ex
  \item Functional \& richly typed
  \item Parallel-friendly
  \item Easily composable
  \end{itemize}
\item
  Compiling for highly parallel execution:\vspace{1.5ex}
  \begin{itemize}\itemsep 2ex
  \item Convert to algebraic vocabulary (CCC).
  \item Interpret vocabulary as ``circuits'' (FPGA, silicon, GPU).
  \item Other interpretations, e.g., automatic differentiation.
  \end{itemize}
\end{itemize}
}

\framet{Composable data structures}{
\pause
\begin{minipage}[c]{0.63\textwidth}
\begin{itemize}\itemsep 2ex
\item Data structure tinker toys:\vspace{1ex}
\begin{code}
data  Empty        a = Empty

data  Id           a = Id a

data  (f  :+:  g)  a = L (f a) | R (g a)

data  (f  :*:  g)  a = Prod (f a) (g a)

data  (g  :.:  f)  a = Comp1 (g (f a))
\end{code}
\end{itemize}
\begin{itemize}\itemsep 2ex
\item Specify algorithm version for each.
\item Automatic, type-directed composition.
\end{itemize}
\end{minipage}
\begin{minipage}[c]{0.36\textwidth}
\wpicture{1.5in}{figures/tinker-toy-bird}
\end{minipage}
}

\nc\Id{\Varid{Id}}

\framet{Vectors}{

\vspace{-4ex}
$$\overbrace{\Id \times \cdots \times \Id}^{n\text{~times}}$$

\pause
Right-associated:
\begin{code}
type family RVec n where
  RVec Z      = U1
  RVec (S n)  = Par1 :*: RVec n
\end{code}

\vspace{2ex}

Left-associated:
\begin{code}
type family LVec n where
  LVec Z      = U1
  LVec (S n)  = LVec n :*: Par1
\end{code}
}

\nc\Pair{\Varid{Pair}}

\framet{Perfect binary leaf trees}{

\vspace{-2ex}
$$\overbrace{\Pair \circ \cdots \circ \Pair}^{n\text{~times}}$$

\pause
Right-associated:
\begin{code}
type family RBin n where
  RBin Z      = Par1
  RBin (S n)  = Pair :.: RBin n
\end{code}

\vspace{0ex}

Left-associated:

\begin{code}
type family LBin n where
  LBin Z      = Par1
  LBin (S n)  = LBin n :.: Pair
\end{code}

\vspace{1ex}

Uniform pairs:
\begin{code}
type Pair = Par1 :*: Par1
\end{code}
}

\framet{Generalized trees}{

\vspace{-3ex}
$$\overbrace{h \circ \cdots \circ h}^{n\text{~times}}$$

\vspace{0ex}

Right-associated:
\begin{code}
type family RPow h n where
  RPow h Z      = Par1
  RPow h (S n)  = h :.: RPow h n
\end{code}

\vspace{1ex}

Left-associated:

\begin{code}
type family LPow h n where
  LPow h Z      = Par1
  LPow h (S n)  = LPow h n :.: h
\end{code}

Binary:
\begin{code}
type RBin n = RPow Pair n
type LBin n = LPow Pair n
\end{code}
}

\framet{Composing scans}{
\vspace{-1ex}
\begin{code}
class LScan f where
  lscan :: Monoid a => f a -> (f :*: Par1) a

pattern And1 fa a = Prod fa (Par1 a)
\end{code}

\begin{code}
instance LScan U1 where
  lscan fa = And1 fa mempty

instance LScan Par1 where
  lscan (Par1 a) = And1 (Par1 mempty) a

instance (LScan f, LScan g)  => LScan (f :*: g) where
  lscan (Prod fa ga)  = And1 (Prod fa' ga') gx
   where
     And1 fa'  fx  = lscan fa
     And1 ga'  gx  = adjust fx (lscan ga)
\end{code}
}

\framet{Composing scans}{
\begin{code}
instance (LScan g, LScan f, Zip g)  => LScan (g :.: f) where
  lscan (Comp1 gfa) = And1 (Comp1 (zipWith adjust tots' gfa')) tot
   where
     (gfa', tots)    = unzipAnd1 (fmap lscan gfa)
     And1 tots' tot  = lscan tots
 
SPACE
 
adjust :: (Monoid a, Functor t) => a -> t a -> t a
adjust a t = fmap (a `mappend`) t
\end{code}
}

\framet{Scan --- |RPow Pair N5|}{
\vspace{-2ex}
\wfig{4.7in}{figures/lsums-rt5}
}
\framet{Scan --- |LPow Pair N5|}{
\vspace{-4ex}
\wfig{4.75in}{figures/lsums-lt5}
}
\framet{Scan --- |RPow (LVec N3) N3|}{
\vspace{-2ex}
\wfig{4.5in}{figures/lsums-rp3-lv3}
}
\framet{Scan --- |RPow (LPow Pair N2) N3|}{
\vspace{-1ex}
\wfig{4.7in}{figures/lsums-rp3-lp2}
}

\framet{Polynomial evaluation}{

\vspace{-8ex}

%% $$\sum_{0 \le i \le n} a_i \cdot x^i$$

$$a_0 \cdot x^0 + \cdots + a_n \cdot x^n$$

\vspace{2ex}

\pause
\begin{code}
evalPoly coeffs x = coeffs <.> powers x
\end{code}

\pause
\begin{code}
powers = lproducts . pure
\end{code}

\begin{code}
lproducts = underF Product lscan
\end{code}
}

%% powers ::  (LScan f, Applicative f, Num a) =>
%%            a -> And1 f a
%% evalPoly ::  (LScan f, Applicative f, Foldable f, Num a) =>
%%              And1 f a -> a -> a

\framet{Powers --- |RBin N4|}{
\vspace{-5ex}
\wfig{4.75in}{figures/powers-rt4}
}

\framet{Polynomial evaluation --- |RBin N4|}{
\vspace{-1ex}
\wfig{4.75in}{figures/evalPoly-rt4}
}


\framet{Fast Fourier transform}{

DFT:
$$X_k = \sum_{n=0}^{N-1} x_n e^{-\frac{2\pi i}{N} nk}$$

\pause
\vspace{7ex}

FFT for $N = N_1 \cdot N_2$ (Gauss / Cooley-Tukey):
$$X_k = 
    \sum_{n_1=0}^{N_1-1} 
      \left[ e^{-\frac{2\pi i}{N} n_1 k_2 } \right]
      \left( \sum_{n_2=0}^{N_2-1} x_{N_1 n_2 + n_1}  
              e^{-\frac{2\pi i}{N_2} n_2 k_2 } \right)
      e^{-\frac{2\pi i}{N_1} n_1 k_1 }
$$
}
\framet{Fast Fourier transform}{

\pause

\begin{code}
class FFT f where
  type FFO f :: * -> *
  fft :: RealFloat a => f (Complex a) -> FFO f (Complex a)

SPACE

instance FFT Par1 where
  type FFO Par1 = Par1
  fft = id

instance FFT Pair where
  type FFO Pair = Pair
  fft (a :# b) = (a + b) :# (a - b)
\end{code}

}

\framet{FFT --- composition (Gauss / Cooley-Tukey)}{
\begin{code}
instance ... => FFT (g :.: f) where
  type FFO (g :.: f) = FFO f :.: FFO g
  fft = O . traverse fft . twiddle . traverse fft . transpose . unO

SPACE

twiddle :: ... => g (f (Complex a)) -> g (f (Complex a))
twiddle = (zipWith.zipWith) (*) twiddles
  where
    n         = size @(g :.: f)
    twiddles  = fmap powers (powers omega)
    omega     = cis (- 2 * pi / fromIntegral n)
    cis a     = cos a :+ sin a
\end{code}
}

\framet{FFT --- |RBin N3| (``Decimation in time'')}{
\vspace{-4ex}
\wfig{4.75in}{figures/fft-rt3}
}

\framet{FFT --- |LBin N3| (``Decimation in frequency'')}{
\vspace{-1ex}
\wfig{4.75in}{figures/fft-lt3}
}

\framet{Bitonic sort}{

}

\framet{Bitonic sort --- depth 1}{
\vspace{0ex}m
\wfig{4.75in}{figures/bitonic-up-1}
}

\framet{Bitonic sort --- depth 2}{
\vspace{-3ex}
\wfig{4.75in}{figures/bitonic-up-2}
}

\framet{Bitonic sort --- depth 3}{
\vspace{-3ex}
\wfig{4.75in}{figures/bitonic-up-3}
}

\framet{Bitonic sort --- depth 4}{
\vspace{-5ex}
\wfig{4.75in}{figures/bitonic-up-4}
}

\framet{Manual vs automatic placement}{
\pause
\emph{\href{http://whyy.org/cms/radiotimes/2011/02/14/the-eniac-anniversary/}{ENIAC, 1946}}:
\vspace{1ex}
\hspace{2ex} \wpicture{4.3in}{figures/eniac-programming}
}

\framet{Manual vs automatic placement}{
\begin{itemize}\itemsep 4ex
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

\end{document}

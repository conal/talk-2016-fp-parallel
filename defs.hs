-- {-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE DeriveFunctor, DeriveFoldable #-}

import Prelude hiding (foldl,scanl)
import Data.Monoid
import Data.Functor ((<$>))
import Data.Foldable (Foldable(fold))

foldl :: (b -> a -> b) -> b -> [a] -> b

foldl op z []     = z
foldl op z (a:as) = foldl op (z `op` a) as

sumList = sumAcc 0
 where
   sumAcc sum []     = 0
   sumAcc sum (a:as) = sumAcc (sum + a) as

foldList :: Monoid a => [a] -> a
foldList []     = mempty
foldList (a:as) = a `mappend` foldList as

data ListR a = NilR | ConsR a (ListR a) deriving Foldable

data ListL a = NilL | ConsL (ListL a) a deriving Foldable

sumR []     = 0
sumR (a:as) = a + sumR as


data Tree a = L a | B (Tree a) (Tree a) deriving Functor

foldlT :: (b -> a -> b) -> b -> Tree a -> b
foldlT op z (L a)   = z `op` a
foldlT op z (B s t) = foldlT op (foldlT op z s) t

foldT :: Monoid a => Tree a -> a
foldT (L a)   = a
foldT (B s t) = foldT s `mappend` foldT t

sumT (L a)   = a
sumT (B s t) = sumT s + sumT t

sumT' t = sumTAcc 0 t
  where
   sumTAcc acc (L a)   = acc + a
   sumTAcc acc (B s t) = sumTAcc (sumTAcc acc s) t

sumT'' = foldlT (+) 0

foldlT' :: (b -> a -> b) -> Tree a -> b -> b

-- foldlT' op (L a) z   = z `op` a
-- foldlT' op (B s t) z = foldlT' op t (foldlT' op s z)

foldlT' op (L a)   = (`op` a)
foldlT' op (B s t) = foldlT' op t . foldlT' op s

-- foldMap with Endo. foldrT' is a bit neater.


t1 :: Int
t1 = foldl (+) 0 [1..100]

-- scanl op z []     = ([], z)
-- scanl op z (a:as) = (z:bs, ...)
--  where

scan :: Monoid a => Tree a -> (Tree a, a)

-- scan (L a)   = (L mempty, a)
-- scan (B u v) = (B u' v'', uTot `mappend` vTot)
--  where
--    (u',uTot) = scan u
--    (v',vTot) = scan v
--    v'' = fmap (uTot `mappend`) v'

scan (L a)   = (L mempty, a)
scan (B u v) = (B u' (adjust <$> v'), adjust vTot)
 where
   (u',uTot) = scan u
   (v',vTot) = scan v
   adjust = (uTot `mappend`)

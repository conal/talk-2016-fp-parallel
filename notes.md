
# Why functional programming?

I assume most of you have been doing programming for a while.
Why might you care about this paradigm of functional programming?
First, and my focus for today's talk, functional programming makes considerable strides in supporting parallel programming, which is much more difficult in mainstream languages.
Second, it provides powerful, rigorous and practical tools to support correctness.
Finally, functional programming gives a big boost to productivity by capturing design patterns in reusable form.


# What is functional programming?

So, what is this functional programming thing?
Well, the name is an unfortunate one, since it's not centrally about functions, though functions play a very important role.
More accurately, it might be called "value-oriented programming", in contrast to the "action-oriented" nature of mainstream programming languages.
By "value", I mean data that can be examined and produced by computations but not altered.
For instance, an increment computation can examine the number 3 an produce the number 4.
Then if you look at three again, it's still 3, because 3 is immutable -- unalterable.

While most languages can talk about small values such as numbers, functional languages go further to big values, like strings, sequences, streams, trees, images, geometry, functions.

# Finishes a shift that Fortran began


In the late 40s and early 50s,
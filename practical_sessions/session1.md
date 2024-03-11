# Python3 crash-course

## PART 1
*Objectives:*

  * Get familiar with the REPL
  * Edit and run a small program
  * Explore some of the builtin features

*Note:* There are no constraints on the editor you want to use.
Use `vi`, `emacs`, `jupyter`: it's all fine.

### Builtins

The easiest way to test things in `python` is using the REPL.
The `R`ead `E`valuate `P`rint `L`oop is accessible by firing 
the interpreter from the command line.

```
$ python3
>>>
```

Commands are typed directly at the prompt `>>>` and executed with `[enter]`.

```
>>> 2 + 3
5
>>> 'ab' * 5
'ababababab'
>>> 
```

Alternatively you can write your program into a file, save it with extension `.py`
and then run it from the command line:

```
$ python3 myfile.py
```

### 1.1: Explore

Try out some of the commands and data structure you heard of earlier.

Additional modules can be used by importing them (`import`).

```
>>> import math
>>> math.sqrt(625)
25.0
>>> type(_)  # '_' is a special placeholder for 'last result' (REPL only)
<class 'float'>
>>> math.pow(2,3)
8.0
>>> 

```

[Official documentation](https://docs.python.org/3/)

### 1.2: List, tuples and for loops

```
>>> array = [1, 2, 3, 4, 5]
>>> # Python has no built-in array data structure
>>> # instead, it uses "list" which is much more general 
>>> # and can be used as a multidimensional array quite easily.
>>> for element in array:
...    print(element)
...
1
2
3
4
5
>>> for (index, element) in enumerate(array):
...    print(index, element)
...
0 1
1 2
2 3
3 4
4 5
```

Actually, Python has no built-in array data structure. It uses the list data structure, which is much more general and can be used as a multidimensional array quite easily. In addtion, elements in a list can be retrieved in a very concise way. For example, we create a 2d-array with 4 rows. Each row has 3 elements.

```
>>> # 2-dimentions array with 4 rows, 3 columns
>>> twod_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
>>> for index, row in enumerate(twod_array):
...     print("row ", index, ":", row)
... 
row  0 : [1, 2, 3]
row  1 : [4, 5, 6]
row  2 : [7, 8, 9]
row  3 : [10, 11, 12]
>>> print(f"row 1 until row 3: {twod_array[1:3]}")  # f-string!!!
row 1 until row 3: [[4, 5, 6], [7, 8, 9]]
>>> print(f"all rows from row 2: {twod_array[2:]}")
all rows from row 2: [[7, 8, 9], [10, 11, 12]]
>>> twod_array[:2]
[[1, 2, 3], [4, 5, 6]]
>>> twod_array[::2]   # all rows with a step of 2
[[1, 2, 3], [7, 8, 9]]
>>> 

```

A tuple is an unmutable record.

```
>>> t = (1, 2, 'abc')
>>> type(t)
<class 'tuple'>
>>> len(t)
3
>>> t[0]
1
>>> t[0] = 10
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
>>> 
```

You can combine these data structure (warning: danger ahead!)
```
>>> a1 = (1,2,3) # a tuple
>>> a2 = (4,5,6) 
>>> b1 = ['a','b']   # a list
>>> b2 = ['c','d']
>>> t = a1 + a2  # (1, 2, 3, 4, 5, 6)
>>> l = b1 + b2  # ['a', 'b', 'c', 'd']
>>> c = a1 + b1
>>> c = a1 + b1
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can only concatenate tuple (not "list") to tuple
>>> c = (*a1, *b1)  # into a tuple
>>> d = [*a1, *b1]  # into a list
```

### 1.3 Dictionaries

Another useful data structure in Python is a dictionary, which we use to store (key, value) pairs. Here's some example usage of dictionaries:

```
>>> d = {'key1': 'value1', 'key2': 'value2'}
>>> d['key1']
'value1'
>>> 'key1' in d
True
>>> d['key3']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'key3'
>>> d.get('key3', 'default_value')
'default_value'
>>> d.keys()
dict_keys(['key1', 'key2'])
>>> d.values()
dict_values(['value1', 'value2'])
>>> d.items()
dict_items([('key1', 'value1'), ('key2', 'value2')])
>>> 
```

Combining dictionaries:

```
>>> a = {'name': 'marco', 'employer': 'unica', 'office': 215}
>>> b = {'course': 'sdlc', 'date': '11-3-2024'}
>>> c = {**a, **b}
>>> a.update(b)  # WARNING: inplace
>>> a == c
True
>>> 
```

**IMPORTANT**

Many operations are related to "assignment" and "storing" values.

```
>>> a = 1
>>> l = [1,2,3]
>>> l.append(4)
>>> d = {'k': 0}
>>> d['k1'] = 1
```

**Assignment operations never make a copy (reference copy)**

Consider this:
```
>>> a = [1,2,3]
>>> b = a
>>> c = [a, b]
>>> a.append(99)
```
What happened to `b` and `c`?

**But**: many objects are immutable
```
>>> a = 'hello'
>>> b = a
>>> a = 'world'
>>> b
'hello'
```
### 1.4 Functions

In Python, we define a function using the keyword `def`.

```
>>> def square(x):
...    return x*x

>>> square(5)
25
>>>
```

We can apply this function to every element of a list in `3` ways: 
  1. `map`
  2. `for` loop
  3. anonymous function `lambda` (if the function is not too complex) 

```
>>> array = [1, 2, 3, 4, 5]
>>> list(map(square, array))  # map
[1, 4, 9, 16, 25]
>>> [square(x) for x in array] # list comprehension
[1, 4, 9, 16, 25]
>>> list(map(lambda x: x**2, array)) # lambda
[1, 4, 9, 16, 25]
>>> array
[1, 2, 3, 4, 5]
```

**NOTE 1** The act of invoking a higher order functions (e.g., `map`) returns 
a `generator` object with the application of the passed function attached to each
element of the iterable. To print the result we need to materialize it: hence we cast to `list` the result of the `map` function.

We can also put a function B inside a function A (that is, we can have nested functions). In that case, function B is only accessed inside function A (the scope that it's declared). For example:

```
>>> def filter_and_square_primes(arr):
...     # also called "closures"
...     def check_prime(x):
...         if x <= 1: 
...             return False
...         for i in range(2, int(x/2) + 1):
...             if x % i == 0:
...                 return False
...         return True
...     prime_numbers = filter(lambda x: check_prime(x), arr)
...     return map(lambda x: square(x), prime_numbers)
...
>>> array = list(range(25))
>>> map_obj = filter_and_square_primes(array)
>>> list(map_obj)
[4, 9, 25, 49, 121, 169, 289, 361, 529]
>>> 
```

Writing complex functions in the REPL can be a real pain.
Use your favorite editor to edit your code. Save with `.py` extension
and then you can fire the REPL in interactive mode with:

```
# examples/my_file.py
x = 42

def myfun(name):
    print(f"Hello {name}")
  
```

```
$ python3 -i examples/my_file.py 
>>> x
42
>>> myfun('marco')
Hello marco
>>> 

```
All names and definition will be available in the REPL.

### 1.5 Errors

Operations might cause errors.

```
>>> def divide(x,y):
...     return x / y
...
>>> divide(3,0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in divide
ZeroDivisionError: division by zero
>>>
```

There are many types of error: `TypeError`, `ValueError`, `AttributeError`, etc.

**YOU ALWAYS HAVE TO MANAGE THE ERRORS.** The way to do it is by using the following construct.

```
def fail():
    return 1 / 0

try:
    fail()
except ZeroDivisionError:
    print('failed')
```

By doing this we assure the continuation of the program, even in case an error occurs.

Other constructs:

```
try:
    <do something>
except <some error>:
    <manage error>
finally:
    <always execute this no matter what>
```

```
try:
    <do something>
except <some error>:
    <manage error>
else:
    <execute here only if no error occurred>
finally:
    <always execute this no matter what>
```

```
try:
    <do something>
except TypeError:
    <manage TypeError>
except AttributeError:
    <manage AttributeError>
except:
    <manage all other possible errors>
else:
    <execute here only if no error occurred>
finally:
    <always execute this no matter what>
```


### 1.6 Exercise: Write a function

Write a function `check_square_number` to check if an integer number is a square or not.

```
def check_square_number(x):
    ...

```

Save it to a file. Import in into the REPL and experiment.


## PART 2: NUMPY

Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.

[Official documentation](https://numpy.org/doc/1.26/user/index.html#user)

### 2.1 Array
A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. Thanks to the same type property, Numpy has the benefits of locality of reference. Besides, many other Numpy operations are implemented in C, avoiding the general cost of loops in Python, pointer indirection and per-element dynamic type checking. So, the speed of Numpy is often faster than using built-in datastructure of Python. When working with massive data with computationally expensive tasks, you should consider to use Numpy.

The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

We can initialize numpy arrays from nested Python lists, and access elements using square brackets:

```
>>> import numpy as np
>>> arr = np.array([1,2,3])
>>> arr2 = np.array([[1,2,3],[4,5,6]])
>>> arr2.shape
(2, 3)
>>> type(arr)
<class 'numpy.ndarray'>
>>> 
```

### 2.2 Slicing

Similar to Python lists, numpy arrays can be sliced. The different thing is that you must specify a slice for each dimension of the array because arrays may be multidimensional.

```
>>> array = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
>>> array
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
>>> array[0:1]
array([[1, 2, 3, 4]])
>>> array[0,1]
2
>>> array[:2,1:3]
array([[2, 3],
       [6, 7]])
>>> array[1, :]
array([5, 6, 7, 8])
>>> 
```

### 2.3 Boolean indexing

We can use boolean array indexing to check whether each element in the array satisfies a condition or use it to do filtering.

```
>>> bool_idx = array % 3 == 0
>>> bool_idx
array([[False, False,  True, False],
       [False,  True, False, False],
       [ True, False, False,  True]])
>>> array[bool_idx]
array([ 3,  6,  9, 12])
>>> array[(array < 7) & (array % 2 == 0)]
array([2, 4, 6])
>>> 
```

### 2.4 Datatypes

Contrary to standard python containers (`list` or `tuple`), the elements in a `numpy` array have the same type. When constructing arrays, `numpy` tries to guess a datatype when you create an array. However, we can specify the datatype explicitly via an optional argument.

```
>>> x = np.array([1,2,3])
>>> x.dtype
dtype('int64')
>>>
>>> x = np.array([1,2,3], dtype=float)
>>> x.dtype
dtype('float64')
>>> x
array([1., 2., 3.])
>>> 
```

### 2.5 Array math

Similar to Matlab or R, in `numpy`, basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module.

```
>>> x = np.array([[1,2],[3,4]], dtype=np.float64)
>>> y = np.array([[5,6],[7,8]], dtype=np.float64)
>>> x + y
array([[ 6.,  8.],
       [10., 12.]])
>>> np.add(x,y)
array([[ 6.,  8.],
       [10., 12.]])
>>> 
```
**Unlike** MATLAB, `*` is elementwise multiplication, **not** matrix multiplication.

```
>>> x * y
array([[ 5., 12.],
       [21., 32.]])
>>> np.multiply(x,y)
array([[ 5., 12.],
       [21., 32.]])
```

To multiply two matrices we use `dot`.

```
>>> x.dot(y)
array([[19., 22.],
       [43., 50.]])
>>> np.dot(x,y)
array([[19., 22.],
       [43., 50.]])
>>> 
```

Elementwise square root of a matrix:

```
>>> np.sqrt(x)
array([[1.        , 1.41421356],
       [1.73205081, 2.        ]])
```

Some more examples:

```
>>> v = np.array([9,10])
>>> w = np.array([11,12])
>>> v.dot(w)  # inner product same as np.dot(v,w)
219
>>> x.shape
(2, 2)
>>> v.shape
(2,)
>>> np.dot(x,v)
array([29., 67.])
```

And some useful functions:

```
>>> np.sum(x)  # sum all elements
10.0 
>>> np.sum(x, axis=0)  # sum all columns
array([4., 6.])
>>> np.sum(x, axis=1)  # sum all rows
array([3., 7.])
>>> x.T
array([[1., 3.],
       [2., 4.]])
>>> 
```

### 2.6 Exercise: Boolean indexing

Given:

```
import numpy as np

arr = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
```

Print all the odd elements using a boolean index.

### 2.7 Exercise: slicing

Print elementwise multiplication of the first row and the last row of the previous array `arr` using numpy's functions. Print the inner product of these two rows.

# PART 3: Object oriented programming

You can define objects as instances of a `class`.

```
# examples/player.py
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move_right(self, dx):
        self.x += dx

    def move_left(self, dx):
        self.x -= dx
```

```
$ python3 -i examples/player.py
>>> p = Player(0,0)
>>> type(p)
<class '__main__.Player'>
>>> p.move_right(5)
>>> p.x
5
>>> p.__dict__   # every object has a dictionary representation
{'x': 5, 'y': 0}
```

### 3.1 Inheritance

Experiment with file: `examples/animals.py`.

Inheritance models what’s called an is a relationship. This means that when you have a `Derived` class that inherits from a `Base` class, you’ve created a relationship where `Derived` is a **specialized** version of `Base`.

   * Classes that inherit from another are called derived classes, subclasses, or subtypes.
   * Classes from which other classes are derived are called base classes or super classes.
   * A derived class is said to derive, inherit, or extend a base class.
  
Data and behaviour are coupled within the definition: `state`.

Type relationship:

```
>>> d = Dog('Rex')
>>> type(d)
<class '__main__.Dog'>
>>> isinstance(d, Animal)
True
```

# PART 4: Functions and Functional programming

### 4.1 Functions
Functions are a basic building block.
   * Top-level in a module
   * Methods of a class
   * Everything else
  
Best practices:
   * self-contained
   * only operate on arguments
   * same result == same arguments
   * avoid (hidden) side-effects
   * simple
   * predictable
  
```
def read_data(filename, debug=False):  
    '''
    This is a docstring. Every function should have one.
    Helper documentation.
    @param filename: str, name of the file
    @param debug: bool, log to stderr
    @return something
    '''
    ...
```

```
>>> d = read_data('data.csv')
>>> e = read_data('data.csv', debug=True)  # yes
>>> e = read_data('data.csv', True)    # also yes, but don't do it!!
```

Force the use of keyword argument:
```
def read_data(filename, *, debug=False):  # everything after '*' must be given as keyword args
    ...
```

```
>>> read_data('data.csv', True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: read_data() takes 1 positional argument but 2 were given
```

**IMPORTANT** avoid problems: don't use mutable values for default arguments.

A function always return something.

```
def divide(x, y):
    # return directly the result 
    return x / y

def divide(x, y):
    # return a tuple
    quot = x // y
    remainder = x % y
    return (quot, remainder)
```

Sometimes a function returns an optional value (commonly, `None`).

```
>>> import re
>>> m = re.search(r'\d{3}', 'abc12')
>>> print(m)
None
>>> m = re.search(r'\d{3}', 'abc123')
>>> print(m)
<re.Match object; span=(3, 6), match='123'>
```

**NOTE** instead of returning `None`, raise an exception.

### 4.2 Concurrency

We can run function concurrently (threads).

```
def func1():
    ...

def func2():
    ...

from threading import Thread
t1 = Thread(target=func1)
t1.start()
t2 = Thread(target=func2)
t2.start()
```

`Future` represents a future results that has to be computed.

```
from concurrent.futures import Future

def func1(x, y, fut):
    # do something
    fut.set_result(x ** y)

def calling_function():
    fut = Future()
    threading.Thread(target=func1, args=(2,3,fut)).start()
    result = fut.result()   # waiting here...
    return result
```
This pattern is used everywhere (multiprocessing, threads, async, ...)

  * file: `examples/thread_safety.py`
  
### 4.3 Functional programming

Three features:
   * only functions
   * no side effects (pure functions)
   * higher order functions 
  
Functions are 1st class citizens, which means: (a) functions can accept functions as input, 
and (b) functions can return functions as results.

```
>>> def build_quadratic_function(a,b,c):
...    return lambda x: a*x**2 + b*x + c
...
>>> f = build_quadratic_function(3, -5, 2)
>>> f
<function build_quadratic_function.<locals>.<lambda> at 0x102b935b0>
>>> f(0)
2
>>> f(-5)
102
```

`lambda` creates an anonymous function on the spot. Can contain only a single expression. No control flow, exceptions, etc.


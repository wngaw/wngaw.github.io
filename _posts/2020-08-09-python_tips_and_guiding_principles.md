---
layout: post
comments: true
date: 2020-07-25 12:00:00
title: Python Tips and Guiding Principles
tags: scripting-language
image: "/images/python.png"
---

If it runs, its fine... right? Not really, because as programmers we should strive to write good quality code in order to make our life easier. In this post, I will first showcase some standard practices then move on to some controversial points which some might disagree.

{: class="table-of-content"}
* TOC
{:toc}

## Standard Practice

### Write docstring and comments

Whenever you write a complicated function, always remember to write docstring. However, if the function only contains a few lines, then docstring can arguably be skipped. In addition to docstring, when you have obscure logics in your code, remember to write comments to remind your future self why you are doing that.

Positive example:

```python
def power(base, exponent):
    """
    A function that takes the base to the power of exponent and returns the result

    Parameters
    __________
    base: integer
        The base
    exponent: integer
        Exponent of a base

    Returns
    _______
    integer_3: integer
        Result of base to the power of exponent
    """
    result = base ** exponent  # Taking base to the power of exponent using Python native operator
    return result
```

Negative example:

```python
def power(base, exponent):
    result = base ** exponent
    return result
```

### Include your data type in your variable name

This is a very simple but often overlooked aspect. Simply give a prefix to your variable to tell you what data type it is. If you have a few hundred lines of cone, you might forget what data type a certain variable is and end up scrolling up and down to check the data type.

Positive example:

```python
import pandas as pd
df_data = pd.read_csv('data/data.csv')
...
df_data_transformed = transform(df_data)
```

Negative example:

```python
import pandas as pd
data = pd.read_Csv('data/data.csv')
...
data_transformed = transform(data)
```

### Be mindful of namespace pollution

Always try to minimize the number of names to be added to your program's namespace. When you import a package, be explicit about the submodule that you are importing, don't just import everything. Be very careful about creating any global variables as the namespace might collide with another variable in your program and cause a bug.

Positive example:

```python
from sklearn import linear model, metrics, model_selection
```

Negative example:

```python
from sklearn import *
```

### Have a configuration file to store variables that you will be changing frequently

If you have many constants that are reused in several places, and that you will change them frequently during development or when switching from development to production, then you might want to store all these constants in a configuration file so that you do not have to look through all your source code and change them one by one.

Positive example:

```python
# config.py

max_depth = 3

# main.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import config

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=config.max_depth)
clf.fit(X, y)
````

Negative example:

```python
# main.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=3)
clf.fit(X, y)
````

### Use auto code formatter to reformat your code to PEP 8 standard

Focus your time and mental engergy on more important things rather the manual code formatting. Just write your Python code in whatever format that runs, then simply use an auto code formatter to automatically change all your code to conform to PEP 8.

One of the auto code formatter that you can use is [black](https://black.readthedocs.io/en/stable/index.html).

Installation:

```sh
pip install black
```

To auto format folder or file:

```sh
black $folder_name
black $file_name
```

### Print out your process ID, time taken, and memory usage

During development and production, there are few standard information that you will need to know such as how long the job is expected to run, how much resources it is taking, and the Python process id of the job (in case you want to kill it). Therefore, it is a good practice to just include a few standard commands in your main script to output this information.

Positive example:

```python
# udf.py

import psutil
import os

def run_check_memory():
    """Check memory resources taken for process
    """
    process = psutil.Process(os.getpid())
    memory = round(process.memory_info().rss / float(2 ** 20), 2)
    memory_percent = round(process.memory_percent(memtype="rss"), 2)

    print(f"Memory usage in MB: {memory}")
    print(f"Memory usage in Percent: {memory_percent}%")

# main.py

from time import time
import os
import udf

def main()
    ...

if __name__ == "__main__":
    print("Process ID: ", os.getpid())
    t0 = time()
    main()
    udf.run_check_memory()
    print("Total time taken: {:.1f} minutes".format((time() - t0) / 60))
```

Negative example:

```python
# main.py

def main()
    ...

if __name__ == "__main__":
    main()
```

Note: There are other kinds of memory profiler which can be used. For example, you can use [memory-profiler](https://pypi.org/project/memory-profiler/) to sample memory usage at a fixed interval if you know in advance roughly how much time your process will take.

### Go beyond procedural programming and learn how to use functional and object-oriented programming

Procedural programming usually contains a sequence of commands to be carried out. This is by far the most widely known paradigm. In fact, many programmers are familiar only with this one. It is also a natural extension of how humans work: we do things sequentially one after another in our day-to-day routine.

Functional programming are more than just procesdures. Functions are used to define relationship between two or more items and are treated as first-class citizens in functional programming. High-order functions are able to receive other functions as arguments and return functions as outputs. If you ever find yourself copying and pasting the same code in different scripts, consider writing a function for it so that you can simply call the function in your script.

Object-oriented programming (OOP) is a paradigm that is based on the concept of "objects". Each object may contain attributes and methods. The methods behave like functions and are only accessible by the declared object. Usually we use OOP when we have logics that we want to isolate and iterate for improvement. For example, in data science, when a model requires many standard preprocessing steps along with the model training and inference, we can isolate all of these steps into a single object and call it from the main script.

Positive example:

```python
# calculation.py

class Calculation:
    def __init__(self, constant_1, constant_2):
        self.constant_1 = constant_1
        self.constant_2 = constant_2

    def sum(self, integer):
        result = (self.constant_1 + self.constant_2) + integer
        return result

    def multiply(self, integer):
        result = (self.constant_1 + self.constant_2) * integer
        return result

# main.py

from calculation import Calculation


def main():
    ...
    constant_1 = 5
    constant_2 = 10
    calculation = Calculation(constant_1, constant_2)
    sum_result = calculation.sum(3)
    multiply_result = calculation.multiply(4)


if __name__ == "__main__":
    main()
```

Negative example:

```python
# main.py

constant_1 = 5
constant_2 = 10
sum_result = (constant_1 + constant_2) + 3
multiply_result = (constant_1 + constant_2) * 4
````

Note: The above example is over simplistic. In reality, when you have numerous steps and complicated logics in your object, isolating these logics will make your code alot more readable.

### Use multi-processing if you are facing CPU bound processes

If you are trying to compute something and that calculation is done on a single CPU, while at the same time that calculation can be done independently for each row (i.e. the result of each row does not depend on the other rows), then you can consider using multi-processing to speed up the calculation. This assumes that you have several other CPU cores that are lying idle during that calculation.

In multi-processing, your data will be split into multiple chunks and calculation is done in parallel using multiple CPU. The result of each chunk is then combined at the end to give you the final result.


Positive example:

```python
# udf.py

import numpy as np
import pandas as pd
import multiprocessing as mp

def apply_function_on_df_multi_processing(input_df, function):
    """Apply a function to dataframe via multi-processing. Use this if your process is CPU bound.
    Input:
        input_df: input Pandas dataframe
        function: function you want to apply
    Returns:
        output_df: combined dataframe after applying the function
    """
    num_cpu = mp.cpu_count()

    with mp.Pool(num_cpu) as pool:
        iterable = iter(np.array_split(input_df, num_cpu))
        list_df_chunks = pool.map(function, iterable)
        output_df = pd.concat(list_df_chunks, axis=0)

    return output_df

# main.py

import udf

def computation_function(input_df):
    ...
    return output_df


def main():
    ...
    output_df = udf.apply_function_on_df_multi_processing(input_df, computation_function)
    ...


if __name__ == "__main__":
    main()
```

Negative example:

```python
def computation_function(input_df):
    ...
    return output_df


def main():
    ...
    output_df = computation_function(input_df)
    ...


if __name__ == "__main__":
    main()
```

### Use multi-threading if you are facing I/O bound processes

This is different from multi-processing. In this case, you might be trying to perform a web request. Therefore your task is I/O bound and not CPU bound. This means that you will not get any speedup even if you perform the task in parallel using multiple CPU. What you need over here is multi-threading.

Positive example:

```python
# udf.py

import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing.dummy import Pool

def apply_function_on_df_multi_processing(input_df, function):
    """Apply a function to dataframe via multi-processing. Use this if your process is CPU bound.
    Input:
        input_df: input Pandas dataframe
        function: function you want to apply
    Returns:
        output_df: combined dataframe after applying the function
    """
    num_cpu = mp.cpu_count()

    with Pool(num_cpu) as pool:
        iterable = iter(np.array_split(input_df, num_cpu))
        list_df_chunks = pool.map(function, iterable)
        output_df = pd.concat(list_df_chunks, axis=0)

    return output_df

# main.py

import udf
import requests

def computation_function(input_df):
    ...
    response = request.get(url)
    score = calculation(response)
    output_df = combine_score(score)
    return output_df


def main():
    ...
    output_df = udf.apply_function_on_df_multi_threading(input_df, computation_function)
    ...


if __name__ == "__main__":
    main()
```

Note: The change from multi-processing to multi-threading is just a switch from `multi-processing.Pool` to `multi-processing.dummy.Pool`.

### Understand the meaning of underscore in Python

In Python, underscore carries several special meanings. However, I will just cover two of the most widely used ones.

1) Underscore is used to give special meanings to variables and functions:

- Single leading under score: indicates private variables, functions, methods, and classes.
- Single trailing under score: used to avoid conflict with Python keywords or built-ins variables.

2) Underscore is used to ignore values:

```python
# Ignore a value when unpacking
x, _, y = (1, 2, 3) # x = 1, y = 3

# Ignore the index
for _ in range(10):
    do_something()
```

## Controversial

### Reserve Jupyter Notebook only for tasks that require high level of debugging

Jupyter Notebook is an interactive web tool for people to write their codes. Often this is one of the first few tools that most people started to use when learning Python. This is because the interactive nature of it allows developers to understand every single line of code in granular detail while providing immediate feedback for new lines of code.

However, there is a tradeoff for this level of granular detail and immediate feedback - the code written is not of production quality. To be of prodution quality, code needs to be extremely readable and abstraction should be applied whenever possible in order to make debugging easier. This is because when a project is in production and something fails, often the orignal developer might not even be the one trying to debug the code; it might be done by an engineer in the production team. Therefore, you cannot dump a few hundred lines of code in one single file Jupyter Notebook and expect others to figure out what is happening.

Once a Python developer has reach an intermediate level of proficiency, he/she should reserve Jupyter Notebook for only tasks that require high level of debugging such as plotting of charts, testing of regular expression, or creating of new algorithms that are not found in the open-source world.

### Be mindful of how much syntactic sugar you are using

Syntactic sugar is a kind of syntax that lets you write expression in simpler and shorter forms.

Here are some commonly used syntactic sugar:

```python
# Addition

## Without syntactic sugar
a = a + b
## With syntactic sugar
a += b

# For loops

## Without syntactic sugar
list_things = []
for item in items:
    if condition_based_on(item):
        list_things.append("something with " + item)

## With syntactic sugar
list_things = ["something with " + item for item in items if condition_based_on(item)]
```

As you can see, syntactic sugar helps to compress complicated logics into a single line at the cost of readability. If you find your syntactic sugar having more than two or three variables, just ask an intermediate python developer if your code is still readible. Otherwise, consider removing the syntactic sugar.

However, if syntactic sugar helps to improve your code's time or space complexity, and you are facing resource constraints, then that itself is a valid reason to use syntactic sugar even if it compromises readibility.

## Comments

At the end of the day, when you write better quality code, others will not need to ask you what it means, and you end up having more time to focus on things that matter.

The purpose of this post is to cover the most commonly seen and useful tips to supercharge your journey as a Python developer. This post is not meant to provide a 100% coverage of all the quirks of Python which is available in most standard Python textbooks.

## Reference

[1] Wladston Ferreira Filho [Computer Science Distilled: Learn the Art of Solving Computational Problems](https://www.amazon.sg/Computer-Science-Distilled-Computational-Problems/dp/0997316020)

[2] mingrammer [Hackernoon: Understanding the underscore( _ ) of Python](https://hackernoon.com/understanding-the-underscore-of-python-309d1a029edc)

---

Thank you for reading! See you in the next post!

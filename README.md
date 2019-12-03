# Section 3: Numpy, Pandas, Matplotlib 🔥

## Anaconda

Anaconda is a distribution of software that comes with `conda`, Python, and over 150 scientific packages and their dependencies. The application `conda` is a package and environment manager.

- To install package with specified version number: `conda install numpy=1.10`
- To uninstall a package: `conda remove package_name`
- To update all packages within an environment: `conda update --all`
- To list installed packages: `conda list`
- To search for a package name: `conda search *search_term*`
- To create an environment: `conda create -n env_name list of packages` / `conda create -n py2 python=2`
- To enter an environment: `conda activate my_env`
- To leave an environment: `conda deactivate`
- To save an environment's packages, dependencies and versions to YAML file: `conda env export > environment.yaml`. This file can now be shared and others will be able to create the same environment you used for the project
- To create an environment from an environment file: `conda env create -f environment.yaml`
- To list all created environments: `conda env list`
- To remove an environment: `conda env remove -n env_name`

#### Sharing environments

When sharing on GitHub, it's good practice to commit an environment file to make it easier for people to install all the dependencies.
Ideally also include a pip requirements.txt file using `pip freeze` for people not using conda.

## Jupyter Notebooks

- The notebook is a web application that allows you to combine explanatory text, math equations, code, and visualizations all in one easily sharable document.
- Notebooks have quickly become an essential tool when working with data. You'll find them being used for data cleaning and exploration, visualization, machine learning, and big data analysis.
- Notebooks are just big JSON files with the extension `.ipynb`.
- Notebooks are also rendered automatically on GitHub.
- By default, the notebook server runs at http://localhost:8888.
- You should consider installing Notebook Conda to help manage your environments: `conda install nb_conda`. You will be able to access any of your conda environments when choosing a kernel.
- Refer to `working-with-code-cells.ipynb` for useful tips when working with notebooks.
- [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) and [LaTeX Tutorial](https://www.latex-tutorial.com/)
- If you want to time how long it takes for a whole cell to run, you’d use `%%timeit`.
- To render figures directly in the notebook: `%matplotlib inline`
- To turn on the interactive debugger and inspect variables in the current namespace: `%pdb`
- Other [magic commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html)
- To convert notebook into html: `jupyter nbconvert --to html notebook.ipynb`
- To create a slideshow from the notebook file and see it in the browser: `jupyter nbconvert notebook.ipynb --to slides --post serve`

![How Jupyter nootebooks work](img.png)

## Numpy

- Check out `numpy arrays.ipynb`

* NumPy stands for Numerical Python and it's a fundamental package for scientific computing in Python. Built on top of C.
* Check out version with `conda list numpy`
* NumPy arrays are several orders of magnitude much faster than regular python lists. This speed comes from the nature of NumPy arrays being memory-efficient and from optimized algorithms used by NumPy for doing arithmetic, statistical, and linear algebra operations.
* Also supports multidimensional arrays to represent vectors and matrices. It is optimized for matrix operations, largely used in ML algorithms.
* Large number of optimized built-in complex mathematical functions.
* Pandas are built on top of Numpy.
* Numpy arrays ndarray -> n-dimensional array. An ndarray is a multidimensional array of elements all of the same type (string or number).
* When we create an ndarray with both floats and integers, NumPy assigns its elements a float64 dtype. This is called **upcasting**. Since all the elements of an ndarray must be of the same type, in this case NumPy upcasts the integers to floats in order to avoid losing precision in numerical computations.
* Specifying the data type of the ndarray can be useful in cases when you don't want NumPy to accidentally choose the wrong data type, or when you only need certain amount of precision in your calculations and you want to save memory.
* Once you create an ndarray, you may want to save it to a file to be read later or to be used by another program: `np.save(filename, array_name)`. To later load it into the notebook: `np.load('array_name.npy')`
* `np.zeros((3,4), dtype=int)`
* `np.ones((3,4), dtype=int)`
* `np.full((4,3), 5)`
* `np.eye(3)`
* `np.diag([10,20,30,40])`
* `np.arange(start, stop, step)`, start is inclusive, stop is exclusive, step is third argument
* `np.linspace(start, stop, N)`, start and stop are both inclusive. Passing in endpoint=False it will exclude the endpoint.
* `np.reshape(ndarray, new_shape)`
* `np.random.random(shape` or `np.random.randint(start, end, shape)`
* NumPy allows you create random ndarrays with numbers drawn from various probability distributions such as: `np.random.normal(mean, standard_deviation, size=shape)`
* To access elements: `array_name[row][column]` or `array_name[row, column]`
* To delete elements: `np.delete(ndarray, elements, axis)`. For rank 2 ndarrays, axis = 0 is used to select rows, and axis = 1 is used to select columns.
* To add elements: `np.append(ndarray, elements, axis)`.
* To insert elements: `np.insert(ndarray, index, elements, axis)`
* Slicing does not create a copy, need to call the copy method.
* To get only unique values: `np.unique(ndarray)`
* To grab diagonally below main diagonal: `np.diag(nparray, k=-1)`
* Sorting: When `np.sort()` is used as a function, it sorts the ndrrays out of place, meaning, that it doesn't change the original ndarray being sorted. However, when you use sort as a method, `ndarray.sort()` sorts the ndarray in place, meaning, that the original array will be changed to the sorted one.
* **Broadcasting** is the term used to describe how NumPy handles element-wise arithmetic operations with ndarrays of different shapes. For example, broadcasting is used implicitly when doing arithmetic operations between scalars and ndarrays. When performing element-wise operations, the shapes of the ndarrays being operated on, must have the same shape or be broadcastable.

- **It is important to remember that one big difference between Python lists and ndarrays, is that unlike Python lists, all the elements of an ndarray must be of the same type.**

## Pandas

* [Official docs](https://pandas.pydata.org/pandas-docs/stable/)
* Check out notebook pandas.ipynb
* Pandas is a package for data manipulation and analysis in Python. Built on top of Numpy.
* Pandas **Series** and Pandas **DataFrames** allow us to work with *labeled* and *relational* data in an easy and intuitive manner
* It often happens that large datasets don’t come ready to be fed into your learning algorithms, will often have missing values, outliers, incorrect values, etc… Having data with a lot of missing or bad values, for example, is not going to allow your machine learning algorithms to perform well. Therefore, one very important step in machine learning is to look at your data first and make sure it is well suited for your training algorithm by doing some basic data analysis.

**Series**

* A Pandas series is a one-dimensional array-like object that can hold many data types, such as numbers or strings. Unlike Numpy ndarrays, you can assign index label to each element and it can hold data of different types.
* Since we can access elements in various ways, in order to remove any ambiguity to whether we are referring to an index label or numerical index, Pandas Series have two attributes, `.loc` and `.iloc` to explicitly state what we mean. The attribute `.loc` stands for location and it is used to explicitly state that we are using a labeled index. Similarly, the attribute `.iloc` stands for integer location and it is used to explicitly state that we are using a numerical index.
* Series are mutable. Delete items in place: `pd.drop(index, inplace=True)`.

**DataFrames**
* We see that DataFrames are displayed in tabular form, much like an Excel spreadsheet. Also notice that the row labels of the DataFrame are built from the union of the index labels of the two Pandas Series we used to construct the dictionary. And the column labels of the DataFrame are taken from the keys of the dictionary.
* Another thing to notice is that the columns are arranged alphabetically and not in the order given in the dictionary. We will see later that this won't happen when we load data into a DataFrame from a data file.
* NaN stands for Not a Number, and is Pandas way of indicating that it doesn't have a value for that particular row and column index. Whenever a DataFrame is created, if a particular column doesn't have values for a particular row index, Pandas will put a NaN value there. If we were to feed this data into a machine learning algorithm we will have to remove these NaN values first.
* To access `dataframe[column][row]`


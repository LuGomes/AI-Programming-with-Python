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

**Matplotlib and Seaborn**

- Bar charts for qualitative variables
- Histograms for quantitative variables

In short, a **tidy** dataset is a tabular dataset where:

1. each variable is a column
2. each observation is a row
3. each type of observational unit is a table

In practice, you may need to perform tidying work before exploration. You should be comfortable with reshaping your data or perform transformations to split or combine features in your data, resulting in new data columns. This work should be performed in the wrangling stage of the data analysis process.

- Bar Charts
A bar chart is used to depict the distribution of a categorical variable. In a bar chart, each level of the categorical variable is depicted with a bar, whose height indicates the frequency of data points that take on that level.
`sns.countplot(data, x | y, color, order)`. `color_palette()` returns a list of RBG tuples.

- Pie charts
- Interest in relative frequencies. Areas should represent parts of a whole.
- Limit the number of slices plotted.
- Donut plot by setting the wedge width to number smaller than 1.

- Histograms
- Used to plot the distribution of a numeric variable. It's the quantitative version of the bar chart. However, rather than plot one bar for each unique numeric value, values are grouped into continuous bins, and one bar for each bin is plotted depicting the number.
- Use `plt.hist(data=df,x='num_var')` or `sb.distplot()`. The latter creates more bins by default and draws the kernel density estimate (KDE) (area under the curve sum to 1). `kde=False` removes it and equals the Matplotlib one. You can set custom bin edge values with arange.

```
bin_edges = np.arange(0, df['num_var'].max()+1, 1)
plt.hist(data = df, x = 'num_var', bins = bin_edges)
```

Figures, Axes and Subplots

```
plt.figure(figsize = [10, 5]) # larger figure size for subplots
plt.subplot(1, 2, 1) # 1 row, 2 cols, subplot 1
bin_edges = np.arange(0, df['num_var'].max()+4, 4)
plt.hist(data = df, x = 'num_var', bins = bin_edges)

plt.subplot(1, 2, 2) # 1 row, 2 cols, subplot 2
bin_edges = np.arange(0, df['num_var'].max()+1/4, 1/4)
plt.hist(data = df, x = 'num_var', bins = bin_edges)
```
`axes = fig.get_axes()` to get all axes in a figure and `ax=plt.gca()` to get the current axes.
```
fig, axes = plt.subplots(3, 4) # grid of 3x4 subplots
axes = axes.flatten() # reshape from 3x4 array into 12-element vector
for i in range(12):
    plt.sca(axes[i]) # set the current Axes
    plt.text(0.5, 0.5, i+1) # print conventional subplot index number to middle of Axes
```
As you create your plots and perform your exploration, make sure that you pay attention to what the plots tell you that go beyond just the basic descriptive statistics. Note any aspects of the data like number of modes and skew, and note the presence of outliers in the data for further investigation.

`plt.xlim(lower,upper)` to change axes limits.

```
plt.figure(figsize = [10, 5])

# histogram on left: full data
plt.subplot(1, 2, 1)
bin_edges = np.arange(0, df['skew_var'].max()+2.5, 2.5)
plt.hist(data = df, x = 'skew_var', bins = bin_edges)

# histogram on right: focus in on bulk of data < 35
plt.subplot(1, 2, 2)
bin_edges = np.arange(0, 35+1, 1)
plt.hist(data = df, x = 'skew_var', bins = bin_edges)
plt.xlim(0, 35) # could also be called as plt.xlim((0, 35))
```

Certain data distributions will find themselves amenable to scale transformations. The most common example of this is data that follows an approximately log-normal distribution. This is data that, in their natural units, can look highly skewed: lots of points with low values, with a very long tail of data points with large values. However, after applying a logarithmic transform to the data, the data will follow a normal distribution.

```
plt.figure(figsize = [10, 5])

# left histogram: data plotted in natural units
plt.subplot(1, 2, 1)
bin_edges = np.arange(0, data.max()+100, 100)
plt.hist(data, bins = bin_edges)
plt.xlabel('values')

# right histogram: data plotted after direct log transformation
plt.subplot(1, 2, 2)
log_data = np.log10(data) # direct data transform
log_bin_edges = np.arange(0.8, log_data.max()+0.1, 0.1)
plt.hist(log_data, bins = log_bin_edges)
plt.xlabel('log(values)')
```
Or better yet, do the scale transform to keep the natural labels of the x-axis:
```
bin_edges = 10 ** np.arange(0.8, np.log10(data.max())+0.1, 0.1)
plt.hist(data, bins = bin_edges)
plt.xscale('log')
tick_locs = [10, 30, 100, 300, 1000, 3000]
plt.xticks(tick_locs, tick_locs)
```
**Bivariate Exploration**

If we want to inspect the relationship between two numeric (quantitative) variables, the standard choice of plot is the scatterplot!

Pearson correlation coefficient - r (-1 to 1) to capture linear relationships
`plt.scatter(data,x,y)` or `sb.regplotdata,x,y)`.
Seaborn's regplot function combines scatterplot creation with regression function fitting.
By default, the regression function is linear, and includes a shaded confidence region for the regression estimate.

```
def log_trans(x, inverse = False):
    if not inverse:
        return np.log10(x)
    else:
        return np.power(10, x)

sb.regplot(df['num_var1'], df['num_var2'].apply(log_trans))
tick_locs = [10, 20, 50, 100, 200, 500]
plt.yticks(log_trans(tick_locs), tick_locs)
```
Overplotting solutions: sampling, transparency and jitter (random noise to position of each point).
Transparency can be added to a scatter call by adding the "alpha" parameter set to a value between 0 (fully transparent, not visible) and 1 (fully opaque). As an alternative or companion to transparency, we can also add jitter to move the position of each point slightly from its true value. This is not a direct option in matplotlib's scatter function, but is a built-in option with seaborn's regplot function. x- and y- jitter can be added independently, and won't affect the fit of any regression function.

```
sb.regplot(data = df, x = 'disc_var1', y = 'disc_var2', fit_reg = False,
           x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/3})
```

A heat map is a 2-d version of the histogram that can be used as an alternative to a scatterplot. Like a scatterplot, the values of the two numeric variables to be plotted are placed on the plot axes. Similar to a histogram, the plotting area is divided into a grid and the number of points in each grid rectangle is added up. Since there won't be room for bar heights, counts are indicated instead by grid cell color.

```
plt.figure(figsize = [12, 5])

# left plot: scatterplot of discrete data with jitter and transparency
plt.subplot(1, 2, 1)
sb.regplot(data = df, x = 'disc_var1', y = 'disc_var2', fit_reg = False,
           x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/3})

# right plot: heat map with bin edges between values
plt.subplot(1, 2, 2)
bins_x = np.arange(0.5, 10.5+1, 1)
bins_y = np.arange(-0.5, 10.5+1, 1)
plt.hist2d(data = df, x = 'disc_var1', y = 'disc_var2',
           bins = [bins_x, bins_y])
plt.colorbar();
```
Furthermore, I would like to distinguish cells with zero counts from those with non-zero counts. The "cmin" parameter specifies the minimum value in a cell before it will be plotted.

```
# hist2d returns a number of different variables, including an array of counts
bins_x = np.arange(0.5, 10.5+1, 1)
bins_y = np.arange(-0.5, 10.5+1, 1)
h2d = plt.hist2d(data = df, x = 'disc_var1', y = 'disc_var2',
               bins = [bins_x, bins_y], cmap = 'viridis_r', cmin = 0.5)
counts = h2d[0]

# loop through the cell counts and add text annotations for each
for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        c = counts[i,j]
        if c >= 7: # increase visibility on darkest cells
            plt.text(bins_x[i]+0.5, bins_y[j]+0.5, int(c),
                     ha = 'center', va = 'center', color = 'white')
        elif c > 0:
            plt.text(bins_x[i]+0.5, bins_y[j]+0.5, int(c),
                     ha = 'center', va = 'center', color = 'black')
```

There are a few ways of plotting the relationship between one quantitative and one qualitative variable, that demonstrate the data at different levels of abstraction. The violin plot is on the lower level of abstraction. For each level of the categorical variable, a distribution of the values on the numeric variable is plotted. The distribution is plotted as a kernel density estimate, something like a smoothed histogram.
`sb.violinplot(data = df, x = 'cat_var', y = 'num_var')`

A box plot is another way of showing the relationship between a numeric variable and a categorical variable. Compared to the violin plot, the box plot leans more on summarization of the data, primarily just reporting a set of descriptive statistics for the numeric values on each categorical level.

```
plt.figure(figsize = [10, 5])
base_color = sb.color_palette()[0]

# left plot: violin plot
plt.subplot(1, 2, 1)
ax1 = sb.violinplot(data = df, x = 'cat_var', y = 'num_var', color = base_color)

# right plot: box plot
plt.subplot(1, 2, 2)
sb.boxplot(data = df, x = 'cat_var', y = 'num_var', color = base_color)
plt.ylim(ax1.get_ylim()) # set y-axis limits to be same as left plot
```
The inner boxes and lines in the violin plot match up with the boxes and whiskers in the box plot. In a box plot, the central line in the box indicates the median of the distribution, while the top and bottom of the box represent the third and first quartiles of the data, respectively. Thus, the height of the box is the interquartile range (IQR). From the top and bottom of the box, the whiskers indicate the range from the first or third quartiles to the minimum or maximum value in the distribution. Typically, a maximum range is set on whisker length; by default this is 1.5 times the IQR. For the Gamma level, there are points below the lower whisker that indicate individual outlier points that are more than 1.5 times the IQR below the first quartile.

To depict the relationship between two categorical variables, we can extend the univariate bar chart seen in the previous lesson into a clustered bar chart. Like a standard bar chart, we still want to depict the count of data points in each group, but each group is now a combination of labels on two variables. So we want to organize the bars into an order that makes the plot easy to interpret. In a clustered bar chart, bars are organized into clusters based on levels of the first variable, and then bars are ordered consistently across the second variable within each cluster. This is easiest to see with an example, using seaborn's countplot function. To take the plot from univariate to bivariate, we add the second variable to be plotted under the "hue" argument:

`sb.countplot(data = df, x = 'cat_var1', hue = 'cat_var2')`

One alternative way of depicting the relationship between two categorical variables is through a heat map. Heat maps were introduced earlier as the 2-d version of a histogram; here, we're using them as the 2-d version of a bar chart. The seaborn function heatmap is at home with this type of heat map implementation, but the input arguments are unlike most of the visualization functions that have been introduced in this course. Instead of providing the original dataframe, we need to summarize the counts into a matrix that will then be plotted.

```
ct_counts = df.groupby(['cat_var1', 'cat_var2']).size()
ct_counts = ct_counts.reset_index('count')
ct_counts = ct_counts.pivot(index = 'cat_var2', columns = 'cat_var1', values = 'count')
sb.heatmap(ct_counts, annot = True, fmt = 'd')
```

**Faceting**
In faceting, the data is divided into disjoint subsets, most often by different levels of a categorical variable. For each of these subsets of the data, the same plot type is rendered on other variables. Faceting is a way of comparing distributions or relationships across levels of additional variables, especially when there are three or more variables of interest overall. While faceting is most useful in multivariate visualization, it is still valuable to introduce the technique here in our discussion of bivariate plots.

Seaborn's FacetGrid class facilitates the creation of faceted plots. There are two steps involved in creating a faceted plot. First, we need to create an instance of the FacetGrid object and specify the feature we want to facet by ("cat_var" in our example). Then we use the map method on the FacetGrid object to specify the plot type and variable(s) that will be plotted in each subset (in this case, histogram on "num_var").

In the `map` call, just set the plotting function and variable to be plotted as positional arguments. Don't set them as keyword arguments, like x = "num_var", or the mapping won't work properly.

```
g = sb.FacetGrid(data = df, col = 'cat_var')
g.map(plt.hist, "num_var")
```

```
group_means = df.groupby(['many_cat_var']).mean()
group_order = group_means.sort_values(['num_var'], ascending = False).index

g = sb.FacetGrid(data = df, col = 'many_cat_var', col_wrap = 5, size = 2,
                 col_order = group_order)
g.map(plt.hist, 'num_var', bins = np.arange(5, 15+1, 1))
g.set_titles('{col_name}')
```

Other operations may be performed to increase the immediate readability of the plots: setting each facet height to 2 inches ("size"), sorting the facets by group mean ("col_order"), limiting the number of bin edges, and changing the titles of each facet to just the categorical level name using the set_titles method and {col_name} template variable.

Adapted Bar Charts
Histograms and bar charts were introduced in the previous lesson as depicting the distribution of numeric and categorical variables, respectively, with the height (or length) of bars indicating the number of data points that fell within each bar's range of values. These plots can be adapted for use as bivariate plots by, instead of indicating count by height, indicating a mean or other statistic on a second variable.

For example, we could plot a numeric variable against a categorical variable by adapting a bar chart so that its bar heights indicate the mean of the numeric variable. This is the purpose of seaborn's barplot function:

```
base_color = sb.color_palette()[0]
sb.barplot(data = df, x = 'cat_var', y = 'num_var', color = base_color)
```

As an alternative, the pointplot function can be used to plot the averages as points rather than bars. This can be useful if having bars in reference to a 0 baseline aren't important or would be confusing.

```
sb.pointplot(data = df, x = 'cat_var', y = 'num_var', linestyles = "")
plt.ylabel('Avg. value of num_var')
```

The above plots can be useful alternatives to the box plot and violin plot if the data is not conducive to either of those plot types. For example, if the numeric variable is binary in nature, taking values only of 0 or 1, then a box plot or violin plot will not be informative, leaving the adapted bar chart as the best choice for displaying the data.

Alternate Variations
Instead of computing summary statistics on fixed bins, you can also make computations on a rolling window through use of pandas' rolling method. Since the rolling window will make computations on sequential rows of the dataframe, we should use sort_values to put the x-values in ascending order first.

```
# compute statistics in a rolling window
df_window = df.sort_values('num_var1').rolling(15)
x_winmean = df_window.mean()['num_var1']
y_median = df_window.median()['num_var2']
y_q1 = df_window.quantile(.25)['num_var2']
y_q3 = df_window.quantile(.75)['num_var2']

# plot the summarized data
base_color = sb.color_palette()[0]
line_color = sb.color_palette('dark')[0]
plt.scatter(data = df, x = 'num_var1', y = 'num_var2')
plt.errorbar(x = x_winmean, y = y_median, c = line_color)
plt.errorbar(x = x_winmean, y = y_q1, c = line_color, linestyle = '--')
plt.errorbar(x = x_winmean, y = y_q3, c = line_color, linestyle = '--')

plt.xlabel('num_var1')
plt.ylabel('num_var2')
```
Note that we're also not limited to just one line when plotting. When multiple Matplotlib functions are called one after the other, all of them will be plotted on the same axes. Instead of plotting the mean and error bars, we will plot the three central quartiles, laid on top of the scatterplot

## Intro to Neural Networks

The design of the Artificial Neural Network was inspired by the biological one. The neurons used in the artificial network below are essentially mathematical functions.

Each network has:

Input neurons - which we refer to as the input layer of neurons.
Output neurons - which we refer to as the output layer of neurons.
Internal neurons - which we refer to as the hidden layer of neurons. Each neural network can have many hidden layers.

Notice that there is no connection between the number of inputs, number of hidden neurons in the hidden layer or number of outputs.

Notice the "lines" connecting the different neurons?

In practice, these lines symbolize a coefficient (a scalar) that is mathematically connecting one neuron to the next. These coefficients are called weights.

The "lines" connect each neuron in a specific layer to all of the neurons on the following. For example, in our example, you can see how each neuron in the hidden layer is connected to a neuron in the output one.

Since there are so many weights connecting one layer to the next, we mathematically organize those coefficients in a matrix, denoted as the weight matrix.

Later you will learn that when we train an artificial neural network, we are actually looking for the best set of weights that will give us a desired outcome.

![Neural network](./neural_network.png)

When working with neural networks we have 2 primary phases: Training and Evaluation.
During the training phase, we take the data set (also called the training set), which includes many pairs of inputs and their corresponding targets (outputs). Our goal is to find a set of weights that would best map the inputs to the desired outputs.

In the evaluation phase, we use the network that was created in the training phase, apply our new inputs and expect to obtain the desired outputs.

The training phase will include two steps: Feedforward and Backpropagation.

We will repeat these steps as many times as we need until we decide that our system has reached the best set of weights, giving us the best possible outputs.

As you saw in the video above, vector $\vec{h'}$ of the hidden layer will be calculated by multiplying the input vector with the weight matrix $W^{1}$ the following way:

${\vec{h'}=(\vec{x}W^1)}$

After finding $\vec{h'}$ we need an **activation function** $\Phi$.

This activation function finalizes the computation of the hidden layer's values.

We can use the following two equations to express the final hidden vector $\vec{h'}$:

$\vec{h} = \Phi(\vec{x} W^1 )$

Since $W_{ij}$ represents the weight component in the weight matrix, connecting neuron i from the input to neuron j in the hidden layer, we can also write these calculations using a *linear combination*: (notice that in this example we have n inputs and only 3 hidden neurons):

$h_1=\Phi(x_1W_{11}+x_2W_{21}+...+x_3W_{n1})$
$h_2=\Phi(x_1W_{12}+x_2W_{22}+...+x_3W_{n2})$
$h_3=\Phi(x_1W_{13}+x_2W_{23}+...+x_3W_{n3})$

We finished our first step, finding $\vec{h}$ and now need to find the output $\vec{y}$. The process of calculating the output vector is mathematically similar to that of calculating the vector of the hidden layer. We use, again, a vector by matrix multiplication. The vector is the newly calculated hidden layer and the matrix is the one connecting the hidden layer to the output.
​
Essentially, each new layer in an neural network is calculated by a vector by matrix multiplication, where the vector represents the inputs to the new layer and the matrix is the one connecting these new inputs to the next layer.

**Classification problems**

- Example: student acceptance at a university by analysis of two variables: test score and grades. We draw a model that is a linear boundary given by the equation $2*x_1+x_2-18=0$ to predict the overall score. If score > 0, we accept the student, otherwise reject. That's our model's **prediction.**
- More generally, our model is a equation of the form: $w_1x_1+w_2x_2+b=0$ or $Wx+b=0$. We try to predict the label y, which in this case is either 0 or 1. Our prediction is the output of our model given by $\hat y$.
- $\hat  y = 1 \ if \ Wx+b \ge 0$ and $\hat y = 0 \ if \ Wx+b<0$.
- Our algorithm seeks to find a model where most of our predictions match the labels.
- If we have more dimensions, our boundary will be a hyperplane that is n-1 dimensional.


**Perceptrons**
Functions that combine inputs in some fashion and output a number from 0 to 1. A step function is applied to the prediction of the model.

- Perceptron trick: we draw a random line, find how badly it is doing by "asking the points". For the misclassified points, they want the boundary to move closer to them, in order to eventually classify them correctly. The trick to do so is to add or subtract to the line coordinates some percentage of the points' coordinates. The multiplicative constant is the **learning rate**. The bias is added/subtracted the learning rate. The new line will be closer to the misclassified point.

The logical operators AND, OR, NOT and XOR can be modeled with perceptrons. Check out `perceptrons.py` to see code.

In real life, though, we can't be building these perceptrons ourselves. The idea is that we give them the result, and they build themselves.

Our error function will tell us the direction to go to. We take at each step the direction that will reduce the error the most. Our error function needs to be continuous as opposed to discrete and also differentiable. That is because we are using derivatives to compute steps in our parameters! Then we need to switch to continuous predictions too. We apply the sigmoid function instead of the step function.

So our predictions now need to be continuous as opposed to 0 or 1 only. Now we our model will need to output the likelihood that a point is classified in as a 1. No more "yes" or "no" but "X% likely". The closer to the line the point is, the greater the likelihood. We go from the step function to the **sigmoid** function that ranges from 0 to 1 continuously. The shape of the **activation function** is now given by: $\sigma(x)=1/(1+e^{-x})$ as opposed to a step function as before.

![Continuous prediction with sigmoid function](./neural_networks/images/sigmoid.png)

The **softmax function** is used as an activation when there is more than two classes in the classification problem. It has to calculate probabilities that sum to 1. It also has to work with scores that are negative. We use exponential to convert every possible number into a positive number. The softmax formula is: $P(class \ i) = e^{z_i}/(e^{z_1}+e^{z_2}+...+e^{z_n})$. The value of the softmax function for a given input is exactly the same as the sigmoid for that input.

**One-hot encoding** concept: to turn multiple categorical variables into a form that we can feed to ML algos.

**Maximum Likelihood and Cross-Entropy**
The maximum likelihood is the idea of how accurate a model is at classifying most points correctly based on their expected labels. It is related to the product of the probabilities of all points being classified as their respective labels. The result will be higher for a more accurate model.

To make the numbers more "good looking", we transform them by taking the logarithm. The product then turns into sums. So we take the -ln of the product to turn that into a sum of lns. The result of the sum is called **cross-entropy**. A good model will have a low cross entropy ans so now we want to minimize the cross entropy! The cross entropy definitely connects probabilities and error functions. The cross-entropy is inversely proportional to the total probability of an outcome.

The formula is:

$$CE=-\sum_{i=1}^{m}y_iln(p_i)+(1-y_i)ln(1-p_i)$$

Note: We are only summing the probabilities of the events that actually occurred since y is 0 or 1.

For more than two classes we write more generically:

$$CE=-\sum_{i=1}^{n}\sum_{j=1}^{m}y_{ij}ln(p_{ij})$$

Then our error function is:

$$E(W,b)=-\frac{1}{m}\sum_{i=1}^{m}(1-y_i)ln(1-\sigma(Wx^{(i)}+b)+y_iln(\sigma(Wx^{(i)}+b)))$$

Or, the error function for a multivariate classification problem is given by:

$$E(W,b)=-\frac{1}{m}\sum_{i=1}^{m} \sum_{j=1}^{n} y_{ij}ln(\hat y_{ij})=-\frac{1}{m}\sum_{i=1}^{m} \sum_{j=1}^{n} y_{ij}ln[\sigma(Wx^{(i)}+b)]$$

**Gradient Descent**
We use the negative of the gradient of the error function as the direction to go that leads to the greatest decrease in the error function, $-\nabla E$. We use a learning rate $\alpha$ to take not too big of a step. We compute the new weights and bias as:

$$w'_i=w_i-\alpha \frac{\partial E}{\partial w_i}$$

$$b'=b-\alpha\frac{\partial E}{\partial b}$$

which assures that the prediction given by $\hat y=\sigma(W'x+b')$ is better than the former one.

After derivation, we find that the gradient vector is simply given by:
$$\nabla E=-(y-\hat y)(x_1,x_2,...x_n,1)$$
The gradient is actually a scalar times the coordinates of the point! And what is the scalar? Nothing less than a multiple of the difference between the label and the prediction. So, if a point is well classified, we will get a small gradient. And if it's poorly classified, the gradient will be quite large.

The gradient descent step will then be:

$$w_i'=w_i-\alpha [-(y-\hat y)x_i] \ or \ w_i+\alpha (y-\hat y)x_i$$

$$b'=b+\alpha (y-\hat y)$$

Note: Since we've taken the average of the errors, the term we are adding should be $\frac{1}{m} \cdot \alpha$ instead of $\alpha$ but as $\alpha$ is a constant, then in order to simplify calculations, we'll just take $\frac{1}{m} \cdot \alpha$ to be our learning rate, and abuse the notation by just calling it $\alpha$.

The gradient descent algorithm is then as follows:
- Start with random weights
- For every point $(x_1,...,x_n)$ we update the weights and bias: $w'_i=w_i-\alpha (\hat y - y)x_i$ and $b'=b-\alpha(\hat y-y)$.
- We repeat until error is small or for a number of times called epochs.

In the gradient descent even correctly classified points have the line move with respect to them, which is not the case for the perceptron algorithm (correctly classified points don't have their weights changed). These points actually have the line move farther away from them, so as to increase their probability of being correctly classified :)

**Non-linear models**

Let's say we want to combine two linear models to produce a non-linear one. For instance we can sum the outputs of two linear models and apply the sigmoid function to produce a number between 0 and 1. We can use weights to use different proportions of the two models too and have a bias as well.

![Neural net arrangement](./neural_networks/images/neural_network_arrangement.png)

In deep neural networks we have more hidden layers where intermediate non-linear models combine to generate other non-linear models.

![Deep neural net arrangement](./neural_networks/images/deep_neural_network_arrangement.png)

And for multi-class classification, as opposed to binary, we can have multiple neural networks, each one to predict the probability of one of the classes and then apply the softmax function. This seems like overkill though. What we actually do is add more nodes to the output layer and each will be the probability of the input belonging to one of the possible classes. We apply the softmax to the different scores.

**Feedforward**

Process of a neural network to obtain the prediction from the input vector given its architecture.

![Feedforward process](./neural_networks/images/feedforward.png)

Now, we're ready to get our hands into training a neural network. For this, we'll use the method known as backpropagation. In a nutshell, **backpropagation** will consist of:

- Doing a feedforward operation.
- Comparing the output of the model with the desired output.
- Calculating the error.
- Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
- Use this to update the weights, and get a better model.
- Continue this until we have a model that is good.

With backpropagation, we "listen" to the point that was misclassified and weigh in the models differently based on the classification they output with respect to that point.

![Backpropagation](./neural_networks/images/backpropagation.png)

![Backpropagation](./neural_networks/images/backpropagation_math.png)

**Gradient Descent with Squared Errors**

Other than the log-loss function, there are many other error functions used for neural networks. Another one is called the mean squared error. As the name says, this one is the mean of the squares of the differences between the predictions and the labels.

We want to find the weights for our neural networks. Let's start by thinking about the goal. The network needs to make predictions as close as possible to the real values. To measure this, we use a metric of how wrong the predictions are, the error. A common metric is the sum of the squared errors (SSE):

$$E=\frac{1}{2}\sum_{\mu}\sum_{j}[y_j^{\mu}-\hat y_j^{\mu}]^2$$

where $\hat y$ is the prediction and y is the true value, and you take the sum over all output units j and another sum over all data points $\mu$. The square ensures the error is always positive and larger errors are penalized more than smaller errors. Also, it makes the math nice, always a plus.

We want the network's prediction error to be as small as possible and the weights are the knobs we can use to make that happen. Our goal is to find weights $w_{ij}$ that minimize the squared error E. To do this with a neural network, typically you'd use gradient descent.

Since the weights will just go wherever the gradient takes them, they can end up where the error is low, but not the lowest. These spots are called local minima. If the weights are initialized with the wrong values, gradient descent could lead the weights into a local minimum.

To calculate the weight step:

$$\frac{\partial E}{\partial w_i}=\frac{\partial}{\partial w_i} \frac{1}{2}(y- \hat y(w_i))^2$$

Applying the chain rule:

$$\frac{\partial E}{\partial w_i}=-(y - \hat y)\frac{\partial \hat y}{\partial w_i}$$

$$\frac{\partial E}{\partial w_i}=-(y - \hat y)f' (h)\frac{\partial}{\partial w_i}\sum w_ix_i \ where \ h=\sum w_ix_i$$

$$\frac{\partial E}{\partial w_i}=-(y - \hat y)f' (h)x_i$$

So we can say that:

$$\Delta w_i = \eta \, \delta x_i$$

with the error term δ as

$$\delta = (y - \hat y) f'(h) = (y - \hat y) f'(\sum w_i x_i)$$

Remember, in the above equation $(y - \hat y)$ is the output error, and $f'(h)$ refers to the derivative of the activation function, f(h). We'll call that derivative the output gradient.

Now I'll write this out in code for the case of only one output unit. We'll also be using the sigmoid as the activation function f(h).

```
# Defining the sigmoid function for activations
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Input data
x = np.array([0.1, 0.3])
# Target
y = 0.2
# Input to output weights
weights = np.array([-0.8, 0.5])

# The learning rate, eta in the weight step equation
learnrate = 0.5

# the linear combination performed by the node (h in f(h) and f'(h))
h = x[0]*weights[0] + x[1]*weights[1]
# or h = np.dot(x, weights)

# The neural network output (y-hat)
nn_output = sigmoid(h)

# output error (y - y-hat)
error = y - nn_output

# output gradient (f'(h))
output_grad = sigmoid_prime(h)

# error term (lowercase delta)
error_term = error * output_grad

# Gradient descent step
del_w = [ learnrate * error_term * x[0],
          learnrate * error_term * x[1]]
# or del_w = learnrate * error_term * x
```

**Implementing the hidden layer**

Now, the weights need to be stored in a matrix. Each row in the matrix will correspond to the weights leading out of a single input unit, and each column will correspond to the weights leading in to a single hidden unit. For our three input units and two hidden units, the weights matrix looks like this.

![hidden layer](./neural_networks/images/hidden_layer.png)

To initialize these weights in NumPy, we have to provide the shape of the matrix. If features is a 2D array containing the input data:

```
# Number of records and input units
n_records, n_inputs = features.shape
# Number of hidden units
n_hidden = 2
weights_input_to_hidden = np.random.normal(0, n_inputs**-0.5, size=(n_inputs, n_hidden))
```

This creates a 2D array (i.e. a matrix) named `weights_input_to_hidden` with dimensions `n_inputs` by `n_hidden`. Remember how the input to a hidden unit is the sum of all the inputs multiplied by the hidden unit's weights. So for each hidden layer unit, $h_j$, we need to calculate the following:
$$h_j=\sum_i w_{ij}xi$$

In NumPy, you can do this for all the inputs and all the outputs at once using np.dot:
`hidden_inputs = np.dot(inputs, weights_input_to_hidden)`

![hidden layer](./neural_networks/images/hidden_layer_2.png)

*Making a column vector*

You see above that sometimes you'll want a column vector, even though by default NumPy arrays work like row vectors. It's possible to get the transpose of an array like so arr.T, but for a 1D array, the transpose will return a row vector. Instead, use `arr[:,None]` to create a column vector:

```
print(features)
> array([ 0.49671415, -0.1382643 ,  0.64768854])

print(features.T)
> array([ 0.49671415, -0.1382643 ,  0.64768854])

print(features[:, None])
> array([[ 0.49671415],
       [-0.1382643 ],
       [ 0.64768854]])
```

Alternatively, you can create arrays with two dimensions. Then, you can use `arr.T` to get the column vector.

```
np.array(features, ndmin=2)
> array([[ 0.49671415, -0.1382643 ,  0.64768854]])

np.array(features, ndmin=2).T
> array([[ 0.49671415],
       [-0.1382643 ],
       [ 0.64768854]])
```

**Backpropagation**

Now we've come to the problem of how to make a multilayer neural network **learn**. Before, we saw how to update weights with gradient descent. The backpropagation algorithm is just an extension of that, using the chain rule to find the error with the respect to the weights connecting the input layer to the hidden layer (for a two layer network).

To update the weights to hidden layers using gradient descent, you need to know how much error each of the hidden units contributed to the final output. Since the output of a layer is determined by the weights between layers, the error resulting from units is scaled by the weights going forward through the network. Since we know the error at the output, we can use the weights to work backwards to hidden layers.

For example, in the output layer, you have errors $\delta^o_k$  attributed to each output unit k. Then, the error attributed to hidden unit j is the output errors, scaled by the weights between the output and hidden layers (and the gradient):

$$\delta_j^h=\sum W_{ij}\delta_o^kf'(h_j)$$

Then, the gradient descent step is the same as before, just with the new errors:
$$\Delta w_{ij}=\eta \delta_j^hx_i$$

where $w_{ij}$ are the weights between the inputs and hidden layer and $x_i$ are input unit values. This form holds for however many layers there are. The weight steps are equal to the step size times the output error of the layer times the values of the inputs to that layer

$$\Delta w_{pq}=\eta \delta_{output}V_{in}$$

![](./neural_networks/images/backpropagation_hidden_layers.png)

Here, you get the output error, $\delta_{output}$, by propagating the errors backwards from higher layers. And the input values, $V_{in}$ are the inputs to the layer, the hidden layer activations to the output unit for example.

*Working through an example*
Let's walk through the steps of calculating the weight updates for a simple two layer network. Suppose there are two input values, one hidden unit, and one output unit, with sigmoid activations on the hidden and output units. The following image depicts this network. (Note: the input values are shown as nodes at the bottom of the image, while the network's output value is shown as $\hat y$ at the top. The inputs themselves do not count as a layer, which is why this is considered a two layer network.)

![](./neural_networks/images/backpropagation_example.png)

Assume we're trying to fit some binary data and the target is y = 1. We'll start with the forward pass, first calculating the input to the hidden unit

$$h = \sum_i w_i x_i = 0.1 \times 0.4 - 0.2 \times 0.3 = -0.02$$

and the output of the hidden unit

$$a = f(h) = \mathrm{sigmoid}(-0.02) = 0.495$$

Using this as the input to the output unit, the output of the network is

$$\hat y = f(W \cdot a) = \mathrm{sigmoid}(0.1 \times 0.495) = 0.512$$

With the network output, we can start the backwards pass to calculate the weight updates for both layers. Using the fact that for the sigmoid function $f'(W \cdot a) ==f(W⋅a)(1−f(W⋅a))$, the error term for the output unit is

$$\delta^o = (y - \hat y) f'(W \cdot a) = (1 - 0.512) \times 0.512 \times(1 - 0.512) = 0.122$$

Now we need to calculate the error term for the hidden unit with backpropagation. Here we'll scale the error term from the output unit by the weight W connecting it to the hidden unit. For the hidden unit error term, $\delta^h_j = \sum_k W_{jk} \delta^o_k f'(h_j)$, but since we have one hidden unit and one output unit, this is much simpler.

$$\delta^h = W \delta^o f'(h) = 0.1 \times 0.122 \times 0.495 \times (1 - 0.495) = 0.003$$

Now that we have the errors, we can calculate the gradient descent steps. The hidden to output weight step is the learning rate, times the output unit error, times the hidden unit activation value.

$$\Delta W = \eta \delta^o a = 0.5 \times 0.122 \times 0.495 = 0.0302$$

Then, for the input to hidden weights $w_i$, it's the learning rate times the hidden unit error, times the input values.

$$\Delta w_i = \eta \delta^h x_i = (0.5 \times 0.003 \times 0.1, 0.5 \times 0.003 \times 0.3) = (0.00015, 0.00045)$$

From this example, you can see one of the effects of using the sigmoid function for the activations. The maximum derivative of the sigmoid function is 0.25, so the errors in the output layer get reduced by at least 75%, and errors in the hidden layer are scaled down by at least 93.75%! You can see that if you have a lot of layers, using a sigmoid activation function will quickly reduce the weight steps to tiny values in layers near the input. This is known as the vanishing gradient problem. Later in the course you'll learn about other activation functions that perform better in this regard and are more commonly used in modern network architectures.

*Implementing in NumPy*

For the most part you have everything you need to implement backpropagation with NumPy.

However, previously we were only dealing with error terms from one unit. Now, in the weight update, we have to consider the error for each unit in the hidden layer, $\delta_j$:

$$\Delta w_{ij} = \eta \delta_j x_i$$

Firstly, there will likely be a different number of input and hidden units, so trying to multiply the errors and the inputs as row vectors will throw an error:

```
hidden_error*inputs
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-22-3b59121cb809> in <module>()
----> 1 hidden_error*x

ValueError: operands could not be broadcast together with shapes (3,) (6,)
```

Also, $w_{ij}$ is a matrix now, so the right side of the assignment must have the same shape as the left side. Luckily, NumPy takes care of this for us. If you multiply a row vector array with a column vector array, it will multiply the first element in the column by each element in the row vector and set that as the first row in a new 2D array. This continues for each element in the column vector, so you get a 2D array that has shape `(len(column_vector), len(row_vector))`.

```
hidden_error*inputs[:,None]
array([[ -8.24195994e-04,  -2.71771975e-04,   1.29713395e-03],
       [ -2.87777394e-04,  -9.48922722e-05,   4.52909055e-04],
       [  6.44605731e-04,   2.12553536e-04,  -1.01449168e-03],
       [  0.00000000e+00,   0.00000000e+00,  -0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,  -0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,  -0.00000000e+00]])
```

It turns out this is exactly how we want to calculate the weight update step. As before, if you have your inputs as a 2D array with one row, you can also do `hidden_error*inputs.T`, but that won't work if inputs is a 1D array.

Backpropagation Exercise:
```
import numpy as np
x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

Backwards pass
TODO: Calculate output error
error = target-output

TODO: Calculate error term for output layer
output_error_term = error*output*(1-output)

TODO: Calculate error term for hidden layer
hidden_error_term = np.dot(output_error_term,weights_hidden_output)*hidden_layer_output*(1-hidden_layer_output)

TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate*output_error_term*hidden_layer_output

TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate*hidden_error_term*x[:,None]

Output
Change in weights for hidden layer to output layer:
[0.00804047 0.00555918]
Change in weights for input layer to hidden layer:
[[ 1.77005547e-04 -5.11178506e-04]
 [ 3.54011093e-05 -1.02235701e-04]
 [-7.08022187e-05  2.04471402e-04]]
```

Now we've seen that the error term for the output layer is

$$\delta_k = (y_k - \hat y_k) f'(a_k)$$

and the error term for the hidden layer is

$$\delta_j=\sum[w_{jk} \delta_k]f'(h_j)$$

For now we'll only consider a simple network with one hidden layer and one output unit. Here's the general algorithm for updating the weights with backpropagation:

- Set the weight steps for each layer to zero
    - The input to hidden weights $\Delta w_{ij} = 0$
    - The hidden to output weights $\Delta W_j = 0$
- For each record in the training data:
    - Make a forward pass through the network, calculating the output $\hat y$​
    - Calculate the error gradient in the output unit, $\delta^o = (y - \hat y) f'(z)$ where $z = \sum_j W_j a_j$, the input to the output unit.
    - Propagate the errors to the hidden layer $\delta^h_j = \delta^o W_j f'(h_j)$
    - Update the weight steps:
        - $\Delta W_j = \Delta W_j + \delta^o a_j$
        - $\Delta w_{ij} = \Delta w_{ij} + \delta^h_j a_i$
- Update the weights, where $\eta$ is the learning rate and m is the number of records:
    - $W_j = W_j + \eta \Delta W_j / m$
    - $w_{ij} = w_{ij} + \eta \Delta w_{ij} / m$
- Repeat for e epochs.

```
import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

n_hidden = 2
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)
for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        Forward pass
        TODO: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        output = sigmoid(np.dot(hidden_output,
                                weights_hidden_output))

        Backward pass
        TODO: Calculate the network's prediction error
        error = y - output

        TODO: Calculate error term for the output unit
        output_error_term = error * output * (1 - output)

        propagate errors to hidden layer

        TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)

        TODO: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        TODO: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:, None]

    TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
```
**Training Neural Networks**

*Underfitting and Overfitting*
Killing godzilla with a flyswatter. Or killing a fly with a bazzoka. Oversimplifying and overcomplicating!

![](./neural_networks/images/underfitting.png)
![](./neural_networks/images/overfitting.png)
![](./neural_networks/images/architectures.png)

We will tend to overfit as opposed to underfit and apply techniques to fix the overfitting errors. So we prefer to choose an overly complicated architecture and fix it.

![](./neural_networks/images/epochs_overfitting.png)
![](./neural_networks/images/early_stopping.png)
![](./neural_networks/images/example_activation_overfitting.png)
![](./neural_networks/images/activation_function_overfitting.png)
`The whole problem with Artificial Intelligence if that bad models are so certain of themselves, and good models are so full of doubts. [Bertraind Russell]`
Large cofficients --> Overfitting
How to prevent this from happening? We have to tweak the error function to punish large cofficients. This is called **regularization**. Two ways of regularization:


$$-\frac{1}{m}\sum_{i=1}^{m}(1-y_i)ln(\hat y_i)+y_iln(\hat y_i)+\lambda(|w_1|+...+|w_n|)$$


$$-\frac{1}{m}\sum_{i=1}^{m}(1-y_i)ln(\hat y_i)+y_iln(\hat y_i)+\lambda(w_1^2+...+w_n^2)$$

We choose $\lambda$ to penalize more or less.

The regularization L1 leads to higher sparsity in weights vectors and it's good for feature selection.
The regularization L2 leads to smaller sparsity and it's normally better for training models.

**Dropout** is the idea of for some epochs turn off some part of the network that is dominating the training process (larger weights) so that the other parts can pick up the slack. We actually randomly turn off some of the nodes. We give the algorithm a parameter for each node to be turned off. On average each node will have the same importance at the end!

**Random restart** to solve local minima. We start from a few random places and do gradient descent from all of them, increasing the chances of at the end arriving at the global minima or at least a pretty good local minima.

**Vanishing gradient**: with sigmoid activation, which is pretty flat on the sides, the gradient can result in tiny values for the change in the weights. So we change the activation function to, for instance, the hyperbolic tangent function or the rectified linear unit (ReLU):

$$tanhh(x)=\frac{e^x-e{-x}}{e^x+e^{-x}}$$

$$relu(x)=x \ if \ x \ge 0 \ and \ 0 \ if \ x<0$$

Here is the final neural network. Note that the final activation function is still a sigmoid returning an output between 0 and 1.

![](./neural_networks/images/relu_activation.png)

**Stochastic gradient descent** is the idea of not using all data points to compute one step (epoch) but instead choose a small random subset of the data to compute the error gradient and hence update the weights. This will make the algo faster. But we still want to use all the data, and then we split the data into a number of *batches* to make "several bad steps" as opposed to "one good step".

**Learning rate**: if too big, huge steps, may miss the minimum and make the model chaotic. If small, higher chances of arriving at the minima but the algorithm can be quite slow. A good one is one that decreases as the model gets closer to a solution.

![](neural_networks/images/learning_rate_tunning.png)

To get over the "hump" (local minima) where the gradient is zero, we can take some weighted average of a few last few steps. We use the **momentum** constant $\beta$ to weigh in each step, to assure that more recent steps matter more. Once we get to the global minima, it pushes us away but not enough to go away from that point. Algos that use momentum tend to perform really well.

![](neural_networks/images/momentum.png)

There are other error functions around the world!

##Deep Learning with PyTorch

PyTorch was released in early 2017 and has been making a pretty big impact in the deep learning community. It's developed as an open source project by the Facebook AI Research team, but is being adopted by teams everywhere in industry and academia. In my experience, it's the best framework for learning deep learning and just a delight to work with in general. By the end of this lesson, you'll have trained your own deep learning model that can classify images of cats and dogs.

I'll first give you a basic introduction to PyTorch, where we'll cover tensors - the main data structure of PyTorch. I'll show you how to create tensors, how to do simple operations, and how tensors interact with NumPy.

Then you'll learn about a module called autograd that PyTorch uses to calculate gradients for training neural networks. Autograd, in my opinion, is amazing. It does all the work of backpropagation for you by calculating the gradients at each operation in the network which you can then use to update the network weights.

Next you'll use PyTorch to build a network and run data forward through it. After that, you'll define a loss and an optimization method to train the neural network on a dataset of handwritten digits. You'll also learn how to test that your network is able to generalize through validation.

However, you'll find that your network doesn't work too well with more complex images. You'll learn how to use pre-trained networks to improve the performance of your classifier, a technique known as transfer learning.


###Git and Github

Remember that the main point of a version control system is to help you maintain a detailed history of the project as well as the ability to work on different versions of it. Having a detailed history of a project is important because it lets you see the progress of the project over time. If needed, you can also jump back to any point in the project to recover data or files.

Version Control System - Git, Subversion and Mercurial
- the centralized model - all users connect to a central, master repository
- the distributed model - each user has the entire repository on their computer
Git is of distributed type -> version control tool or source code manager!
Github -> service that hosts Git projects

Some Terminology:
- Commit
Git thinks of its data like a set of snapshots of a mini filesystem. Every time you commit (save the state of your project in Git), it basically takes a picture of what all your files look like at that moment and stores a reference to that snapshot. You can think of it as a save point in a game - it saves your project's files and any information about them.

Everything you do in Git is to help you make commits, so a commit is the fundamental unit in Git.

- Repository / repo
A repository is a directory which contains your project work, as well as a few files (hidden by default on Mac OS X) which are used to communicate with Git. Repositories can exist either locally on your computer or as a remote copy on another computer. A repository is made up of commits.

- Working Directory
The Working Directory is the files that you see in your computer's file system. When you open your project files up on a code editor, you're working with files in the Working Directory.

This is in contrast to the files that have been saved (in commits!) in the repository.

When working with Git, the Working Directory is also different from the command line's concept of the current working directory which is the directory that your shell is "looking at" right now.

- Checkout
A checkout is when content in the repository has been copied to the Working Directory.

- Staging Area / Staging Index / Index
A file in the Git directory that stores information about what will go into your next commit. You can think of the staging area as a prep table where Git will take the next commit. Files on the Staging Index are poised to be added to the repository.

- SHA
A SHA is basically an ID number for each commit. Here's what a commit's SHA might look like: `e2adf8ae3e2e4ed40add75cc44cf9d0a869afeb6`.

It is a 40-character string composed of characters (0–9 and a–f) and calculated based on the contents of a file or directory structure in Git. "SHA" is shorthand for "Secure Hash Algorithm".

Running the `git init` command sets up all of the necessary files and directories that Git will use to keep track of everything. All of these files are stored in a directory called `.git` (notice the . at the beginning - that means it'll be a hidden directory on Mac/Linux). This .git directory is the "repo"! This is where git records all of the commits and keeps track of everything!

Here's a brief synopsis on each of the items in the `.git` directory:

config file - where all project specific configuration settings are stored.
From the Git Book:

Git looks for configuration values in the configuration file in the Git directory (.git/config) of whatever repository you’re currently using. These values are specific to that single repository.

For example, let's say you set that the global configuration for Git uses your personal email address. If you want your work email to be used for a specific project rather than your personal email, that change would be added to this file.

description file - this file is only used by the GitWeb program, so we can ignore it

hooks directory - this is where we could place client-side or server-side scripts that we can use to hook into Git's different lifecycle events

info directory - contains the global excludes file

objects directory - this directory will store all of the commits we make

refs directory - this directory holds pointers to commits (basically the "branches" and "tags")

Remember, other than the "hooks" directory, you shouldn't mess with pretty much any of the content in here. The "hooks" directory can be used to hook into different parts or events of Git's workflow, but that's a more advanced topic that we won't be getting into in this course.

The git clone command is used to create an identical copy of an existing repository.

`git clone <path-to-repository-to-clone>`

This command:

takes the path to an existing repository
by default will create a directory with the same name as the repository that's being cloned
can be given a second argument that will be used as the name of the directory
will create the new repository inside of the current working directory

The `git status` is our key to the mind of Git. It will tell us what Git is thinking and the state of our repository as Git sees it. When you're first starting out, you should be using the git status command all of the time! Seriously. You should get into the habit of running it after any other command. This will help you learn how Git works and it'll help you from making (possibly) incorrect assumptions about the state of your files/repository.

The git status command will display the current status of the repository.

This command will:

tell us about new files that have been created in the Working Directory that Git hasn't started tracking, yet
files that Git is tracking that have been modified
whole bunch of other info...

Let's do a quick recap of the git log command. The git log command is used to display all of the commits of a repository.

`git log`
By default, this command displays:

the SHA
the author
the date
and the message

What could we do here to not waste a lot of space and make the output smaller? We can use a flag.
`git log --oneline` -> 7 initial chars of the SHA and message only displayed

The git log command has a flag that can be used to display the files that have been changed in the commit, as well as the number of lines that have been added or deleted. The flag is --stat ("stat" is short for "statistics"):

`git log --stat`

This command:

displays the file(s) that have been modified
displays the number of lines that have been added/removed
displays a summary line with the total number of modified files and lines that have been added/removed

The git log command has a flag that can be used to display the actual changes made to a file. The flag is --patch which can be shortened to just -p:

`git log -p`

🔵 - the file that is being displayed
🔶 - the hash of the first version of the file and the hash of the second version of the file - not usually important, so it's safe to ignore
❤️ - the old version and current version of the file
🔍 - the lines where the file is added and how many lines there are
✏️ - the actual changes made in the commit
lines that are red and start with a minus (-) were in the original version of the file but have been removed by the commit
lines that are green and start with a plus (+) are new lines that have been added in the commit

`git log -p -w` will ignore changes in whitespaces like indentation changes.

`git log -p fdf5493`
By supplying a SHA, the git log -p command will start at that commit! No need to scroll through everything! Keep in mind that it will also show all of the commits that were made prior to the supplied SHA.

The `git show` command will show only one commit. So don't get alarmed when you can't find any other commits - it only shows one. The output of the git show command is exactly the same as the git log -p command. So by default, git show displays:

the commit
the author
the date
the commit message
the patch information

The git add command is used to move files from the Working Directory to the Staging Index.

`git add <file1> <file2> … <fileN>`
This command:

takes a space-separated list of file names
alternatively, the period . can be used in place of a list of files to tell Git to add the current directory (and all nested files)

`git config --global core.editor <your-editor's-config-went-here>` - commit to open specific code editor to include message

Terminal Hangs
If you switch back to the Terminal for a quick second, you'll see that the Terminal is chillin' out just waiting for you to finish with the code editor that popped up. You don't need to worry about this, though. Once we add the necessary content to the code editor and finally close the code editor window, the Terminal will unfreeze and return to normal.

TIP: If the commit message you're writing is short and you don't want to wait for your code editor to open up to type it out, you can pass your message directly on the command line with the -m flag:
`git commit -m "Initial commit"`

The goal is that each commit has a single focus. Each commit should record a single-unit change. Now this can be a bit subjective (which is totally fine), but each commit should make a change to just one aspect of the project.

Now this isn't limiting the number of lines of code that are added/removed or the number of files that are added/removed/modified. Let's say you want to change your sidebar to add a new image. You'll probably:

add a new image to the project files
alter the HTML
add/modify CSS to incorporate the new image
A commit that records all of these changes would be totally fine!

The best way that I've found to think about what should be in a commit is to think, "What if all changes introduced in this commit were erased?". If a commit were erased, it should only remove one thing.

The `git commit` command takes files from the Staging Index and saves them in the repository.

will open the code editor that is specified in your configuration
Inside the code editor:

a commit message must be supplied
lines that start with a # are comments and will not be recorded
save the file after adding a commit message
close the editor to make the commit

do keep the message short (less than 60-ish characters)
do explain what the commit does (not how or why!)
do not explain why the changes are made (more on this below)
do not explain how the changes are made (that's what `git log -p` is for!)
do not use the word "and"
if you have to use "and", your commit message is probably doing too many changes - break the changes into separate commits

Explain the Why
If you need to explain why a commit needs to be made, you can!

When you're writing the commit message, the first line is the message itself. After the message, leave a blank line, and then type out the body or explanation including details about why the commit is needed (e.g. URL links).
Only the message (the first line) is included in `git log --oneline`, though!

**Udacity Git Commit Message Style Guide**
```
Example Commit Message
feat: Summarize changes in around 50 characters or less

More detailed explanatory text, if necessary. Wrap it to about 72
characters or so. In some contexts, the first line is treated as the
subject of the commit and the rest of the text as the body. The
blank line separating the summary from the body is critical (unless
you omit the body entirely); various tools like `log`, `shortlog`
and `rebase` can get confused if you run the two together.

Explain the problem that this commit is solving. Focus on why you
are making this change as opposed to how (the code explains that).
Are there side effects or other unintuitive consequenses of this
change? Here's the place to explain them.

Further paragraphs come after blank lines.

 - Bullet points are okay, too

 - Typically a hyphen or asterisk is used for the bullet, preceded
   by a single space, with blank lines in between, but conventions
   vary here

If you use an issue tracker, put references to them at the bottom,
like this:

Resolves: #123
See also: #456, #789
```

To recap, the `git diff` command is used to see changes that have been made but haven't been committed, displaying:

the files that have been modified
the location of the lines that have been added/removed
the actual changes that have been made

Git Ignore
If you want to keep a file in your project's directory structure but make sure it isn't accidentally committed to the project, you can use the specially named file, .gitignore (note the dot at the front, it's important!). Add this file to your project in the same directory that the hidden .git directory is located. All you have to do is list the names of files that you want Git to ignore (not track) and it will ignore them.

Git knows to look at the contents of a file with the name .gitignore. Since it saw "project.docx" in it, it ignored that file and doesn't show it in the output of git status.

Let's say that you add 50 images to your project, but want Git to ignore all of them. Does this mean you have to list each and every filename in the .gitignore file? Oh gosh no, that would be crazy! Instead, you can use a concept called globbing.

Globbing lets you use special characters to match patterns/characters. In the .gitignore file, you can use the following:

blank lines can be used for spacing
- marks line as a comment
* - matches 0 or more characters
? - matches 1 character
[abc] - matches a, b, _or_ c
** - matches nested directories - a/**/z matches
a/z
a/b/z
a/b/c/z
So if all of the 50 images are JPEG images in the "samples" folder, we could add the following line to .gitignore to have Git ignore all 50 images.

samples/*.jpg

To recap, the .gitignore file is used to tell Git about the files that Git should not track. This file should be placed in the same directory that the .git directory is in.

The `git tag` command is used to add a marker on a specific commit. The tag does not move around as new commits are added.

`git tag -a beta`
This command will:

add a tag to the most recent commit
add a tag to a specific commit if a SHA is passed

`git tag -a v1.0 a87984` (after popping open a code editor to let you supply the tag's message) this command will tag the commit with the SHA a87084 with the tag v1.0.

The head pointer points to the branch that is active (git checkout to switch where it points to)

To switch between branches, we need to use Git's checkout command.

`git checkout new-branch`
It's important to understand how this command works. Running this command will:

remove all files and directories from the Working Directory that Git is tracking
(files that Git tracks are stored in the repository, so nothing is lost)
go into the repository and pull out all of the files and directories of the commit that the branch points to

So this will remove all of the files that are referenced by commits in the master branch. It will replace them with the files that are referenced by the commits in the sidebar branch.

To recap, the git branch command is used to manage branches in Git:

to list all branches
`git branch`

to create a new "footer-fix" branch
`git branch footer-fix SHA(optional)`

to delete the "footer-fix" branch
`git branch -d footer-fix`

Deleting something can be quite nerve-wracking. Don't worry, though. Git won't let you delete a branch if it has commits on it that aren't on any other branch (meaning the commits are unique to the branch that's about to be deleted). If you created the sidebar branch, added commits to it, and then tried to delete it with the git branch -d sidebar, Git wouldn't let you delete the branch because you can't delete a branch that you're currently on. If you switched to the master branch and tried to delete the sidebar branch, Git also wouldn't let you do that because those new commits on the sidebar branch would be lost! To force deletion, you need to use a capital D flag - `git branch -D sidebar`.

`git log --oneline --graph --all` to show all branchs at once.
`git diff` to see changes that are not committed yet.

As of right now, we do not know how to undo changes. If you make a merge on the wrong branch, use this command to undo the merge:

`git reset --hard HEAD^`

(Make sure to include the ^ character! It's a known as a "Relative Commit Reference" and indicates "the parent commit"

The git merge command is used to combine Git branches:

`git merge <name-of-branch-to-merge-in>`
When a merge happens, Git will:

look at the branches that it's going to merge
look back along the branch's history to find a single commit that both branches have in their commit history
combine the lines of code that were changed on the separate branches together
makes a commit to record the merge

Fast-forward Merge
In our project, I've checked out the master branch and I want _it_ to have the changes that are on the footer branch. If I wanted to verbalize this, I could say this is - "I want to merge in the footer branch". That "merge in" is important; when a merge is performed, the other branch's changes are brought into the branch that's currently checked out.

There are two types of merges:

Fast-forward merge – the branch being merged in must be ahead of the checked out branch. The checked out branch's pointer will just be moved forward to point to the same commit as the other branch.
the regular type of merge
two divergent branches are combined
a merge commit is created

Merge Conflict Recap
A merge conflict happens when the same line or lines have been changed on different branches that are being merged. Git will pause mid-merge telling you that there is a conflict and will tell you in what file or files the conflict occurred. To resolve the conflict in a file:

locate and remove all lines with merge conflict indicators
determine what to keep
save the file(s)
stage the file(s)
make a commit

Changing The Last Commit
With the `--amend` flag, you can alter the most-recent commit.

`git commit --amend`
If your Working Directory is clean (meaning there aren't any uncommitted changes in the repository), then running git commit --amend will let you provide a new commit message. Your code editor will open up and display the original commit message. Just fix a misspelling or completely reword it! Then save it and close the editor to lock in the new commit message.

Alternatively, git commit --amend will let you include files (or changes to files) you might've forgotten to include. Let's say you've updated the color of all navigation links across your entire website. You committed that change and thought you were done. But then you discovered that a special nav link buried deep on a page doesn't have the new color. You could just make a new commit that updates the color for that one link, but that would have two back-to-back commits that do practically the exact same thing (change link colors).

The `git revert` command is used to reverse a previously made commit:

`git revert <SHA-of-commit-to-revert>`
This command:

will undo the changes that were made by the provided commit
creates a new commit to record the change

Reset vs Revert
At first glance, resetting might seem coincidentally close to reverting, but they are actually quite different. Reverting creates a new commit that reverts or undos a previous commit. Resetting, on the other hand, erases commits!

⚠️ Resetting Is Dangerous ⚠️
You've got to be careful with Git's resetting capabilities. This is one of the few commands that lets you erase commits from the repository. If a commit is no longer in the repository, then its content is gone.

To alleviate the stress a bit, Git does keep track of everything for about 30 days before it completely erases anything. To access this content, you'll need to use the git reflog command.

Relative Commit References
You already know that you can reference commits by their SHA, by tags, branches, and the special HEAD pointer. Sometimes that's not enough, though. There will be times when you'll want to reference a commit relative to another commit. For example, there will be times where you'll want to tell Git about the commit that's one before the current commit...or two before the current commit. There are special characters called "Ancestry References" that we can use to tell Git about these relative references. Those characters are:

^ – indicates the parent commit
~ – indicates the first parent commit
Here's how we can refer to previous commits:

the parent commit – the following indicate the parent commit of the current commit
HEAD^
HEAD~
HEAD~1
the grandparent commit – the following indicate the grandparent commit of the current commit
HEAD^^
HEAD~2
the great-grandparent commit – the following indicate the great-grandparent commit of the current commit
HEAD^^^
HEAD~3
The main difference between the ^ and the ~ is when a commit is created from a merge. A merge commit has two parents. With a merge commit, the ^ reference is used to indicate the first parent of the commit while ^2 indicates the second parent. The first parent is the branch you were on when you ran git merge while the second parent is the branch that was merged in.


The git reset command is used to reset (erase) commits:

`git reset <reference-to-commit>`
It can be used to:

move the HEAD and current branch pointer to the referenced commit
erase commits
move committed changes to the staging index
unstage committed changes

Git Reset's Flags
The way that Git determines if it erases, stages previously committed changes, or unstages previously committed changes is by the flag that's used. The flags are:
example: `git reset HEAD~1`
--mixed (default) - will move the commit to working directory - the new SHA will be different (timestamp changed)
--soft - will move the commit to staging area - the new SHA will be different (timestamp changed)
--hard - will erase the commit

💡 Backup Branch 💡
Remember that using the git reset command will erase commits from the current branch. So if you want to follow along with all the resetting stuff that's coming up, you'll need to create a branch on the current commit that you can use as a backup.

Before I do any resetting, I usually create a backup branch on the most-recent commit so that I can get back to the commits if I make a mistake.

Reset's --mixed Flag
Let's look at each one of these flags.

```
* 9ec05ca (HEAD -> master) Revert "Set page heading to "Quests & Crusades""
* db7e87a Set page heading to "Quests & Crusades"
* 796ddb0 Merge branch 'heading-update'
```

Using the sample repo above with HEAD pointing to master on commit 9ec05ca, running git reset --mixed HEAD^ will take the changes made in commit 9ec05ca and move them to the working directory.

💡 Back To Normal 💡
If you created the backup branch prior to resetting anything, then you can easily get back to having the master branch point to the same commit as the backup branch. You'll just need to:

remove the uncommitted changes from the working directory
merge backup into master (which will cause a Fast-forward merge and move master up to the same point as backup)

Reset's --hard Flag
Last but not least, let's look at the --hard flag:

```
* 9ec05ca (HEAD -> master) Revert "Set page heading to "Quests & Crusades""
* db7e87a Set page heading to "Quests & Crusades"
* 796ddb0 Merge branch 'heading-update'
```

Running `git reset --hard HEAD^` will take the changes made in commit 9ec05ca and erases them.

To recap, the git reset command is used erase commits:

`git reset <reference-to-commit>`
It can be used to:

move the HEAD and current branch pointer to the referenced commit
erase commits with the `--hard` flag
moves committed changes to the staging index with the `--soft` flag
unstages committed changes `--mixed` flag


A remote repository is a repository that's just like the one you're using but it's just stored at a different location. To manage a remote repository, use the git remote command:

`git remote`
It's possible to have links to multiple different remote repositories.
A shortname is the name that's used to refer to a remote repository's location. Typically the location is a URL, but it could be a file path on the same computer.
`git remote add` is used to add a connection to a new remote repository.
`git remote -v` is used to see the details about a connection to a remote.

The git push command is used to send commits from a local repository to a remote repository.

`git push origin master`
The git push command takes:

the shortname of the remote repository you want to send commits to
the name of the branch that has the commits you want to send

I said it before but I'll say it again, the branch that appears in the local repository is actually tracking a branch in the remote repository (e.g. origin/master in the local repository is called a tracking branch because it's tracking the progress of the master branch on the remote repository that has the shortname "origin").

If you don't want to automatically merge the local branch with the tracking branch then you wouldn't use git pull you would use a different command called git fetch. You might want to do this if there are commits on the repository that you don't have but there are also commits on the local repository that the remote one doesn't have either.

When git pull is run, the following things happen:

the commit(s) on the remote branch are copied to the local repository
the local tracking branch (origin/master) is moved to point to the most recent commit
the local tracking branch (origin/master) is merged into the local branch (master)

Git fetch is used to retrieve commits from a remote repository's branch but it does not automatically merge the local branch with the remote tracking branch after those commits have been received.

When git fetch is run, the following things happen:

the commit(s) on the remote branch are copied to the local repository
the local tracking branch (e.g. origin/master) is moved to point to the most recent commit
The important thing to note is that the local branch does not change at all.

You can think of git fetch as half of a git pull. The other half of git pull is the merging aspect.

One main point when you want to use git fetch rather than git pull is if your remote branch and your local branch both have changes that neither of the other ones has. In this case, you want to fetch the remote changes to get them in your local branch and then perform a merge manually. Then you can push that new merge commit back to the remote.

In version control terminology if you "fork" a repository that means you duplicate it. Typically you fork a repository that belongs to someone else. So you make an identical copy of their repository and that duplicate copy now belongs to you.

This concept of "forking" is also different from "cloning". When you clone a repository, you get an identical copy of the repository. But cloning happens on your local machine and you clone a remote repository. When you fork a repository, a new duplicate copy of the remote repository is created. This new copy is also a remote repository, but it now belongs to you.

We can see from this little experiment that if a repository doesn't belong to your account then it means you do not have permission to modify it.

This is where forking comes in! Instead of modifying the original repository directly, if you fork the repository to your own account then you will have full control over that repository.

Forking is an action that's done on a hosting service, like GitHub. Forking a repository creates an identical copy of the original repository and moves this copy to your account. You have total control over this forked repository. Modifying your forked repository does not alter the original repository in any way.

This is not a massive project, but it does have well over 1,000 commits. A quick way that we can see how many commits each contributor has added to the repository is to use the git shortlog command:

`git shortlog` displays an alphabetical list of names and the commit messages that go along with them. If we just want to see just the number of commits that each developer has made, we can add a couple of flags: -s to show just the number of commits (rather than each commit's message) and -n to sort them numerically (rather than alphabetically by author name).

Filter By Author
Another way that we can display all of the commits by an author is to use the regular git log command but include the --author flag to filter the commits to the provided author.

`git log --author=Surma`

How about we filter down to just the commits that reference the word "bug". We can do that with either of the following commands:

`git log --grep=bug`
`git log --grep bug`

A pull request is a request to the original or source repository's maintainer to include changes in their project that you made in your fork of their project. You are requesting that they pull in changes you've made.

A pull request is a request for the source repository to pull in your commits and merge them with their project. To create a pull request, a couple of things need to happen:

you must fork the source repository
clone your fork down to your machine
make some commits (ideally on a topic branch!)
push the commits back to your fork
create a new pull request and choose the branch that has your new commits

One thing that can be a tiny bit confusing right now is the difference between the origin and upstream. What might be confusing is that origin does not refer to the source repository (also known as the "original" repository) that we forked from. Instead, it's pointing to our forked repository. So even though it has the word origin is not actually the original repository.

Remember that the names origin and upstream are just the default or de facto names that are used. If it's clearer for you to name your origin remote mine and the upstream remote source-repo, then by all means, go ahead and rename them. What you name your remote repositories in your local repository does not affect the source repository at all.

`git remote rename origin mine`
`git remote rename upstream source-repo`

When working with a project that you've forked. The original project's maintainer will continue adding changes to their project. You'll want to keep your fork of their project in sync with theirs so that you can include any changes they make.

To get commits from a source repository into your forked repository on GitHub you need to:

get the cloneable URL of the source repository
create a new remote with the git remote add command
use the shortname upstream to point to the source repository
provide the URL of the source repository
fetch the new upstream remote
merge the upstream's branch into a local branch
push the newly updated local branch to your origin repo


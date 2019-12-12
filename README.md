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

## Intro to Neural Networks

The design of the Artificial Neural Network was inspired by the biological one. The neurons used in the artificial network below are essentially mathematical functions.

Each network has:

Input neurons- which we refer to as the input layer of neurons
Output neurons- which we refer to as the output layer of neurons
Internal neurons- which we refer to as the hidden layer of neurons. Each neural network can have many hidden layers

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

**Classification problems**: we find the boundary line that predicts labels (0 or 1) the best way possible: $w_1x_1+w_2x_2+w_3x_3+b=0$ or $Wx+b=0$.

**Perceptrons**
Functions that combine inputs in some fashion and output a number from 0 to 1. A step function is applied to the prediction of the model.
If a point is misclassified, that point would want the line to come closer to it. The trick to make this happen is to multiply the point's coordinates to some percentage (referred to as learning rate) and add or subtract these to the line parameters. The bias is added/subtracted the learning rate.

The logical operators AND, OR, NOT and XOR can be modeled with perceptrons. Check out `perceptrons.py` to see code.

In real life, though, we can't be building these perceptrons ourselves. The idea is that we give them the result, and they build themselves.

Our error function will tell us the direction to go to. We take at each step the direction that will reduce the error the most. Our error function needs to be continuous as opposed to discrete and also differentiable.
So our predictions now need to be continuous as opposed to 0 or 1 only. Now we our model will need to output the likelihood that a point is classified in as a 1. No more "yes" or "no" but "X% likely". The closer to the line the point is, the greater the likelihood. We go from the step function to the **sigmoid** function that ranges from 0 to 1 continuously. The shape of the **activation function** is now given by: $\sigma(x)=1/(1+e^{-x})$.

![Continuous prediction with sigmoid function](./neural_networks/sigmoid.png)

The **softmax function** is used as an activation when there is more than two classes in the classification problem. It has to calculate probabilities that sum to 1. It also has to work with scores that are negative. We use exponential to convert every possible number into a positive number. The softmax formula is: $P(class \ i) = e^{z_i}/(e^{z_1}+e^{z_2}+...+e^{z_n})$. The value of the softmax function for a given input is exactly the same as the sigmoid for that input.

**Maximum Likelihood and Cross-Entropy**
The maximum likelihood is the idea of how accurate a model is at classifying most points correctly based on their expected labels. The multiplication of the probabilities for all points will be higher than of a less accurate model. We want to use sums instead to have a less sensitive result. So we take the -ln of the product to turn that into a sum of lns. The result of the sum is called **cross-entropy**. A good model will have a low cross entropy. We now want to minimize the cross entropy!  The cross entropy definitely connects probabilities and error functions. The cross-entropy is inversely proportional to the total probability of an outcome.

The formula is:
$$CE=-\sum_{i=1}^{m}y_iln(p_i)+(1-y_i)ln(1-p_i)$$
For more than two classes we write more generically:
$$CE=-\sum_{i=1}^{n}\sum_{j=1}^{m}y_{ij}ln(p_{ij})$$
Then our error function is:
$$E(W,b)=-1/m\sum_{i=1}^{m}(1-y_i)ln(1-\sigma(Wx_i+b)+y_iln(\sigma(Wx_i+b)))$$

**Gradient Descent**
We use the negative of the gradient of the error function as the direction to go that leads to the greatest decrease in the error function, $-\nabla E$. We use a learning rate $\alpha$ to take not too big of a step. We compute the new weights and bias as:
$$w'_i=w_i-\alpha \frac{\delta E}{\delta w_i}$$
$$b'=b-\alpha\frac{\delta E}{\delta b}$$
which assures that the prediction given by $\hat y=\sigma(W'x+b')$ is better than the former one.
The first thing to notice is that the sigmoid function has a really nice derivative:
$$\sigma'(x)=\sigma(x)(1-\sigma(x))$$
If we calculate the gradient vector, we find:
$$\nabla E=-(y-\hat y)(x_1,x_2,...x_n,1)$$
The gradient is actually a scalar times the coordinates of the point! And what is the scalar? Nothing less than a multiple of the difference between the label and the prediction. So, if a point is well classified, we will get a small gradient. And if it's poorly classified, the gradient will be quite large.

The gradient descent step will then be:
$$w_i'=w_i-\alpha [-(y-\hat y)x_i] \ or \ w_i+\alpha (y-\hat y)x_i$$
$$b'=b+\alpha (y-\hat y)$$
Note: Since we've taken the average of the errors, the term we are adding should be $\frac{1}{m} \cdot \alpha$ instead of $\alpha$ but as $\alpha$ is a constant, then in order to simplify calculations, we'll just take $\frac{1}{m} \cdot \alpha$ to be our learning rate, and abuse the notation by just calling it $\alpha$.

In the gradient descent even correctly classified points have the line move with respect to them, which is not the case for the perceptron algorithm (correctly classified points don't have their weights changed). These points actually have the line move farther away from them, so as to increase their probability of being correctly classified :)

**Non-linear models**
The non-linear boundary is where points have a 50% probability of belonging to one and the other regions.
We combine linear models to produce a non-linear one. For instance, we sum the outputs of two linear models and apply the
sigmoid function to produce a number between 0 and 1. We can use weights to use different weights of the two models too. We can have a bias too.

![Neural net arrangement](./neural_networks/neural_network_arrangement.png)

In deep neural networks we have more hidden layers where intermediate non-linear models combine to generate other non-linear models.

![Deep neural net arrangement](./neural_networks/deep_neural_network_arrangement.png)

And for multi-class classification, as opposed to binary, we can have multiple neural networks, each one to predict the probability of one of the classes and then apply the softmax function. This seems like overkill though. What we actually do is add more nodes to the output layer and each will be the probability of the input belonging to one of the possible classes. We apply the softmax to the different scores.

Feedforward process for combining the layers and find the non-linear output:

![Feedforward process](./neural_networks/feedforward.png)

Now, we're ready to get our hands into training a neural network. For this, we'll use the method known as backpropagation. In a nutshell, **backpropagation** will consist of:

- Doing a feedforward operation.
- Comparing the output of the model with the desired output.
- Calculating the error.
- Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
- Use this to update the weights, and get a better model.
- Continue this until we have a model that is good.

With backpropagation, we "listen" to the point that was misclassified and weigh in the models differently based on the classification they output with respect to that point.

![Backpropagation](./neural_networks/backpropagation.png)

![Backpropagation](./neural_networks/backpropagation_math.png)

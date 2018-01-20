# Bayesian estimation of group averages, for machine learning feature engineering

In machine learning problems, it is common to have categorical variables with a large number (hundreds or thousands)
of levels. In this case, memory constraints make one-hot encoding these variables impractical. Therefore, a common way
to use such a variable in the model is to create a feature which is the average of the prediction variable (whether binary
or continuous) in each category. (In pandas, this can be done with groupby and transform.)
However, this approach will give misleading estimates of the group mean for groups
where the training set contains only a few samples. In this situation, a high or low average is more likely to occur
just by chance. So we need to adjust the
estimated group average to make it more conservative. We can do this using Bayesian methods. We assume a prior
distribution based on the overall data set, and combine this with the sample data in each group to calculate a
Bayesian posterior estimate of the group average.

This module also adds some other useful functionality which is not available in pandas. There is the option to calculate
the group averages using only a subset of the dataframe (so we can calculate the averages only using the training data,
and avoid leaking the target variable of test data). There is also an
option to exclude the current row from the average, which avoids overfitting.



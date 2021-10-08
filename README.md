# Linear Gaussian Batch Estimation and RTS Smoother

We use the batch linear-Gaussian algorithm to estimate the robot's one-dimensional position from odometry and laser measurements.

![](out\meas_vs_t.png)

## Measurement Data Analysis

In order to perform linear-Gaussian estimation we must assume zero-mean normally distributed noise. Using the ground truth position information from the Vicon motion capture system, we can rearrange the motion and observation model to solve for the process and measurement noise.

The mean of the measurement model and motion model noise was computed to be 1.005d-15 m and 4.505d-5 m respectively, which is quite small relative to the mean true position magnitude 1.780 m. Therefore, the assumption of zero-mean measurement model noise is reasonable over small time scales. However, since the motion model noise has an additive effect this will lead to 0.45 m of error over 1000 seconds if left uncorrected.

![test](out\model_noise_vs_t.png)

The assumption that the noise follows a Gaussian distribution can be qualitatively assessed by plotting the histogram of the noise against a best fit normal distribution. We can also assess the fit using a Quantile-Quantile Plot, which shows the fraction of points below the given value in the data against that of the best first normal distribution. Based on these plots, it is reasonable to claim that the measurement and modified process noise roughly follow a normal distribution. However, it is not a perfect fit and can be seen that the measurement model noise has a significant deviation from normal at the edges of the distribution where there is less data.

![](out\hist_model_noise.png)

![](out\qq_model_noise.png)

The variances of the process model and observation model noise are
computed to be 0.00002 m^2 and 0.00037 m^2 respectively. However, these values were later found to provide an inconsistent variance estimate in the smoother due to overconfidence in the process model. For a more consistent smoother uncertainty, the variance of the process model noise was inflated to 0.002 m^2.

## Batch Estimation

Our goal is to find the most likely values for the state, meaning the state values that maximize the posterior probability density function given all our measurements of the state. We can solve the weighted least squares optimization problem using calculus to find minimum of the objective function as the derivative of the objective function with respect to the state variable.  This leads to a linear system of equations in the form of Ax=b. 

## Recursive Smoothing Method

In order to solve the system of equations we require the inverse of a large
matrix of size12709 x 12709. The matrix has a symmetric tridiagonal sparsity pattern. This pattern allows us to solve for the inverse more efficiently than
using traditional algorithms, such as Gaussian elimination, which typically run in O(n^3) time. Given the tridiagonal sparsity pattern, we can invert the matrix using the tridiagonal matrix algorithm (also known as the forward block tridiagonal or Thomas algorithm), Rauch-Tung-Striebel (RTS) Smoother, or Cholesky Smoother. All three of these algorithms can be shown to be equivalent. The algorithms use a single forward pass followed by a single backward substitution pass and run in O(n) time. Once A inverse is found we can compute the most likely state estimate and confidence in our solution as x* = inv(A) b, P* = inv(A).

The algorithm chosen is the RTS Smoother. The first predicted state estimate is assigned the first measurement and the first predicted state variance is assigned the measurement noise variance. For subsequent steps during the forward pass the predicted state variance and estimate is computed for the current step based on the previous state variance, estimate, and process model. If our iterator is divisible by the range measurement update interval, then we compute the Kalman gain, otherwise it is set to zero. When range measurements are available, we use the Kalman gain to solve for the corrected state variance and estimate using the optimal weighting of the predicted state and measurement. In the backwards pass we apply the backward substitution smoothing equation to solve for the optimal state estimate and variance.

## RTS Smoother Results

A Python script was created to solve for the robots position estimate
and uncertainty at all K time steps. Four cases were considered with
different range measurement update step sizes of and 1, 10, 100, 1000. Over each the trajectories we see that at each of the range measurement updates our uncertainty in our estimate is the smallest, while at the mid-point between measurement updates our uncertainty is the largest. This is to be expected since the range measurement provides an exteroceptive update which can correct any drift accumulated by the interoceptive odemetry measurements and the process model. In all scenarios a filter estimation error is consistent with the filter posterior variance was achieved, meaning that the estimation error rarely exceeds the three sigma uncertianty. As we increase the time step of the range measurement update we see increases in the error estimate variance and the upper bound of our uncertainty over the trajectory, however the lower bound of of the uncertainty does not change. However, these results required the manual tuning of the process noise, since the initial value found based on analysis of the odometery data led to an overconfident estimate. Since the real world process and model noise don't perfectly follow a zero-mean normal distribution it is expected that tuning is required for consistent results.

![](out\est_err_and_3std_1steps.png)

![](out\est_err_and_3std_100steps.png)

![](out\est_err_and_3std_1000steps.png)

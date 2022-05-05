# Packages requires to load the dataset
using Pkg
Pkg.add("DataFrames")
using DataFrames

# Packages installed
Pkg.status()

date = [1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,
    2012,2013,2014,2015,2016,2017,2018,2019];
    
pop = [2,1,4,2,1,4,2,2,1,6,7,7,8,8,8,8,11,11,13,21,21,36,24,41,46,46,59,53];
df = DataFrame( varx = date, vary = pop);

# Define time series for Chow test:
#point break = 2010
start_df1, end_df1 = 1992, 2010                                      # range for series 1 (obs:"end_df1 = point break")
start_df2, end_df2 = end_df1+1, 2019                                 # range for series 2
df1 = filter(r -> r.varx >= start_df1 && r.varx <= end_df1, df)      # time series before point break
df2 = filter(r -> r.varx >= start_df2 && r.varx <= end_df2, df)      # time series after point break

# Variables for Chow test:
x , y = copy(df[:,:varx]), copy(df[:,:vary])         # variables from original dataset
x1 , y1 = copy(df1[:,:varx]), copy(df1[:,:vary])     # variables from series before point break
x2 , y2 = copy(df2[:,:varx]), copy(df2[:,:vary])     # variables from series after point break

# Alternative formulas ***************************************************************************************************
# Original dataset ----------------------------------------------------------------------------------------------- 
k = 2                                        # number of samples (series before and after point break)
n = length(x)                                # total scores from original dataset
sum_x, sum_y = sum(x), sum(y)                # sum of independent and dependent variables from original series
mean_x, mean_y = sum(x)/n, sum(y)/n          # average of independent and dependent variables from original series
SSxy = sum((x.-mean_x).*(y.-mean_y))         # covariance of X and Y
SSxx = sum((x.-mean_x).^2)                   # variance in X
b = SSxy/SSxx                                # regression coefficient
a = mean_y - b*mean_x                        # intercept
yi = a .+ b.*x                               # estimated (or predicted) y value for observation i
RSSp = sum((y .- yi).^2)                     # Resiual sum of square (errors)
# Series 1 ------------------------------------------------------------------------------------------------------ 
n1 = length(x1)                              # total scores from series 1
gl1 = n1 - k                                 # numerator degrees freedom
sum_x1, sum_y1 = sum(x1), sum(y1)            # sum of independent and dependent variables from series 1
mean_x1, mean_y1 = sum(x1)/n1, sum(y1)/n1    # average of independent and dependent variables from series 1
SSxy1 = sum((x1.-mean_x1).*(y1.-mean_y1))    # covariance of X1 and Y1
SSxx1 = sum((x1.-mean_x1).^2)                # variance in X1
b1 = SSxy1/SSxx1                             # regression coefficient (series 1)
a1 = mean_y1 - b1*mean_x1                    # intercept (series 1)
yi1 = a1 .+ b1.*x1                           # estimated y value for observation i (series 1)
RSS1 = sum((y1 .- yi1).^2)                   # Residual sum of square before break
# Series 2 ------------------------------------------------------------------------------------------------------ 
n2 = length(x2)                              # total scores from series 2
gl2 = n2 - k                                 # denominator degrees freedom
sum_x2, sum_y2 = sum(x2), sum(y2)            # sum of independent and dependent variables from series 2
mean_x2, mean_y2 = sum(x2)/n2, sum(y2)/n2    # average of independent and dependent variables from series 2
SSxy2 = sum((x2.-mean_x2).*(y2.-mean_y2))    # covariance of X2 and Y2
SSxx2 = sum((x2.-mean_x2).^2)                # variance in X2
b2 = SSxy2/SSxx2                             # regression coefficient (series 2)
a2 = mean_y2 - b2*mean_x2                    # intercept (series 2)
yi2 = a2 .+ b2.*x2                           # estimated y value for observation i (series 2)
RSS2 = sum((y2 .- yi2).^2)                   # Residual sum of square after break

#Pkg.add("Folds")

using Folds

# Return regression coefficient
function mat_coeff(y, x)
    k = 2
    n = length(x)
    sum_x, sum_y = sum(x), sum(y)
    mean_x, mean_y = sum(x)/n, sum(y)/n
    SSxy = sum((x.-mean_x).*(y.-mean_y))
    SSxx = sum((x.-mean_x).^2)
    b = SSxy/SSxx
    return(b)
end

# Return regression coefficient
function mat_coeff1(y, x)
    k = 2
    n = length(x)
    sum_x, sum_y = sum(x), sum(y)
    mean_x, mean_y = sum(x)/n, sum(y)/n
    SSxy = Folds.sum((x.-mean_x).*(y.-mean_y))  # parallelism
    SSxx = Folds.sum((x.-mean_x).^2)            # parallelism
    b = SSxy/SSxx
    return(b)
end

# Results:
println(mat_coeff(y, x))
println(mat_coeff1(y, x))

# Packages requires to load the dataset
#using Pkg
#Pkg.add("BenchmarkTools")
using BenchmarkTools

@btime mat_coeff(y, x)

@btime mat_coeff1(y, x)

@benchmark mat_coeff(y,x)

@benchmark mat_coeff1(y,x)

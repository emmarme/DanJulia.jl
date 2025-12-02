module DanJulia

using Convex,SCS

# Generate random problem data
using Random,LinearAlgebra
Random.seed!(2)
n = 10
m = 20
sigma = 0.1
X = randn(m,n)
theta_true = rand(n)
xi = randn(m)
y = X * theta_true + sigma * xi

# Define the optimization variable
function p1(X, y)
    n = size(X, 2)
    theta = Variable(n)
    L = sumsquares(X * theta - y)
    problem = minimize(L, [theta >= 0])
    solve!(problem, SCS.Optimizer)
    println("theta_estime = ", theta.value)
    return theta.value
end

p1(X, y)
println("theta_vraie = ", theta_true)

# Define another optimization variable but thiss time with constraints
# la contraintes est que theta>=0 et sum(theta)<=S
function p2(X, y, S)
    n = size(X, 2)
    theta = Variable(n)
    L = sumsquares(X * theta - y)
    problem = minimize(L, [theta >= 0, sum(theta) <= S])
    solve!(problem, SCS.Optimizer)
    println("theta_estime = ", theta.value)
    return theta.value
end
# pour S=3
p2(X, y, 3.0)
println("theta_vraie = ", theta_true)
   

# faire varier S
for S in 1.0:1.0:10.0
    
    println("Pour S = ", S)
    p2(X, y, S)
    println("theta_vraie = ", theta_true)
    
end

end
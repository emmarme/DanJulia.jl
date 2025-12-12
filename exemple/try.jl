using GLMakie
using Random,LinearAlgebra
using DanJulia
using Convex, SCS

Random.seed!(2)
n = 10
m = 20
sigma = 0.1
X = randn(m,n)
theta_true = rand(n)
xi = randn(m)
y = X * theta_true + sigma * xi

DanJulia.p1(X, y)
println("theta_vraie = ", theta_true)

# pour S=3
DanJulia.p2(X, y, 3.0)
println("theta_vraie = ", theta_true)

errors = []
S_values = 1.0:1.0:10.0
for S in S_values
    theta_estime = DanJulia.p2(X, y, S)
    error = norm(theta_estime - theta_true, 1)
    push!(errors, error)
end 

fig_errors = Figure()
ax = Axis(fig_errors[1, 1], xlabel="S", ylabel="Erreur (norme 1)", title="Erreur entre theta_estime et theta_vraie en fonction de S")
lines!(ax, S_values, errors)
display(fig_errors)
save("errors_vs_S.png", fig_errors)

#trouver le S optimal
S_opt, min_error, fig_S_opt = DanJulia.S_optimal(X, y, theta_true)
println("S optimal et erreur minimale obtenue = ", S_opt, ", ", min_error)
display(fig_S_opt)
save("S_optimal_plot.png", fig_S_opt)

#valeurs duales
dual_values_inf, dual_values_sum = DanJulia.solve_p2_duale(X,y,1.0)
println("Valeurs duales pour S=1 : inf = ", dual_values_inf, ", sup = ", dual_values_sum)

fig_dual = DanJulia.dual_values_vs_S(X, y, 1.0:0.5:10.0)
display(fig_dual)
save("dual_values_plot.png", fig_dual)


# comparaison θ_est vs θ_true pour S_opt
theta_est_opt = DanJulia.p2(X, y, S_opt)
fig_theta = DanJulia.plot_theta_comparison(theta_est_opt, theta_true)
display(fig_theta)
save("theta_comparison.png", fig_theta)

# résidu en fonction de S
fig_resid, res = DanJulia.residual_vs_S(X, y, S_values)
display(fig_resid)
save("residual_vs_S.png", fig_resid)

# chemin de regulas θᵢ(S)
fig_path, path = DanJulia.theta_path(X, y, S_values)
display(fig_path)
save("theta_path.png", fig_path)

# verifier KKT pour S_opt
dual_inf_opt, dual_sum_opt = DanJulia.solve_p2_duale(X, y, S_opt)
kkt_info = DanJulia.check_KKT(X, y, theta_est_opt, dual_inf_opt, dual_sum_opt, S_opt)
println("Vérification KKT pour S_opt : ", kkt_info)
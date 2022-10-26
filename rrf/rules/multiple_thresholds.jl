using JuMP, Gurobi, Random

println("Starting parameter definition")
n = 1000
M = 10*n
n_pass = 700
MAX_FP = 50
p = 9

r = 1:n
for i in 1:p-1
    global r = hcat(r, shuffle(1:n))
end

P = Set(shuffle(1:n)[1:n_pass])
NP = setdiff(1:n, P)

println("Defining model")
m = Model(Gurobi.Optimizer)

@variable(m, x[i=1:n], Bin)
@variable(m, y[i=1:n,j=1:p], Bin)
@variable(m, R[j=1:p])
#@variable(m, 1 <= R[j=1:p] <= n, Int)

@objective(m, Max, sum(x[i] for i in P))

@constraint(m, fp, sum(x[i] for i in NP) <= MAX_FP)
@constraint(m, inclusion[i=1:n,j=1:p], r[i,j] <= R[j] + M*(1-x[i]))
@constraint(m, exclusion[i=1:n,j=1:p], R[j] + 1 <= r[i,j] + M*(1-y[i,j]))
@constraint(m, exclusion_sum_min[i=1:n], sum(y[i,j] for j in 1:p) >= 1 - x[i])
@constraint(m, exclusion_sum_max[i=1:n], sum(y[i,j] for j in 1:p) <= M*(1 - x[i]))

println("Solving model")
optimize!(m)

println(solve_time(m))


using JuMP
import SCIP
import StatsFuns
using ProgressMeter
using NPZ

struct ValueDistributionMoments
    var::Float64
    third_absolute::Float64
end

function load_moments(mdp_fname::String, num_states::Integer, horizon::Integer)
    moments = Array{ValueDistributionMoments}(undef, horizon, num_states)
    for t = 1:horizon
        value_dists = npzread(splitext(mdp_fname)[1] * "_analyzed_value_dists_$(t).npy")
        dist_size = size(value_dists)[2]
        for state = 1:num_states
            mean::Float64 = 0
            for dist_index = 1:dist_size
                value = value_dists[1, dist_index, state]
                prob = value_dists[2, dist_index, state]
                mean += value * prob
            end
            var::Float64 = 0
            third_absolute::Float64 = 0
            for dist_index = 1:dist_size
                centralized_value = value_dists[1, dist_index, state] - mean
                prob = value_dists[2, dist_index, state]
                var += prob * centralized_value^2
                third_absolute += prob * abs(centralized_value)^3
            end
            moments[t, state] = ValueDistributionMoments(var, third_absolute)
        end
    end
    moments
end

function berry_esseen_bound(moments::ValueDistributionMoments, m::Integer)
    if moments.var == 0
        return 0
    end
    std_cubed = moments.var^1.5
    return (
        min(
            0.33554 * (moments.third_absolute + 0.415 * std_cubed),
            0.3328 * (moments.third_absolute + 0.429 * std_cubed),
        ) / (std_cubed * sqrt(Float64(m)))
    )
end

function is_mdp_binomial(rewards::Array{Reward,2}, vi::ValueIterationResults)
    max_reward = maximum(vi.optimal_values)
    for reward in rewards
        if !(reward == max_reward || reward == 0)
            return false
        end
    end
    return true
end

struct MDPForAnalysis
    transitions::Array{Int,2}
    rewards::Array{Reward,2}
    horizon::Int
    vi::ValueIterationResults
    moments::Union{Nothing,Array{ValueDistributionMoments,2}}
    is_binomial::Bool
end

function MDPForAnalysis(
    transitions::Array{Int,2},
    rewards::Array{Reward,2},
    horizon::Int,
    vi::ValueIterationResults,
    moments::Union{Nothing,Array{ValueDistributionMoments,2}},
)
    is_binomial = is_mdp_binomial(rewards, vi)
    if is_binomial
        max_reward = vi.optimal_values[1, 1]
        rewards ./= max_reward
        vi.exploration_qs ./= max_reward
        vi.exploration_values ./= max_reward
        vi.optimal_qs ./= max_reward
        vi.optimal_values ./= max_reward
        vi.worst_qs ./= max_reward
        vi.worst_values ./= max_reward
    end
    MDPForAnalysis(transitions, rewards, horizon, vi, moments, is_binomial)
end

struct FailureProbabilityLinearProgram
    model::JuMP.Model
    action_seq_probs::Vector{JuMP.VariableRef}
    action_seq_prob_lower_constraints::Vector{JuMP.ConstraintRef}
    action_seq_prob_upper_constraints::Vector{JuMP.ConstraintRef}
end

function FailureProbabilityLinearProgram(num_actions::Integer, k::Integer)
    num_action_seqs = num_actions^k
    action_seqs = 1:num_action_seqs
    action_seq_prob_lowers = zeros(Float64, num_action_seqs)
    action_seq_prob_uppers = ones(Float64, num_action_seqs)
    model = Model(SCIP.Optimizer; add_bridges = false)
    set_silent(model)
    @variable(model, action_seq_probs[action_seqs])
    action_seq_prob_lower_constraints = @constraint(
        model,
        [action_seq in action_seqs],
        action_seq_probs[action_seq] >= action_seq_prob_lowers[action_seq]
    )
    action_seq_prob_upper_constraints = @constraint(
        model,
        [action_seq in action_seqs],
        action_seq_probs[action_seq] <= action_seq_prob_uppers[action_seq]
    )
    @constraint(model, sum(action_seq_probs) == 1)
    @objective(
        model,
        Max,
        sum(action_seq_probs[action_seq] for action_seq in action_seqs)
    )
    FailureProbabilityLinearProgram(
        model,
        action_seq_probs,
        action_seq_prob_lower_constraints,
        action_seq_prob_upper_constraints,
    )
end


struct FailureProbabilityThreadContext
    resolution::Float64
    k_step_exploration_qs::Vector{Float64}
    k_step_optimal_qs::Vector{Float64}
    k_step_worst_qs::Vector{Float64}
    k_step_moments::Vector{ValueDistributionMoments}
    action_seq_prob_lowers::Vector{Float64}
    action_seq_prob_uppers::Vector{Float64}
    action_failure_probs::Vector{Float64}
    quantiles::Vector{Int64}
    float_quantiles::Vector{Float64}
    failure_prob_lp::FailureProbabilityLinearProgram
    lock::ReentrantLock
end

function FailureProbabilityThreadContext(num_actions::Integer, k::Integer)
    resolution = 0.01
    FailureProbabilityThreadContext(
        resolution,
        Vector{Float64}(undef, num_actions^k),
        Vector{Float64}(undef, num_actions^k),
        Vector{Float64}(undef, num_actions^k),
        Vector{ValueDistributionMoments}(undef, num_actions^k),
        Vector{Float64}(undef, num_actions^k),
        Vector{Float64}(undef, num_actions^k),
        Vector{Float64}(undef, num_actions),
        Vector{Int64}(undef, length(0:resolution:1)),
        Vector{Int64}(undef, length(0:resolution:1)),
        FailureProbabilityLinearProgram(num_actions, k),
        ReentrantLock(),
    )
end

function compute_failure_prob(mdp::MDPForAnalysis, k::Int, m::Integer)
    num_states, num_actions = size(mdp.transitions)

    states_optimal = zeros(Bool, mdp.horizon, num_states)
    states_optimal[1, 1] = true
    num_optimal_states::Int64 = 1
    for t = 1:mdp.horizon-1
        for state = 1:num_states
            if states_optimal[t, state]
                for action = 1:num_actions
                    if (
                        mdp.vi.optimal_qs[t, state, action] >
                        mdp.vi.optimal_values[t, state] - REWARD_PRECISION
                    )
                        next_state = mdp.transitions[state, action] + 1
                        if !states_optimal[t+1, next_state]
                            num_optimal_states += 1
                            states_optimal[t+1, next_state] = true
                        end
                    end
                end
            end
        end
    end

    thread_contexts = Vector{FailureProbabilityThreadContext}(undef, Threads.nthreads())
    for thread_id = 1:Threads.nthreads()
        thread_contexts[thread_id] = FailureProbabilityThreadContext(num_actions, k)
    end

    state_failure_probs = Array{Float64}(undef, mdp.horizon, num_states)
    state_failure_probs .= NaN64
    progress = Progress(num_optimal_states; desc = "m = $(m)")
    ProgressMeter.update!(progress, 0)
    for t = mdp.horizon:-1:1
        optimal_states = findall(states_optimal[t, :])
        Threads.@threads for optimal_state_index = length(optimal_states):-1:1
            state = optimal_states[optimal_state_index][1]
            state_failure_probs[t, state] = compute_failure_prob(
                mdp,
                state,
                t,
                k,
                m,
                state_failure_probs,
                thread_contexts[Threads.threadid()],
            )
            next!(progress)
        end
    end
    finish!(progress)

    state_failure_probs[1, 1]
end

function get_k_step_qs(
    mdp::MDPForAnalysis,
    state::Int,
    t::Int,
    k::Int,
    exploration_qs::AbstractVector{Float64},
    optimal_qs::AbstractVector{Float64},
    worst_qs::AbstractVector{Float64},
    moments::AbstractVector{ValueDistributionMoments},
)
    num_actions = size(mdp.transitions)[2]
    if t > mdp.horizon
        exploration_qs .= 0
        optimal_qs .= 0
        worst_qs .= 0
        for action_seq = 1:length(moments)
            moments[action_seq] = ValueDistributionMoments(0, 0)
        end
    elseif k == 1
        for action = 1:num_actions
            exploration_qs[action] = mdp.vi.exploration_qs[t, state, action]
            optimal_qs[action] = mdp.vi.optimal_qs[t, state, action]
            worst_qs[action] = mdp.vi.worst_qs[t, state, action]
            if mdp.moments !== nothing
                next_state = mdp.transitions[state, action] + 1
                if t + 1 <= mdp.horizon
                    moments[action] = mdp.moments[t+1, next_state]
                else
                    moments[action] = ValueDistributionMoments(0, 0)
                end
            end
        end
    else
        for action = 1:num_actions
            next_state = mdp.transitions[state, action] + 1
            reward = mdp.rewards[state, action]
            sub_slice = associated_action_seqs(action, num_actions, k)
            sub_exploration_qs = @view exploration_qs[sub_slice]
            sub_optimal_qs = @view optimal_qs[sub_slice]
            sub_worst_qs = @view worst_qs[sub_slice]
            sub_moments = @view moments[sub_slice]
            get_k_step_qs(
                mdp,
                next_state,
                t + 1,
                k - 1,
                sub_exploration_qs,
                sub_optimal_qs,
                sub_worst_qs,
                sub_moments,
            )
            sub_exploration_qs .+= reward
            sub_optimal_qs .+= reward
            sub_worst_qs .+= reward
        end
    end
end

associated_action_seqs(action::Integer, num_actions::Integer, k::Integer) =
    (action-1)*num_actions^(k-1)+1:action*num_actions^(k-1)
associated_action(action_seq::Integer, num_actions::Integer, k::Integer) =
    (action_seq - 1) ÷ (num_actions^(k - 1)) + 1

const ACTION_SEQ_PROB_BOUNDS_METHODS =
    Symbol[:binomial, :berry_esseen, :bernstein, :bennett]

function compute_failure_prob(
    mdp::MDPForAnalysis,
    state::Int,
    t::Int,
    k::Int,
    m::Integer,
    state_failure_probs::Array{Float64,2},
    thread_context::FailureProbabilityThreadContext,
)
    lock(thread_context.lock) do
        num_states, num_actions = size(mdp.transitions)
        num_action_seqs = num_actions^k

        action_failure_probs = thread_context.action_failure_probs
        for action = 1:num_actions
            # Leave some wiggle room when determining if the action is optimal in order
            # to deal with floating point errors. None of the MDPs should have reward
            # increments of less than 1e-3 so this should be fine.
            if mdp.vi.optimal_qs[t, state, action] <
               mdp.vi.optimal_values[t, state] - REWARD_PRECISION
                action_failure_probs[action] = 1
            elseif t == mdp.horizon
                action_failure_probs[action] = 0
            else
                next_state = mdp.transitions[state, action] + 1
                action_failure_probs[action] = state_failure_probs[t+1, next_state]
            end
        end

        get_k_step_qs(
            mdp,
            state,
            t,
            k,
            thread_context.k_step_exploration_qs,
            thread_context.k_step_optimal_qs,
            thread_context.k_step_worst_qs,
            thread_context.k_step_moments,
        )

        lowest_failure_prob = 1
        for method in ACTION_SEQ_PROB_BOUNDS_METHODS
            if method == :binomial
                if mdp.is_binomial && k == 1 && m <= 1000000
                    compute_binomial_action_prob_bounds(thread_context, m)
                else
                    continue
                end
            elseif method == :berry_esseen
                if mdp.moments !== nothing && num_action_seqs <= 100
                    compute_berry_esseen_action_prob_bounds(thread_context, m)
                else
                    continue
                end
            elseif method == :bernstein
                if num_action_seqs <= 100
                    compute_bernstein_action_prob_bounds(thread_context, m)
                else
                    continue
                end
            elseif method == :bennett
                compute_bennett_action_prob_bounds(thread_context, m)
            else
                @assert false
            end

            action_seq_prob_lowers = thread_context.action_seq_prob_lowers
            action_seq_prob_uppers = thread_context.action_seq_prob_uppers
            sum_under_1 = 1 - sum(action_seq_prob_uppers)
            if sum_under_1 > 0
                # Seems to happen sometimes due to numerical issues.
                for action_seq = 1:num_action_seqs
                    action_seq_prob_uppers[action_seq] =
                        min(1, action_seq_prob_uppers[action_seq] + sum_under_1)
                end
            end

            failure_prob_lower = 0
            failure_prob_upper = 0
            max_action_failure_prob = 0
            for action = 1:num_actions
                for action_seq in associated_action_seqs(action, num_actions, k)
                    @assert isfinite(action_seq_prob_lowers[action_seq])
                    @assert isfinite(action_seq_prob_uppers[action_seq])

                    failure_prob_lower +=
                        action_seq_prob_lowers[action_seq] *
                        action_failure_probs[action]
                    failure_prob_upper +=
                        action_seq_prob_uppers[action_seq] *
                        action_failure_probs[action]
                end
                if action_failure_probs[action] > max_action_failure_prob
                    max_action_failure_prob = action_failure_probs[action]
                end
            end
            failure_prob_upper = min(failure_prob_upper, max_action_failure_prob)
            if failure_prob_upper - failure_prob_lower > 0.01
                failure_prob = max_failure_prob(
                    thread_context.failure_prob_lp,
                    action_failure_probs,
                    action_seq_prob_lowers,
                    action_seq_prob_uppers,
                )
            else
                failure_prob = failure_prob_upper
            end
            if failure_prob < lowest_failure_prob
                lowest_failure_prob = failure_prob
            end
        end
        lowest_failure_prob
    end
end

function bennett_tail_bound(m::Integer, bound::Float64, var::Float64, t::Float64)
    if t <= 0
        return 1
    elseif var == 0
        return 0
    end
    mf = Float64(m)
    u = bound * t / var
    h_u = (1 + u) * Base.log1p(u) - u
    exp(-mf * var / (bound^2) * h_u)
end

function compute_bennett_action_prob_bounds(
    thread_context::FailureProbabilityThreadContext,
    m::Integer,
)
    num_action_seqs = length(thread_context.k_step_exploration_qs)
    max_exploration_q = maximum(thread_context.k_step_exploration_qs)
    next_highest_exploration_q = -Inf64
    for action_seq = 1:num_action_seqs
        action_q = thread_context.k_step_exploration_qs[action_seq]
        if action_q < max_exploration_q - REWARD_PRECISION &&
           action_q > next_highest_exploration_q
            next_highest_exploration_q = action_q
        end
    end
    if next_highest_exploration_q == -Inf64
        thread_context.action_seq_prob_uppers .= 1
        thread_context.action_seq_prob_lowers .= 0
        return
    end
    threshold = (max_exploration_q + next_highest_exploration_q) / 2

    upper_prob_all_max_below_threshold = 1
    for action_seq = 1:num_action_seqs
        exploration_q = thread_context.k_step_exploration_qs[action_seq]
        if exploration_q == max_exploration_q
            optimal_q = thread_context.k_step_optimal_qs[action_seq]
            worst_q = thread_context.k_step_worst_qs[action_seq]
            action_var = (exploration_q - worst_q) * (optimal_q - worst_q)
            upper_prob_below_threshold = bennett_tail_bound(
                m,
                optimal_q - worst_q,
                action_var,
                exploration_q - threshold,
            )
            upper_prob_all_max_below_threshold *= upper_prob_below_threshold
        end
    end

    for action_seq = 1:num_action_seqs
        thread_context.action_seq_prob_lowers[action_seq] = 0
        exploration_q = thread_context.k_step_exploration_qs[action_seq]
        optimal_q = thread_context.k_step_optimal_qs[action_seq]
        worst_q = thread_context.k_step_worst_qs[action_seq]
        action_seq_var = (exploration_q - worst_q) * (optimal_q - exploration_q)
        upper_prob_above_threshold = bennett_tail_bound(
            m,
            optimal_q - worst_q,
            action_seq_var,
            threshold - exploration_q,
        )
        thread_context.action_seq_prob_uppers[action_seq] =
            1 - (
                (1 - upper_prob_above_threshold) *
                (1 - upper_prob_all_max_below_threshold)
            )
    end
end

function bernstein_tail_bound(m::Integer, bound::Real, var::Real, t::Real)
    if t <= 0
        return 1
    elseif t == Inf
        return 0
    end
    exp(-t^2 * Float64(m) / 2 / (var + bound * t / 3))
end

function inv_bernstein_tail_bound(m::Integer, bound::Real, var::Real, delta::Real)
    if bound == 0 || var == 0
        return 0
    end
    log_1_over_delta = -Base.log(delta)
    mf = Float64(m)
    1 / 3 * (
        bound * log_1_over_delta / mf +
        sqrt(18 * var / mf * log_1_over_delta + bound^2 * log_1_over_delta^2 / mf^2)
    )
end

function compute_bernstein_action_prob_bounds(
    thread_context::FailureProbabilityThreadContext,
    m::Integer,
)
    num_action_seqs = length(thread_context.action_seq_prob_lowers)
    resolution = thread_context.resolution
    prob_lowers = thread_context.action_seq_prob_lowers
    prob_uppers = thread_context.action_seq_prob_uppers

    prob_lowers .= 0
    prob_uppers .= 0

    for action_seq = 1:num_action_seqs
        exploration_q = thread_context.k_step_exploration_qs[action_seq]
        optimal_q = thread_context.k_step_optimal_qs[action_seq]
        worst_q = thread_context.k_step_worst_qs[action_seq]
        var = (optimal_q - exploration_q) * (exploration_q - worst_q)
        bound = optimal_q - worst_q
        for quantile = resolution:resolution:1
            quantile_upper::Float64 =
                exploration_q + inv_bernstein_tail_bound(m, bound, var, 1 - quantile)
            quantile_lower::Float64 =
                exploration_q -
                inv_bernstein_tail_bound(m, bound, var, quantile - resolution)
            quantile_prob_lower::Float64 = resolution
            quantile_prob_upper::Float64 = resolution
            for other_action_seq = 1:num_action_seqs
                if other_action_seq == action_seq
                    continue
                end
                other_exploration_q =
                    thread_context.k_step_exploration_qs[other_action_seq]
                other_optimal_q = thread_context.k_step_optimal_qs[other_action_seq]
                other_worst_q = thread_context.k_step_worst_qs[other_action_seq]
                other_var =
                    (other_optimal_q - other_exploration_q) *
                    (other_exploration_q - other_worst_q)
                other_bound = other_optimal_q - other_worst_q
                quantile_prob_upper *= bernstein_tail_bound(
                    m,
                    other_bound,
                    other_var,
                    other_exploration_q - quantile_upper,
                )
                quantile_prob_lower *=
                    1 - bernstein_tail_bound(
                        m,
                        other_bound,
                        other_var,
                        quantile_lower - other_exploration_q,
                    )
            end
            prob_lowers[action_seq] += quantile_prob_lower
            prob_uppers[action_seq] += quantile_prob_upper
        end
    end
end

function binomcdf(n::Int64, p::Float64, k::Int64)::Float64
    ccall(
        (:binomcdf, "./src/binom/build/libbinom.so"),
        Float64,
        (Int64, Float64, Int64),
        n,
        p,
        k,
    )
end

function binominvcdf(n::Int64, p::Float64, q::Float64)::Int64
    ccall(
        (:binominvcdf, "./src/binom/build/libbinom.so"),
        Int64,
        (Int64, Float64, Float64),
        n,
        p,
        q,
    )
end

function compute_binomial_action_prob_bounds(
    thread_context::FailureProbabilityThreadContext,
    m_integer::Integer,
)
    m = Int64(m_integer)
    q_values = thread_context.k_step_exploration_qs

    num_actions = length(q_values)
    resolution = thread_context.resolution
    prob_lowers = thread_context.action_seq_prob_lowers
    prob_uppers = thread_context.action_seq_prob_uppers
    quantiles = thread_context.quantiles

    prob_lowers .= 0
    prob_uppers .= 0

    for action = 1:num_actions
        p = q_values[action]
        mean = p * m
        std = (m * p * (1 - p))^0.5
        for (quantile_index, q) in enumerate(0:resolution:1)
            if m >= 30
                quantiles[quantile_index] =
                    floor(Int64, max(0, min(m, StatsFuns.norminvcdf(mean, std, q))))
            else
                quantiles[quantile_index] = binominvcdf(m, p, q)
            end
        end

        quantile_index::Int64 = 1
        while quantile_index <= length(quantiles) - 1
            quantile_lower = zero(Float64)
            quantile_upper = zero(Float64)
            while quantile_lower == quantile_upper &&
                quantile_index <= length(quantiles) - 1
                quantile_lower = quantiles[quantile_index]
                quantile_upper = quantiles[quantile_index+1]
                quantile_index += 1
            end
            # quantile_prob = P(quantile_lower < x <= quantile_upper)
            quantile_prob =
                binomcdf(m, p, quantile_upper) - binomcdf(m, p, quantile_lower)
            quantile_prob_lower::Float64 = quantile_prob
            quantile_prob_upper::Float64 = quantile_prob
            for other_action = 1:num_actions
                if other_action == action
                    continue
                end
                other_p = q_values[other_action]
                quantile_prob_lower *= binomcdf(m, other_p, quantile_lower)
                quantile_prob_upper *= binomcdf(m, other_p, quantile_upper)
            end
            prob_lowers[action] += quantile_prob_lower
            prob_uppers[action] += quantile_prob_upper
        end
    end
end

function compute_normal_action_prob_bounds(
    thread_context::FailureProbabilityThreadContext,
    m::Integer,
)
    num_action_seqs = length(q_values)
    resolution = thread_context.resolution
    prob_lowers = thread_context.action_seq_prob_lowers
    prob_uppers = thread_context.action_seq_prob_uppers
    quantiles = thread_context.float_quantiles

    prob_lowers .= 0
    prob_uppers .= 0

    for action_seq = 1:num_action_seqs
        mean = thread_context.k_step_exploration_qs[action_seq]
        std = sqrt(
            (thread_context.k_step_optimal_qs[action_seq] - mean) *
            (mean - thread_context.k_step_worst_qs[action_seq]) / Float64(m),
        )
        for (quantile_index, q) in enumerate(0:resolution:1)
            quantiles[quantile_index] = StatsFuns.norminvcdf(mean, std, q)
        end
        quantiles[2] = mean - max(std * 10, 1e-4)
        quantiles[length(quantiles)-1] = mean + max(std * 10, 1e-4)

        quantile_index::Int64 = 1
        while quantile_index <= length(quantiles) - 1
            quantile_lower = zero(Float64)
            quantile_upper = zero(Float64)
            while quantile_lower == quantile_upper &&
                quantile_index <= length(quantiles) - 1
                quantile_lower = quantiles[quantile_index]
                quantile_upper = quantiles[quantile_index+1]
                quantile_index += 1
            end
            # quantile_prob = P(quantile_lower < x <= quantile_upper)
            quantile_prob = (
                StatsFuns.normcdf(mean, std, quantile_upper) -
                StatsFuns.normcdf(mean, std, quantile_lower)
            )
            quantile_prob_lower::Float64 = quantile_prob
            quantile_prob_upper::Float64 = quantile_prob
            for other_action_seq = 1:num_action_seqs
                if other_action_seq == action_seq
                    continue
                end
                other_mean = thread_context.k_step_exploration_qs[other_action_seq]
                other_std = sqrt(
                    (thread_context.k_step_optimal_qs[other_action_seq] - other_mean) * (
                        other_mean - thread_context.k_step_worst_qs[other_action_seq]
                    ) / Float64(m),
                )
                quantile_prob_lower *=
                    StatsFuns.normcdf(other_mean, other_std, quantile_lower)
                quantile_prob_upper *=
                    StatsFuns.normcdf(other_mean, other_std, quantile_upper)
            end
            prob_lowers[action_seq] += quantile_prob_lower
            prob_uppers[action_seq] += quantile_prob_upper
        end
    end
end

function compute_berry_esseen_action_prob_bounds(
    thread_context::FailureProbabilityThreadContext,
    m::Integer,
)
    num_action_seqs = length(thread_context.action_seq_prob_lowers)
    resolution = thread_context.resolution
    prob_lowers = thread_context.action_seq_prob_lowers
    prob_uppers = thread_context.action_seq_prob_uppers

    prob_lowers .= 0
    prob_uppers .= 0

    for action_seq = 1:num_action_seqs
        mean = thread_context.k_step_exploration_qs[action_seq]
        std = (thread_context.k_step_moments[action_seq].var / Float64(m))^0.5
        be_bound = berry_esseen_bound(thread_context.k_step_moments[action_seq], m)
        for quantile = resolution:resolution:1
            quantile_upper::Float64 =
                StatsFuns.norminvcdf(mean, std, min(1, quantile + be_bound))
            quantile_lower::Float64 = StatsFuns.norminvcdf(
                mean,
                std,
                max(0, quantile - resolution - be_bound),
            )
            quantile_prob_lower::Float64 = resolution
            quantile_prob_upper::Float64 = resolution
            for other_action_seq = 1:num_action_seqs
                if other_action_seq == action_seq
                    continue
                end
                other_mean = thread_context.k_step_exploration_qs[other_action_seq]
                other_std =
                    (
                        thread_context.k_step_moments[other_action_seq].var /
                        Float64(m)
                    )^0.5
                other_be_bound = berry_esseen_bound(
                    thread_context.k_step_moments[other_action_seq],
                    m,
                )
                quantile_prob_upper *= min(
                    1,
                    StatsFuns.normcdf(other_mean, other_std, quantile_upper) +
                    other_be_bound,
                )
                quantile_prob_lower *= max(
                    0,
                    StatsFuns.normcdf(other_mean, other_std, quantile_lower - 1e-8) - other_be_bound,
                )
            end
            prob_lowers[action_seq] += quantile_prob_lower
            prob_uppers[action_seq] += quantile_prob_upper
        end
    end
end

function max_failure_prob(
    lp::FailureProbabilityLinearProgram,
    action_failure_probs::Vector{Float64},
    action_seq_prob_lowers::Vector{Float64},
    action_seq_prob_uppers::Vector{Float64},
)
    num_actions = length(action_failure_probs)
    num_action_seqs = length(action_seq_prob_lowers)
    k = round(Int, Base.log(Float64(num_actions), Float64(num_action_seqs)))
    action_seqs = 1:num_action_seqs

    for action_seq in action_seqs
        set_normalized_rhs(
            lp.action_seq_prob_lower_constraints[action_seq],
            action_seq_prob_lowers[action_seq],
        )
        set_normalized_rhs(
            lp.action_seq_prob_upper_constraints[action_seq],
            action_seq_prob_uppers[action_seq],
        )
        set_objective_coefficient(
            lp.model,
            lp.action_seq_probs[action_seq],
            action_failure_probs[associated_action(action_seq, num_actions, k)],
        )
    end

    optimize!(lp.model)
    objective_value(lp.model)
end

mutable struct EffectiveHorizonResults
    ks::Vector{Int32}
    ms::Vector{BigInt}
    vars::Vector{Float64}
    gaps::Vector{Float64}
    effective_horizon::Float64
end

function compute_simple_effective_horizon(
    transitions::Matrix{Int},
    rewards::Matrix{Float32},
    horizon::Int;
    exploration_policy::Union{Nothing,Array{Float32,3}} = nothing,
)
    num_states, num_actions = size(transitions)
    vi = value_iteration(
        transitions,
        rewards,
        horizon,
        exploration_policy = exploration_policy,
    )

    var_bounds =
        TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
    for timestep in ProgressBar(1:horizon)
        Threads.@threads for state in vi.visitable_states[timestep]
            for action = 1:num_actions
                q = vi.exploration_qs[timestep, state, action]
                worst_q = vi.worst_qs[timestep, state, action]
                optimal_q = vi.optimal_qs[timestep, state, action]
                var_bound = (q - worst_q) * (optimal_q - worst_q)
                var_bounds[timestep, state, action] = var_bound
            end
        end
    end

    current_qs = vi.exploration_qs

    results = EffectiveHorizonResults(
        Vector{Int32}(undef, 0),
        Vector{BigInt}(undef, 0),
        Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0),
        horizon,
    )

    k = 1
    while k < results.effective_horizon
        k_works = Threads.Atomic{Bool}(true)
        state_ms = TimestepStateDictArray{BigInt,2,0}(0, horizon, num_states)
        state_vars = TimestepStateDictArray{Float64,2,0}(0, horizon, num_states)
        state_gaps = TimestepStateDictArray{Float64,2,0}(0, horizon, num_states)
        states_can_be_visited =
            TimestepStateDictArray{Bool,2,0}(false, horizon, num_states)
        states_can_be_visited[1, 1] = true
        timesteps_iter = ProgressBar(1:horizon)
        set_description(timesteps_iter, "Trying k = $(k)")
        for timestep in timesteps_iter
            Threads.@threads for state in vi.visitable_states[timestep]
                if states_can_be_visited[timestep, state]
                    max_q = -Inf64
                    max_suboptimal_q = -Inf64
                    max_var = 0
                    for action = 1:num_actions
                        q = current_qs[timestep, state, action]
                        max_q = max(max_q, q)
                        if (
                            vi.optimal_qs[timestep, state, action] <
                            vi.optimal_values[timestep, state] - REWARD_PRECISION
                        )
                            max_suboptimal_q = max(max_suboptimal_q, q)
                        end
                        max_var = max(max_var, var_bounds[timestep, state, action])
                    end

                    if max_q == max_suboptimal_q
                        k_works[] = false
                    else
                        gap = max_q - max_suboptimal_q
                        state_gaps[timestep, state] = gap
                        state_vars[timestep, state] = max_var
                        m = ceil(
                            BigInt,
                            16 * max_var / (gap^2) *
                            Base.log(2 * horizon * Float64(num_actions)^k),
                        )
                        m = max(1, m)
                        state_ms[timestep, state] = m
                    end

                    for action = 1:num_actions
                        if current_qs[timestep, state, action] > max_suboptimal_q
                            next_state = transitions[state, action] + 1
                            if timestep < horizon
                                states_can_be_visited[timestep+1, next_state] = true
                            end
                        end
                    end
                end
            end
            if !k_works[]
                break
            end
        end

        if k_works[]
            push!(results.ks, k)
            highest_m, timestep_state = findmax(state_ms)
            timestep, state = Tuple(timestep_state)
            push!(results.ms, highest_m)
            H_k = k + Base.log(num_actions, highest_m)
            println("H_$(k) = $(H_k)")
            push!(results.gaps, state_gaps[timestep, state])
            push!(results.vars, state_vars[timestep, state])
            results.effective_horizon = min(results.effective_horizon, H_k)
        end

        # Run a Bellman backup.
        for timestep in ProgressBar(1:horizon-1)
            Threads.@threads for state in vi.visitable_states[timestep]
                for action = 1:num_actions
                    next_state = transitions[state, action] + 1
                    max_next_q = -Inf64
                    max_next_var_bound = 0
                    for action = 1:num_actions
                        next_q = current_qs[timestep+1, next_state, action]
                        max_next_q = max(max_next_q, next_q)
                        next_var_bound = var_bounds[timestep+1, next_state, action]
                        max_next_var_bound = max(max_next_var_bound, next_var_bound)
                    end
                    current_qs[timestep, state, action] =
                        rewards[state, action] + max_next_q
                    var_bounds[timestep, state, action] = max_next_var_bound
                end
            end
        end
        k += 1
    end

    results
end

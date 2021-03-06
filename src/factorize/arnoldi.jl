# arnoldi.jl
#
# Arnoldi iteration for constructing the orthonormal basis of a Krylov subspace.
struct ArnoldiIterator{F,T,O<:Orthogonalizer}
    operator::F
    v₀::T
    orth::O
end
ArnoldiIterator(A, v₀) = ArnoldiIterator(A, v₀, Defaults.orth)

Base.iteratorsize(::Type{<:ArnoldiIterator}) = Base.SizeUnknown()

mutable struct ArnoldiFact{T,S} <: KrylovFactorization{T}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k
    H::Vector{S} # stores the Hessenberg matrix in packed form
    r::T # residual
end

Base.length(F::ArnoldiFact) = F.k
Base.sizehint!(F::ArnoldiFact, n) = begin
    sizehint!(F.H, (n*n + 3*n) >> 1)
    return F
end
Base.eltype(F::ArnoldiFact) = eltype(typeof(F))
Base.eltype(::Type{<:ArnoldiFact{<:Any,S}}) where {S} = S

basis(F::ArnoldiFact) = F.V
rayleighquotient(F::ArnoldiFact) = PackedHessenberg(F.H, F.k)
normres(F::ArnoldiFact) = abs(F.H[end])
residual(F::ArnoldiFact) = F.r

function Base.start(iter::ArnoldiIterator)
    β₀ = vecnorm(iter.v₀)
    T = typeof(one(eltype(iter.v₀))/β₀) # division might change eltype
    v₀ = scale!(similar(iter.v₀, T), iter.v₀, 1/β₀)
    w = apply(iter.operator, v₀) # applying the operator might change eltype
    v = copy!(similar(w), v₀)
    r, α = orthogonalize!(w, v, iter.orth)
    β = vecnorm(r)
    V = OrthonormalBasis([v])
    H = [α, β]
    return ArnoldiFact(1, V, H, r)
end
function start!(iter::ArnoldiIterator, state::ArnoldiFact) # recylcle existing state
    v₀ = iter.v₀
    V = state.V
    while length(V) > 1
        pop!(V)
    end
    H = empty!(state.H)

    v = scale!(V[1], v₀, 1/vecnorm(v₀))
    w = apply(iter.operator, v)
    r, α = orthogonalize!(w, v, iter.orth)
    β = vecnorm(r)
    state.k = 1
    push!(H, α, β)
    state.r = r
    return state
end

Base.done(iter::ArnoldiIterator, state::ArnoldiFact) = normres(state) < eps(real(eltype(state)))

function Base.next(iter::ArnoldiIterator, state::ArnoldiFact)
    value = (basis(state), rayleighquotient(state), residual(state))
    state = next!(iter, deepcopy(state))
    return value, state
end
function next!(iter::ArnoldiIterator, state::ArnoldiFact)
    state.k += 1
    k = state.k
    V = state.V
    H = state.H
    r = state.r
    β = normres(state)
    push!(V, scale!(r, r,  1/β))
    m = length(H)
    resize!(H, m+k+1)
    r, β = arnoldirecurrence!(iter.operator, V, view(H, (m+1):(m+k)), iter.orth)
    H[m+k+1] = β
    state.r = r
    return state
end

function shrink!(state::ArnoldiFact, k)
    length(state) <= k && return state
    V = state.V
    H = state.H
    while length(V) > k+1
        pop!(V)
    end
    r = pop!(V)
    resize!(H, (k*k + 3*k) >> 1)
    state.k = k
    state.r = scale!(r, r, normres(state))
    return state
end

# Arnoldi recurrence: simply use provided orthonormalization routines
function arnoldirecurrence!(operator, V::OrthonormalBasis, h::AbstractVector, orth::Orthogonalizer)
    w = apply(operator, last(V))
    r, h = orthogonalize!(w, V, h, orth)
    return r, vecnorm(r)
end

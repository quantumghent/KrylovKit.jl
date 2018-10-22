struct RecursiveVec{T<:Tuple}
    vecs::T
end
RecursiveVec(arg1, args...) = RecursiveVec((arg1, args...))

Base.getindex(v::RecursiveVec, i) = v.vecs[i]
Base.start(v::RecursiveVec) = start(v.vecs)
Base.next(v::RecursiveVec, s) = next(v.vecs, s)
Base.done(v::RecursiveVec, s) = done(v.vecs, s)

Base.eltype(v::RecursiveVec) = eltype(typeof(v))
Base.eltype(::Type{RecursiveVec{T}}) where {T<:Tuple} = _eltype(T)

_eltype(::Type{Tuple{T}}) where {T} = eltype(T)
function _eltype(::Type{TT}) where {TT<:Tuple}
    T = eltype(Base.tuple_type_head(TT))
    T2 = _eltype(Base.tuple_type_tail(TT))
    T == T2 ? T : error("all elements of a `RecursiveVec` should have same `eltype`")
end

Base.:-(v::RecursiveVec) = RecursiveVec(map(-,v.vecs))
Base.:+(v::RecursiveVec, w::RecursiveVec) = RecursiveVec(map(+,v.vecs, w.vecs))
Base.:-(v::RecursiveVec, w::RecursiveVec) = RecursiveVec(map(-,v.vecs, w.vecs))
Base.:*(v::RecursiveVec, a) = RecursiveVec(map(x->x*a, v.vecs))
Base.:*(a, v::RecursiveVec) = RecursiveVec(map(x->a*x, v.vecs))
Base.:/(v::RecursiveVec, a) = RecursiveVec(map(x->x/a, v.vecs))
Base.:\(a, v::RecursiveVec) = RecursiveVec(map(x->a\x, v.vecs))

function Base.similar(v::RecursiveVec, ::Type{T} = eltype(v)) where {T}
    RecursiveVec(map(x->similar(x,T), v.vecs))
end

function Base.fill!(v::RecursiveVec, a)
    for x in v.vecs
        fill!(x, a)
    end
    return v
end

function Base.copy!(w::RecursiveVec, v::RecursiveVec)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i = 1:length(w.vecs)
        copy!(w.vecs[i], v.vecs[i])
    end
    return w
end

function Base.scale!(w::RecursiveVec, a, v::RecursiveVec)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i = 1:length(w.vecs)
        scale!(w.vecs[i], a, v.vecs[i])
    end
    return w
end

function Base.scale!(w::RecursiveVec, v::RecursiveVec, a)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i = 1:length(w.vecs)
        scale!(w.vecs[i], v.vecs[i], a)
    end
    return w
end

function Base.scale!(v::RecursiveVec, a)
    for x in v.vecs
        scale!(x, a)
    end
    return v
end

function LinAlg.axpy!(a, v::RecursiveVec, w::RecursiveVec)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i = 1:length(w.vecs)
        LinAlg.axpy!(a, v.vecs[i], w.vecs[i])
    end
    return w
end

LinAlg.vecdot(v::RecursiveVec{T}, w::RecursiveVec{T}) where {T} = sum(x->vecdot(x...), zip(v.vecs, w.vecs))
LinAlg.vecnorm(v::RecursiveVec, p::Real = 2) = vecnorm(map(x->vecnorm(x,p), v.vecs), p)

Base.zero(v::RecursiveVec) = RecursiveVec(zero.(v.vecs))

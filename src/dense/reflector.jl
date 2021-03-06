# Elementary Householder reflection
struct Householder{T,V<:AbstractVector,R<:IndexRange}
    β::T
    v::V
    r::R
end

function householder(x::AbstractVector, r::IndexRange = indices(x,1), k = first(r))
    i = findfirst(equalto(k), r)
    i == 0 && error("k = $k should be in the range r = $r")
    β, v, ν = _householder!(x[r], i)
    return Householder(β,v,r), ν
end
# Householder reflector h that zeros the elements A[r,col] (except for A[k,col]) upon lmul!(A,h)
function householder(A::AbstractMatrix, r::IndexRange, col::Int, k = first(r))
    i = findfirst(equalto(k), r)
    i == 0 && error("k = $k should be in the range r = $r")
    β, v, ν = _householder!(A[r,col], i)
    return Householder(β,v,r), ν
end
# Householder reflector that zeros the elements A[row,r] (except for A[row,k]) upon rmulc!(A,h)
function householder(A::AbstractMatrix, row::Int, r::IndexRange, k = first(r))
    i = findfirst(equalto(k), r)
    i == 0 && error("k = $k should be in the range r = $r")
    β, v, ν = _householder!(conj!(A[row,r]), i)
    return Householder(β,v,r), ν
end

# generate Householder vector based on vector v, such that applying the reflection
# to v yields a vector with single non-zero element on position i, whose value is
# positive and thus equal to norm(v)
function _householder!(v::AbstractVector{T}, i::Int) where {T}
    β::T = zero(T)
    @inbounds begin
        σ = abs2(zero(T))
        @simd for k=1:i-1
            σ += abs2(v[k])
        end
        @simd for k=i+1:length(v)
            σ += abs2(v[k])
        end
        vi = v[i]
        ν = sqrt(abs2(vi)+σ)

        if σ == 0 && vi == ν
            β = zero(vi)
        else
            if real(vi) < 0
                vi = vi - ν
            else
                vi = ((vi-conj(vi))*ν - σ)/(conj(vi)+ν)
            end
            @simd for k=1:i-1
                v[k] /= vi
            end
            v[i] = 1
            @simd for k=i+1:length(v)
                v[k] /= vi
            end
            β = -conj(vi)/(ν)
        end
    end
    return β, v, ν
end

function lmul!(x::AbstractVector, H::Householder)
    v = H.v
    r = H.r
    β = H.β
    β == 0 && return x
    @inbounds begin
        μ::eltype(x) = 0
        i = 1
        @simd for j in r
            μ += conj(v[i])*x[j]
            i += 1
        end
        μ *= β
        i = 1
        @simd for j in H.r
            x[j] -= μ*v[i]
            i += 1
        end
    end
    return x
end
function lmul!(A::AbstractMatrix, H::Householder, cols=indices(A,2))
    v = H.v
    r = H.r
    β = H.β
    β == 0 && return A
    @inbounds begin
        for k in cols
            μ::eltype(A) = 0
            i = 1
            @simd for j in r
                μ += conj(v[i])*A[j,k]
                i += 1
            end
            μ *= β
            i = 1
            @simd for j in H.r
                A[j,k] -= μ*v[i]
                i += 1
            end
        end
    end
    return A
end
function rmulc!(A::AbstractMatrix, H::Householder, rows=indices(A,1))
    v = H.v
    r = H.r
    β = H.β
    β == 0 && return A
    w = similar(A, length(rows))
    fill!(w, 0)
    @inbounds begin
        l = 1
        for k in r
            j = 1
            @simd for i in rows
                w[j] += A[i,k]*v[l]
                j += 1
            end
            l += 1
        end
        l = 1
        for k in r
            j = 1
            @simd for i in rows
                A[i,k] -= conj(β)*w[j]*conj(v[l])
                j += 1
            end
            l += 1
        end
    end
    return A
end
function rmulc!(b::OrthonormalBasis, H::Householder)
    v = H.v
    r = H.r
    β = H.β
    β == 0 && return b
    w = fill!(similar(b[first(r)]), zero(eltype(b[first(r)])))
    @inbounds begin
        l = 1
        for k in r
            axpy!(v[l], b[k], w)
            l += 1
        end
        l = 1
        for k in r
            axpy!(-conj(v[l])*conj(β), w, b[k])
            l += 1
        end
    end
    return b
end

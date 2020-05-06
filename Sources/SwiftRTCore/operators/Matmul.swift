//******************************************************************************
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Foundation

//==============================================================================
/// matmul
/// performs a matrix cross product
/// - Parameters:
///  - lhs: left hand tensor
///  - transposeLhs: `true` to transpose `lhs`, default is `false`
///  - rhs: right hand tensor.
///  - transposeRhs: `true` to transpose `rhs`, default is `false`
/// - Returns: a new tensor containing the result
// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
@differentiable(where E: DifferentiableElement)
@inlinable public func matmul<E>(
    _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> TensorR2<E> where E: Numeric
{
    let lhsShape = transposeLhs ? lhs.shape.t : lhs.shape
    let rhsShape = transposeRhs ? rhs.shape.t : rhs.shape
    assert(lhsShape[1] == rhsShape[0], "matmul inner dimensions must be equal")
    var result = TensorR2<E>(Shape2(lhsShape[0], rhsShape[1]))
    Context.currentQueue.matmul(lhs, transposeLhs, rhs, transposeRhs, &result)
    return result
}

@derivative(of: matmul)
@inlinable public func _vjpMatmul<E>(
    _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>, TensorR2<E>))
where E: DifferentiableElement
{
    let value = matmul(lhs, transposed: transposeLhs,
                       rhs, transposed: transposeRhs)
    return (value, {
        let (lhsGrad, rhsGrad): (TensorR2<E>, TensorR2<E>)
        switch (transposeLhs, transposeRhs) {
        case (false, false):
            lhsGrad = matmul($0, transposed: false, rhs, transposed: true)
            rhsGrad = matmul(lhs, transposed: true, $0, transposed: false)
        case (false, true):
            lhsGrad = matmul($0, rhs)
            rhsGrad = matmul(lhs, transposed: true, $0, transposed: false)
        case (true, false):
            lhsGrad = matmul($0, transposed: false, rhs, transposed: true)
            rhsGrad = matmul(lhs, $0)
        case (true, true):
            lhsGrad = matmul($0, transposed: true, rhs, transposed: true)
            rhsGrad = matmul(lhs, transposed: true, $0, transposed: true)
        }
        return (lhsGrad, rhsGrad)
    })
}

//==============================================================================
/// matmul
/// performs a batched matrix cross product
/// - Parameters:
///  - lhs: left hand batched tensor
///  - transposeLhs: `true` to transpose `lhs`, default is `false`
///  - rhs: right hand 2D tensor
///  - transposeRhs: `true` to transpose `rhs`, default is `false`
/// - Returns: a new tensor containing the result
// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
@differentiable(where E: DifferentiableElement)
@inlinable public func matmul<S,E>(
    _ lhs: Tensor<S,E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> Tensor<S,E> where S: TensorShape, E: Numeric
{
    fatalError()
}

@derivative(of: matmul)
@inlinable public func _vjpMatmul<S,E>(
    _ lhs: Tensor<S,E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, TensorR2<E>))
where S: TensorShape, E: DifferentiableElement
{
    fatalError()
}

//==============================================================================
/// matmul
/// performs a batched matrix cross product
/// - Parameters:
///  - lhs: left hand batched tensor
///  - transposeLhs: `true` to transpose `lhs`, default is `false`
///  - rhs: right hand 2D tensor
///  - transposeRhs: `true` to transpose `rhs`, default is `false`
/// - Returns: a new tensor containing the result
// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
@differentiable(where E: DifferentiableElement)
@inlinable public func matmul<S,E>(
    _ lhs: TensorR2<E>, transposed transposeRhs: Bool = false,
    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
) -> Tensor<S,E> where S: TensorShape, E: Numeric
{
    fatalError()
}

@derivative(of: matmul)
@inlinable public func _vjpMatmul<S,E>(
    _ lhs: TensorR2<E>, transposed transposeRhs: Bool = false,
    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (TensorR2<E>, Tensor<S,E>))
where S: TensorShape, E: DifferentiableElement
{
    fatalError()
}

//==============================================================================
/// matmul
/// performs a batched matrix cross product
/// - Parameters:
///  - lhs: left hand batched tensor
///  - transposeLhs: `true` to transpose `lhs`, default is `false`
///  - rhs: right hand batched tensor
///  - transposeRhs: `true` to transpose `rhs`, default is `false`
/// - Returns: a new tensor containing the result
// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
@differentiable(where E: DifferentiableElement)
@inlinable public func matmul<S,E>(
    _ lhs: Tensor<S,E>, transposed transposeRhs: Bool = false,
    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
) -> Tensor<S,E> where S: TensorShape, E: Numeric
{
    fatalError()
}

@derivative(of: matmul)
@inlinable public func _vjpMatmul<S,E>(
    _ lhs: Tensor<S,E>, transposed transposeRhs: Bool = false,
    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
where S: TensorShape, E: DifferentiableElement
{
    fatalError()
}
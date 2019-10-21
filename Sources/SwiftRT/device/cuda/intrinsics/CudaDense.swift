//******************************************************************************
// Copyright 2019 Google LLC
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
import CCuda

public final class CudaDense<T> where
    T: TensorView, T.Element: AnyFloatingPoint
{
    // properties
    private var zero: T.Element = 0
    private var one: T.Element = 1
    private let activation: ActivationMode
    private let biasTensorDescriptor: TensorDescriptor
    private let yTensorDescriptor: TensorDescriptor
    private let weight: T
    private let bias: T
    private var xDiff: T!
    private var y: T

    //--------------------------------------------------------------------------
    // initializer
    public init(x: T, weight: T, bias: T, activation: ActivationMode) throws {
        assert(x.rank == 2 && weight.rank == 2 && bias.rank == 1)
        self.weight = weight
        self.bias = bias
        self.activation = activation
        
        // setup bias
        biasTensorDescriptor = try bias.createTensorDescriptor()
        
        // create output
        y = x.createDense(with: [x.extents[0], weight.extents[1]])
        yTensorDescriptor = try y.createTensorDescriptor()
    }

    //--------------------------------------------------------------------------
    // inferring
    public func inferring(from x: T) throws -> T {
        let deviceQueue = _Queues.current as! CudaQueue
        
        // TODO: is there a better fused kernel for y = wx + b??
        try deviceQueue.gemm(
            transA: .noTranspose, matrixA: x,
            transB: .noTranspose, matrixB: weight,
            matrixC: &y)
        
        try cudaCheck(status: cudnnAddTensor(
            deviceQueue.cudnn.handle,
            // alpha
            &one,
            // bias
            biasTensorDescriptor.desc,
            bias.deviceReadOnly(using: deviceQueue),
            // beta
            &one,
            // y
            yTensorDescriptor.desc,
            y.deviceReadWrite(using: deviceQueue)))

        return y
    }

    //--------------------------------------------------------------------------
    // gradient
    public func gradient(yDiff: T, x: T) throws -> T {
        // lazy create and retain
        if xDiff == nil { xDiff = x.createDense() }
        let deviceQueue = _Queues.current as! CudaQueue

        // TODO: hmm... should bias be part of the calculation??
        try deviceQueue.gemm(transA: .noTranspose, matrixA: yDiff,
                             transB: .transpose, matrixB: weight,
                             matrixC: &xDiff)
        return xDiff
    }
}

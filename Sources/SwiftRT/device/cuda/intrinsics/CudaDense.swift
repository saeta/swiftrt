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
    private var output: T
    private let weight: T
    private let bias: T
    private let activation: ActivationMode
    private let biasTensorDescriptor: TensorDescriptor
    private let outputTensorDescriptor: TensorDescriptor
//    private var backReductionContext: ReductionContext!
    private var inputGradient: T!
    private var zero: T.Element = 0
    private var one: T.Element = 1

    //--------------------------------------------------------------------------
    // initializer
    public init(input: T,
                weight: T,
                bias: T,
                activation: ActivationMode,
                evaluationMode: EvaluationMode) throws
    {
        assert(input.rank == 2 && weight.rank == 2 && bias.rank == 1)
        self.weight = weight
        self.bias = bias
        self.activation = activation
        
        // create inputGradientTensor if training will be done
        if evaluationMode == .training {
            inputGradient = input.createDense()
        }
        
        // setup bias
        biasTensorDescriptor = try bias.createTensorDescriptor()
        
        // create output
        output = input.createDense(with: [input.extents[0], weight.extents[1]])
        outputTensorDescriptor = try output.createTensorDescriptor()
    }

    //--------------------------------------------------------------------------
    // inferring
    public func inferring(from input: T) throws -> T {
        let deviceQueue = _Queues.current as! CudaQueue
        
        try deviceQueue.gemm(
            transA: .noTranspose, matrixA: input,
            transB: .noTranspose, matrixB: weight,
            matrixC: &output)
        
        try cudaCheck(status: cudnnAddTensor(
            deviceQueue.cudnn.handle,
            &one,
            biasTensorDescriptor.desc,
            bias.deviceReadOnly(using: deviceQueue),
            &one,
            outputTensorDescriptor.desc,
            output.deviceReadWrite(using: deviceQueue)))

        return output
    }

    //--------------------------------------------------------------------------
    // gradient
    // TODO: gradients need to be sorted out w.r.t. S4TF handling
    public func gradient(outputGradient: T, input: T) throws -> T {
//        let deviceQueue = _Queues.current as! CudaQueue
        

        return inputGradient
    }
}

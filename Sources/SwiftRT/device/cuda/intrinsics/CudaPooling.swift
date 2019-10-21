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

public final class CudaPooling<T> where
    T: TensorView, T.Element: AnyFloatingPoint
{
    // properties
    private let poolingDescriptor: PoolingDescriptor
    private let inputTensorDescriptor: TensorDescriptor
    private let outputTensorDescriptor: TensorDescriptor
    private var output: T
    private var inputGradient: T!
    private var zero: T.Element = 0
    private var one: T.Element = 1

    //--------------------------------------------------------------------------
    // initializer
    public init(input: T,
                filterSize: [Int],
                strides: [Int],
                padding: [Int],
                poolingMode: PoolingMode,
                nan: NanPropagation,
                evaluationMode: EvaluationMode) throws
    {
        // create the descriptor
        poolingDescriptor = try PoolingDescriptor(
            mode: poolingMode,
            nan: nan,
            filterSize: filterSize,
            padding: padding,
            strides: strides)

        // create input tensor descriptor
        inputTensorDescriptor = try input.createTensorDescriptor()
        
        // create inputGradientTensor if training will be done
        if evaluationMode == .training {
            inputGradient = input.createDense()
        }
        
        // get output tensor size
        var extents = [Int32](repeating: 0, count: input.rank)
        try cudaCheck(status: cudnnGetPoolingNdForwardOutputDim(
            poolingDescriptor.desc,
            inputTensorDescriptor.desc,
            Int32(input.rank),
            &extents))

        // create retained output tensor
        output = input.createDense(with: extents.map { Int($0) })
        outputTensorDescriptor = try output.createTensorDescriptor()
    }
    
    //--------------------------------------------------------------------------
    // inferring
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnPoolingForward
    public func inferring(from input: T) throws -> T {
        let deviceQueue = _Queues.current as! CudaQueue
        
        try cudaCheck(status: cudnnPoolingForward(
            deviceQueue.cudnn.handle,
            poolingDescriptor.desc,
            &one,
            inputTensorDescriptor.desc,
            input.deviceReadOnly(using: deviceQueue),
            &zero,
            outputTensorDescriptor.desc,
            output.deviceReadWrite(using: deviceQueue)))

        return output
    }
    
    //--------------------------------------------------------------------------
    // gradient
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnPoolingBackward
    public func gradient(outputGradient: T, input: T) throws -> T {
        let deviceQueue = _Queues.current as! CudaQueue
        
        try cudaCheck(status: cudnnPoolingBackward(
            deviceQueue.cudnn.handle,
            poolingDescriptor.desc,
            &one,
            outputTensorDescriptor.desc,
            output.deviceReadOnly(using: deviceQueue),
            outputTensorDescriptor.desc,
            outputGradient.deviceReadOnly(using: deviceQueue),
            inputTensorDescriptor.desc,
            input.deviceReadOnly(using: deviceQueue),
            &zero,
            inputTensorDescriptor.desc,
            inputGradient.deviceReadWrite(using: deviceQueue)))
        
        return inputGradient
    }
}

//==============================================================================
// PoolingMode
extension cudnnPoolingMode_t : Hashable {}

extension PoolingMode {
    public var cudnn: cudnnPoolingMode_t {
        get {
            let modes: [PoolingMode: cudnnPoolingMode_t] = [
                .max: CUDNN_POOLING_MAX,
                .maxDeterministic: CUDNN_POOLING_MAX_DETERMINISTIC,
                .averageExcludePadding: CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                .averageIncludePadding: CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            ]
            return modes[self]!
        }
    }
}

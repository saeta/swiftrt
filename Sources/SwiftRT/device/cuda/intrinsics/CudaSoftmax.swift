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

// *** TODO design questions!
// 1) class or struct, how are things retained, reused, etc?
// 2) should the input be retained to guarentee that init
//    matches the same shape as inferring? Or just assert in inferring?

public final class CudaSoftmax<T> where
    T: TensorView, T.Element: AnyFloatingPoint
{
    // properties
    private var zero: T.Element = 0
    private var one: T.Element = 1
    private var cudnnAlgorithm: cudnnSoftmaxAlgorithm_t
    private var cudnnMode: cudnnSoftmaxMode_t
    private let xTensorDescriptor: TensorDescriptor
    private let yTensorDescriptor: TensorDescriptor
    private var xDiff: T!
    private var y: T

    //--------------------------------------------------------------------------
    // initializer
    public init(x: T, algorithm: SoftmaxAlgorithm, mode: SoftmaxMode) throws {
        cudnnAlgorithm = algorithm.cudnn
        cudnnMode = mode.cudnn
        
        // create input tensor descriptor
        xTensorDescriptor = try x.createTensorDescriptor()

        // create retained y tensor the queried output size
        // based on configuration
        y = x.createDense()
        yTensorDescriptor = try y.createTensorDescriptor()
    }
    
    //--------------------------------------------------------------------------
    // inferring
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSoftmaxForward
    public func inferring(from x: T) throws -> T {
        let deviceQueue = _Queues.current as! CudaQueue
        
        try cudaCheck(status: cudnnSoftmaxForward(
            deviceQueue.cudnn.handle,
            cudnnAlgorithm,
            cudnnMode,
            // alpha
            &one,
            // x
            xTensorDescriptor.desc,
            x.deviceReadOnly(using: deviceQueue),
            // beta
            &zero,
            // y
            yTensorDescriptor.desc,
            y.deviceReadWrite(using: deviceQueue)))

        return y
    }
    
    //--------------------------------------------------------------------------
    // gradient
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSoftmaxBackward
    public func gradient(yDiff: T, x: T) throws -> T {
        // lazy create and retain
        if xDiff == nil { xDiff = x.createDense() }
        let deviceQueue = _Queues.current as! CudaQueue
        
        // if there aren't any labels then do a normal backward
        try cudaCheck(status: cudnnSoftmaxBackward(
            deviceQueue.cudnn.handle,
            cudnnAlgorithm,
            cudnnMode,
            // alpha
            &one,
            // y
            yTensorDescriptor.desc,
            y.deviceReadOnly(using: deviceQueue),
            // dy
            yTensorDescriptor.desc,
            yDiff.deviceReadOnly(using: deviceQueue),
            // beta
            &zero,
            // dx
            xTensorDescriptor.desc,
            xDiff.deviceReadWrite(using: deviceQueue)))

        return xDiff
    }
}

//==============================================================================
// Enum --> Cuda value mapping
//
extension SoftmaxAlgorithm {
    public var cudnn: cudnnSoftmaxAlgorithm_t {
        get {
            let algorithms: [SoftmaxAlgorithm: cudnnSoftmaxAlgorithm_t] = [
                .accurate: CUDNN_SOFTMAX_ACCURATE,
                .fast: CUDNN_SOFTMAX_FAST,
                .log: CUDNN_SOFTMAX_LOG,
            ]
            return algorithms[self]!
        }
    }
}

extension SoftmaxMode {
    public var cudnn: cudnnSoftmaxMode_t {
        get {
            switch self {
            case .channel : return CUDNN_SOFTMAX_MODE_CHANNEL
            case .instance: return CUDNN_SOFTMAX_MODE_INSTANCE
            }
        }
    }
}

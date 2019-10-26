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

public struct CudaActivation<T> where
    T: TensorView, T.Element: AnyFloatingPoint
{
    // properties
    private let activationDescriptor: ActivationDescriptor
    private let tensorDescriptor: TensorDescriptor

    //--------------------------------------------------------------------------
    // initializer
    public init(x: T,
                yShape: inout DataShape,
                mode: ActivationMode,
                nan: NanPropagation,
                reluCeiling: Double = 0) throws
    {
        // create descriptor
        activationDescriptor =
            try ActivationDescriptor(mode: mode, nan: nan,
                                     reluCeiling: reluCeiling)
        
        // TODO: figure out how S4TF wants to handle layouts
        // create tensor descriptors
//        let tensorShape = inData.layout != .matrix ? inData.shape :
//            Shape(extent: [inData.rows, inData.cols, 1, 1], layout: .nchw)

        tensorDescriptor = try x.createTensorDescriptor()
        
        // return the shape of the output y and create a tensorDescriptor
        // with the same scalarType for y
        yShape = x.shape
    }
    
    //--------------------------------------------------------------------------
    // inferring
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationForward
    public func inferring(y: inout T, from x: T) throws {
        let deviceQueue = DeviceContext.currentQueue as! CudaQueue
        
        try cudaCheck(status: cudnnActivationForward(
            deviceQueue.cudnn.handle,
            activationDescriptor.desc,
            // alpha
            T.Element.onePointer,
            // x
            tensorDescriptor.desc,
            x.deviceReadOnly(using: deviceQueue),
            // beta
            T.Element.zeroPointer,
            // y
            tensorDescriptor.desc,
            y.deviceReadWrite(using: deviceQueue)))
    }
    
    //--------------------------------------------------------------------------
    // gradient
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationBackward
    public func gradient(y: T, yDiff: T, x: T, xDiff: inout T) throws {
        let deviceQueue = DeviceContext.currentQueue as! CudaQueue

        try cudaCheck(status: cudnnActivationBackward(
            deviceQueue.cudnn.handle,
            activationDescriptor.desc,
            // alpha
            T.Element.onePointer,
            // y
            tensorDescriptor.desc,
            y.deviceReadOnly(using: deviceQueue),
            // dy
            tensorDescriptor.desc,
            yDiff.deviceReadOnly(using: deviceQueue),
            // x
            tensorDescriptor.desc,
            x.deviceReadOnly(using: deviceQueue),
            // beta
            T.Element.zeroPointer,
            // dx
            tensorDescriptor.desc,
            xDiff.deviceReadWrite(using: deviceQueue)))
    }
}

//==============================================================================
extension ActivationMode {
    public var cudnn: cudnnActivationMode_t {
        get {
            let modes: [ActivationMode: cudnnActivationMode_t] = [
                .sigmoid: CUDNN_ACTIVATION_SIGMOID,
                .relu: CUDNN_ACTIVATION_RELU,
                .tanh: CUDNN_ACTIVATION_TANH,
                .clippedRelu: CUDNN_ACTIVATION_CLIPPED_RELU,
                .elu: CUDNN_ACTIVATION_ELU,
                .identity: CUDNN_ACTIVATION_IDENTITY,
            ]
            return modes[self]!
        }
    }
}


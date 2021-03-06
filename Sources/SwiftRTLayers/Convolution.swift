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
import Foundation
import Numerics
import SwiftRTCore

//==============================================================================
// convenience types
public typealias Convolution1 = Convolution<Shape1,Float,Float>
public typealias Convolution2 = Convolution<Shape2,Float,Float>
public typealias Convolution3 = Convolution<Shape3,Float,Float>

//==============================================================================
/// Convolution
public struct Convolution<Shape, Element, FilterElement>: Logging
where Shape: TensorShape,
      Element: ScalarElement,
      FilterElement: ScalarElement,
      FilterElement.Value: BinaryFloatingPoint
{
    // types
    public typealias Data = Tensor<Shape,Element>
    public typealias Filter = Tensor<Shape,FilterElement>
    public typealias Bias = TensorR1<FilterElement>
    public typealias Op = DeviceConvolution<Shape, Element, FilterElement>

    /// The convolution filter
    public var filter: Filter
    /// The bias vector
    public var bias: Bias
    /// The element-wise activation function type
    @noDerivative public let activation: ActivationType
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: Shape
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The dilation factor for spatial dimensions.
    @noDerivative public let dilations: Shape
    /// device specific convolution operator
    @noDerivative public let convolutionOp: Op

    //--------------------------------------------------------------------------
    /// Creates a `Convolution` layer with the specified filter, bias,
    /// activation function, stride, dilation and padding.
    ///
    /// - Parameters:
    ///   - filter: The convolution filter of shape
    ///     [filter width, input channel count, output channel count].
    ///   - bias: The bias vector of shape [output channel count].
    ///   - activation: The element-wise activation function.
    ///   - strides: The stride of the sliding window for the temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilation: The dilation factor for the temporal dimension.
    @inlinable public init(
        filter: Filter,
        bias: Bias? = nil,
        activation: ActivationType = .identity,
        strides: Shape = Shape.one,
        padding: Padding = .valid,
        dilations: Shape = Shape.one,
        properties: ConvolutionProperties = ConvolutionProperties())
    {
        self.filter = filter
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.dilations = dilations

        self.bias = bias ??
            TensorR1<FilterElement>(zeros: [filter.shape[1]],
                                    layout: filter.layout)

        // create the device op and save the output shape
        self.convolutionOp = Context.currentQueue.convolution(
            activation: activation,
            strides: strides,
            padding: padding,
            dilations: dilations,
            properties: properties,
            deviceId: Context.currentQueue.deviceIndex,
            filterBiasBackpropQueueIndex: 2)
    }
    
    //--------------------------------------------------------------------------
    ///
    @inlinable public func callAsFunction(_ input: Data) -> Data {
        convolutionOp.forward(x: input, filter: filter, bias: bias)
    }
}

extension Convolution where FilterElement.Value: Real & BinaryFloatingPoint {
    //--------------------------------------------------------------------------
    /// Creates a `Convolution` layer with the specified filter shape,
    /// stride, padding, dilation and element-wise activation function.
    ///
    /// - Parameters:
    ///   - filterShape: The 3-D shape of the filter, representing
    ///     (filter width, input channel count, output channel count).
    ///   - stride: The stride of the sliding window for the temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilation: The dilation factor for the temporal dimension.
    ///   - activation: The element-wise activation function.
    ///   - filterInitializer: Initializer to use for the filter parameters.
    ///   - biasInitializer: Initializer to use for the bias parameters.
    @inlinable public init(
        filterShape: Shape,
        stride: Int = 1,
        padding: Padding = .valid,
        dilation: Int = 1,
        activation: ActivationType = .identity,
        filterInitializer: ParameterInitializer<Shape,FilterElement> = glorotUniform(),
        biasInitializer: ParameterInitializer<Shape1,FilterElement> = zeros()
    ) {
        let biasShape = Shape1(filterShape[Shape.rank - 1])
        self.init(filter: filterInitializer(filterShape),
                  bias: biasInitializer(biasShape),
                  activation: activation,
                  strides: Shape(repeating: stride),
                  padding: padding,
                  dilations: Shape(repeating: dilation))
    }
}

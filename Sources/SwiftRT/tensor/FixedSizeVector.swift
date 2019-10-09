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

//==============================================================================
/// conformance indicates that scalar components are of the same type and
/// densely packed. This is necessary for zero copy view type casting of
/// short vector Element types.
/// For example: Matrix<RGBA<Float>> -> NHWCTensor<Float>
///
public protocol FixedSizeVector: Equatable {
    associatedtype Scalar
    static var count: Int { get }
}

public extension FixedSizeVector {
    static var count: Int {
        return MemoryLayout<Self>.size / MemoryLayout<Scalar>.size
    }
}

//==============================================================================
// RGB
public protocol RGBProtocol: FixedSizeVector, Codable {}

public struct RGB<Scalar>: RGBProtocol where Scalar: Numeric & Codable {
    public var r, g, b: Scalar

    @inlinable @inline(__always)
    public init() { r = Scalar.zero; g = Scalar.zero; b = Scalar.zero }

    @inlinable @inline(__always)
    public init(_ r: Scalar, _ g: Scalar, _ b: Scalar) {
        self.r = r; self.g = g; self.b = b
    }
}

//==============================================================================
// RGBA
public protocol RGBAProtocol: FixedSizeVector, Codable {}

public struct RGBA<Scalar> : RGBAProtocol where Scalar: Numeric & Codable {
    public var r, g, b, a: Scalar

    @inlinable @inline(__always)
    public init() {
        r = Scalar.zero; g = Scalar.zero; b = Scalar.zero; a = Scalar.zero
    }
    
    @inlinable @inline(__always)
    public init(_ r: Scalar, _ g: Scalar, _ b: Scalar, _ a: Scalar) {
        self.r = r; self.g = g; self.b = b; self.a = a
    }
}

//==============================================================================
// Stereo
public protocol StereoProtocol: FixedSizeVector, Codable {}

public struct Stereo<Scalar>: StereoProtocol where Scalar: Numeric & Codable {
    public var left, right: Scalar

    @inlinable @inline(__always)
    public init() { left = Scalar.zero; right = Scalar.zero }

    @inlinable @inline(__always)
    public init(_ left: Scalar, _ right: Scalar) {
        self.left = left; self.right = right
    }
}


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
// shaped positions and extents used for indexing and selection
public enum MatrixLayout { case rowMajor, columnMajor }
public typealias NDPosition = [Int]
public typealias ScalarPosition = Int
public typealias VectorPosition = Int
public typealias VectorExtents = Int
public typealias MatrixPosition = (r: Int, c: Int)
public typealias MatrixExtents = (rows: Int, cols: Int)
public typealias VolumePosition = (d: Int, r: Int, c: Int)
public typealias VolumeExtents = (depths: Int, rows: Int, cols: Int)
public typealias NCHWPosition = (i: Int, ch: Int, r: Int, c: Int)
public typealias NCHWExtents = (items: Int, channels: Int, rows: Int, cols: Int)
public typealias NHWCPosition = (i: Int, r: Int, c: Int, ch: Int)
public typealias NHWCExtents = (items: Int, rows: Int, cols: Int, channels: Int)

public extension TensorView {
    //--------------------------------------------------------------------------
    /// returns a collection of read only values
    func values(using stream: DeviceStream? = nil) throws
        -> TensorValueCollection<Self>
    {
        let buffer = try readOnly(using: stream)
        return try TensorValueCollection(view: self, buffer: buffer)
    }
    
    //--------------------------------------------------------------------------
    /// returns a collection of read write values
    mutating func mutableValues(using stream: DeviceStream? = nil) throws
        -> TensorMutableValueCollection<Self>
    {
        let buffer = try readWrite(using: stream)
        return try TensorMutableValueCollection(view: &self, buffer: buffer)
    }
}

//==============================================================================
// Codable extensions
extension ScalarValue: Codable where Element: Codable {}
extension Vector: Codable where Element: Codable {}
extension Matrix: Codable where Element: Codable {}
extension Volume: Codable where Element: Codable {}
extension NDTensor: Codable where Element: Codable {}
extension NHWCTensor: Codable where Element: Codable {}
extension NCHWTensor: Codable where Element: Codable {}

//==============================================================================
// ScalarView
public protocol ScalarView: TensorView {}

public extension ScalarView {
    //--------------------------------------------------------------------------
    var startIndex: ScalarIndex { return ScalarIndex(endOf: self) }
    var endIndex: ScalarIndex { return ScalarIndex(view: self, at: 0) }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolTensor(with extents: [Int]) -> ScalarValue<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray<Bool>(count: shape.elementCount,
                                      name: String(describing: Self.self))
        return ScalarValue<Bool>(shape: shape, tensorArray: array,
                                 viewOffset: 0,isShared: false)
    }

    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexTensor(with extents: [Int]) -> ScalarValue<IndexElement> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<IndexElement>(count: shape.elementCount,
                                              name: name)
        return ScalarValue<IndexElement>(shape: shape, tensorArray: array,
                                         viewOffset: 0, isShared: false)
    }
}

public extension ScalarView {
    //--------------------------------------------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: [element], name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
}

//------------------------------------------------------------------------------
// ScalarValue
public struct ScalarValue<Element>: ScalarView {
    // properties
    public let shape: DataShape
    public let isShared: Bool
    public var tensorArray: TensorArray<Element>
    public var viewOffset: Int
    
    public init(shape: DataShape,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
    }
}

extension ScalarValue: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VectorView
public protocol VectorView: TensorView { }

extension Vector: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VectorView extensions
public extension VectorView {
    //--------------------------------------------------------------------------
    var startIndex: VectorIndex { return VectorIndex(view: self, at: 0) }
    var endIndex: VectorIndex { return VectorIndex(endOf: self) }

    //-------------------------------------
    /// empty array
    init(count: Int, name: String? = nil) {
        let shape = DataShape(extents: [count])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(count: shape.elementCount, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeBufferPointer<Element>, name: String? = nil)
    {
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)

        // create shape considering column major
        let shape = DataShape(extents: [buffer.count])
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolTensor(with extents: [Int]) -> Vector<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray<Bool>(count: shape.elementCount,
                                      name: String(describing: Self.self))
        return Vector<Bool>(shape: shape, tensorArray: array,
                            viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexTensor(with extents: [Int]) -> Vector<IndexElement> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<IndexElement>(count: shape.elementCount,
                                              name: name)
        return Vector<IndexElement>(shape: shape, tensorArray: array,
                                    viewOffset: 0, isShared: false)
    }
}

//==============================================================================
public extension VectorView {
    //-------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: [element], name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }

    //-------------------------------------
    /// with convertible collection
    init<C>(name: String? = nil, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(name: name, elements: any.lazy.map { Element(any: $0) })
    }

    //-------------------------------------
    /// with an element collection
    init<C>(name: String? = nil, elements: C) where
        C: Collection, C.Element == Element
    {
        let shape = DataShape(extents: [elements.count])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: elements, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
}

//==============================================================================
// Vector
public struct Vector<Element>: VectorView {
    // properties
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray<Element>
    public var viewOffset: Int
    
    public init(shape: DataShape,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
    }
}

//==============================================================================
// MatrixView
public protocol MatrixView: TensorView {}

extension Matrix: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// MatrixView extensions
public extension MatrixView {
    var startIndex: MatrixIndex { return MatrixIndex(view: self, at: (0, 0)) }
    var endIndex: MatrixIndex { return MatrixIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: MatrixExtents, name: String? = nil) {
        let shape = DataShape(extents: [extents.rows, extents.cols])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(count: shape.elementCount, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: MatrixExtents, repeating other: Self) {
        let extents = [extents.rows, extents.cols]
        self.init(with: extents, repeating: other)
    }
    
    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: MatrixExtents,
         layout: MatrixLayout = .rowMajor,
         referenceTo buffer: UnsafeBufferPointer<Element>,
         name: String? = nil)
    {
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)

        // create shape considering column major
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolTensor(with extents: [Int]) -> Matrix<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray<Bool>(count: shape.elementCount,
                                      name: String(describing: Self.self))
        return Matrix<Bool>(shape: shape, tensorArray: array,
                            viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexTensor(with extents: [Int]) -> Matrix<IndexElement> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<IndexElement>(count: shape.elementCount,
                                              name: name)
        return Matrix<IndexElement>(shape: shape, tensorArray: array,
                                    viewOffset: 0, isShared: false)
    }

    //--------------------------------------------------------------------------
    // transpose
    var t: Self {
        return Self.init(shape: shape.transposed(),
                         tensorArray: tensorArray,
                         viewOffset: viewOffset,
                         isShared: isShared)
    }
}

//==============================================================================
// MatrixView data initialization extensions
public extension MatrixView {
    //-------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: [element], name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //-------------------------------------
    /// with convertible collection
    init<C>(_ extents: MatrixExtents, name: String? = nil,
            layout: MatrixLayout = .rowMajor, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(extents, name: name, layout: layout,
                  elements: any.lazy.map { Element(any: $0) })
    }
    
    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: MatrixExtents, name: String? = nil,
            layout: MatrixLayout = .rowMajor, elements: C) where
        C: Collection, C.Element == Element
    {
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: elements, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
}

//==============================================================================
// Matrix
public struct Matrix<Element>: MatrixView {
    // properties
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray<Element>
    public var viewOffset: Int
    
    public init(shape: DataShape,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
    }
}

//==============================================================================
// VolumeView
public protocol VolumeView: TensorView { }

extension Volume: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VolumeView extension
public extension VolumeView {
    var startIndex: VolumeIndex { return VolumeIndex(view: self, at: (0, 0, 0))}
    var endIndex: VolumeIndex { return VolumeIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: VolumeExtents, name: String? = nil) {
        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(count: shape.elementCount, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: VolumeExtents, repeating other: Self) {
        let extents = [extents.depths, extents.rows, extents.cols]
        self.init(with: extents, repeating: other)
    }

    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: VolumeExtents, name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)
        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolTensor(with extents: [Int]) -> Volume<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray<Bool>(count: shape.elementCount,
                                      name: String(describing: Self.self))
        return Volume<Bool>(shape: shape, tensorArray: array,
                            viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexTensor(with extents: [Int]) -> Volume<IndexElement> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<IndexElement>(count: shape.elementCount,
                                              name: name)
        return Volume<IndexElement>(shape: shape, tensorArray: array,
                                    viewOffset: 0, isShared: false)
    }
}

//==============================================================================
// VolumeView extension
public extension VolumeView {
    //-------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: [element], name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }

    //-------------------------------------
    /// with convertible collection
    init<C>(_ extents: VolumeExtents, name: String? = nil, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(extents, name: name,
                  elements: any.lazy.map { Element(any: $0) })
    }
    
    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: VolumeExtents, name: String? = nil, elements: C) where
        C: Collection, C.Element == Element
    {
        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: elements, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
}

//==============================================================================
/// Volume
public struct Volume<Element>: VolumeView {
    // properties
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray<Element>
    public var viewOffset: Int

    public init(shape: DataShape,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
    }
}

//==============================================================================
// NDTensorView
public protocol NDTensorView: TensorView { }

extension NDTensor: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// NDTensorView extensions
public extension NDTensorView {
    var startIndex: NDIndex { return NDIndex(view: self, at: [0]) }
    var endIndex: NDIndex { return NDIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: [Int], name: String? = nil) {
        let shape = DataShape(extents: extents)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(count: shape.elementCount, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(extents: [Int], name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>)
    {
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)

        // create shape considering column major
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolTensor(with extents: [Int]) -> NDTensor<Bool> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<Bool>(count: shape.elementCount, name: name)
        return NDTensor<Bool>(shape: shape, tensorArray: array,
                              viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexTensor(with extents: [Int]) -> NDTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<IndexElement>(count: shape.elementCount,
                                              name: name)
        return NDTensor<IndexElement>(shape: shape, tensorArray: array,
                                      viewOffset: 0, isShared: false)
    }
}

//==============================================================================
// NDTensorView extensions
public extension NDTensorView {
    //-------------------------------------
    /// with convertible collection
    init<C>(_ extents: [Int], name: String? = nil, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(extents, name: name,
                  elements: any.lazy.map { Element(any: $0) })
    }
    
    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: [Int], name: String? = nil, elements: C) where
        C: Collection, C.Element == Element
    {
        let shape = DataShape(extents: extents)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: elements, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
}

//------------------------------------------------------------------------------
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Element>: NDTensorView {
    // properties
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray<Element>
    public var viewOffset: Int

    public init(shape: DataShape,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
    }
}

//==============================================================================
/// NCHWTensorView
/// An NCHW tensor is a standard layout for use with cuDNN.
/// It has a layout of numerics organized as:
/// n: items
/// c: channels
/// h: rows
/// w: cols
public protocol NCHWTensorView: TensorView { }

extension NCHWTensor: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
/// NCHWTensorView extensions
public extension NCHWTensorView {
    var startIndex: NDIndex { return NDIndex(view: self, at: [0]) }
    var endIndex: NDIndex { return NDIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: NCHWExtents, name: String? = nil) {
        let extent = [extents.items, extents.channels,
                      extents.rows, extents.cols]
        let shape = DataShape(extents: extent)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(count: shape.elementCount, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: NCHWExtents, repeating other: Self) {
        let extent = [extents.items, extents.channels,
                      extents.rows, extents.cols]
        self.init(with: extent, repeating: other)
    }

    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: NCHWExtents, name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)

        let extents = [extents.items, extents.channels,
                       extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolTensor(with extents: [Int]) -> NCHWTensor<Bool> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<Bool>(count: shape.elementCount, name: name)
        return NCHWTensor<Bool>(shape: shape, tensorArray: array,
                                viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexTensor(with extents: [Int]) -> NCHWTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<IndexElement>(count: shape.elementCount,
                                              name: name)
        return NCHWTensor<IndexElement>(shape: shape, tensorArray: array,
                                        viewOffset: 0, isShared: false)
    }
}

//==============================================================================
/// NCHWTensorView extensions for data initialization
public extension NCHWTensorView {
    //-------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1, 1])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: [element], name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //-------------------------------------
    /// with convertible collection
    init<C>(_ extents: NCHWExtents, name: String? = nil, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(extents, name: name,
                  elements: any.lazy.map { Element(any: $0) })
    }
    
    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: NCHWExtents, name: String? = nil, elements: C) where
        C: Collection, C.Element == Element
    {
        let extents = [extents.items, extents.channels,
                       extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: elements, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
}

//==============================================================================
// NCHWTensor
public struct NCHWTensor<Element>: NCHWTensorView {
    // properties
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray<Element>
    public var viewOffset: Int

    public init(shape: DataShape,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
    }
}

//==============================================================================
/// NHWCTensorView
/// An NHWC tensor is a standard layout for use with cuDNN.
/// It has a layout of numerics organized as:
/// n: items
/// h: rows
/// w: cols
/// c: channels
public protocol NHWCTensorView: TensorView { }

extension NHWCTensor: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
/// NHWCTensorView extensions
public extension NHWCTensorView {
    var startIndex: NDIndex { return NDIndex(view: self, at: [0]) }
    var endIndex: NDIndex { return NDIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: NHWCExtents, name: String? = nil) {
        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        let shape = DataShape(extents: extents)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(count: shape.elementCount, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: NHWCExtents, repeating other: Self) {
        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        self.init(with: extents, repeating: other)
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: NHWCExtents, name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)

        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")

        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolTensor(with extents: [Int]) -> NHWCTensor<Bool> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<Bool>(count: shape.elementCount, name: name)
        return NHWCTensor<Bool>(shape: shape, tensorArray: array,
                                viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexTensor(with extents: [Int]) -> NHWCTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<IndexElement>(count: shape.elementCount,
                                              name: name)
        return NHWCTensor<IndexElement>(shape: shape, tensorArray: array,
                                        viewOffset: 0, isShared: false)
    }
}

//==============================================================================
/// NHWCTensorView extensions for data initialization
public extension NHWCTensorView {
    //-------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1, 1])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: [element], name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //-------------------------------------
    /// with convertible collection
    init<C>(_ extents: NHWCExtents, name: String? = nil, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(extents, name: name,
                  elements: any.lazy.map { Element(any: $0) })
    }
    
    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: NHWCExtents, name: String? = nil, elements: C) where
        C: Collection, C.Element == Element
    {
        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        let shape = DataShape(extents: extents)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: elements, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
}

//==============================================================================
/// NHWCTensor
public struct NHWCTensor<Element>: NHWCTensorView {
    // properties
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray<Element>
    public var viewOffset: Int

    public init(shape: DataShape,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
    }
}

//==============================================================================
/// NHWCTensor cast
public extension NHWCTensor {
    /// zero copy cast of a matrix of dense uniform values to NHWC
    init<T>(vector: T, name: String? = nil) where
        T: VectorView, T.Element: FixedSizeVector, T.Element.Scalar == Element
    {
        let viewExtents = [1, 1, vector.shape.extents[0], T.Element.count]
        let array = TensorArray<Element>(vector.tensorArray)
        self.init(shape: DataShape(extents: viewExtents),
                  tensorArray: array,
                  viewOffset: vector.viewOffset,
                  isShared: vector.isShared)
    }

    /// zero copy cast of a matrix of dense uniform values to NHWC
    init<T>(_ matrix: T, name: String? = nil) where
        T: MatrixView, T.Element: FixedSizeVector, T.Element.Scalar == Element
    {
        let viewExtents = [1,
                           matrix.shape.extents[0],
                           matrix.shape.extents[1],
                           T.Element.count]
        let viewStrides = [matrix.shape.elementSpanCount * T.Element.count,
                           matrix.shape.strides[0] * T.Element.count,
                           matrix.shape.strides[1] * T.Element.count,
                           1]
        let array = TensorArray<Element>(matrix.tensorArray)
        self.init(shape: DataShape(extents: viewExtents, strides: viewStrides),
                  tensorArray: array,
                  viewOffset: matrix.viewOffset,
                  isShared: matrix.isShared)
    }
}

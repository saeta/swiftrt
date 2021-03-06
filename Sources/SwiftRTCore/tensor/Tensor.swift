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
import Numerics


public let defaultElementName = "Element"
public let defaultTensorName = "Tensor"
public let defaultReferenceTensorName = "Reference"

//==============================================================================
/// Tensor
public struct Tensor<Shape, TensorElement>:
    TensorProtocol,
    MutableCollection,
    CustomStringConvertible,
    Logging
where Shape: TensorShape, TensorElement: StorageElement
{
    // types
    public typealias Index = ElementIndex<Shape>
    public typealias Element = TensorElement.Value
    
    // properties
    /// the number of element
    public let count: Int
    /// `true` if the view will be shared by by multiple writers
    public var isShared: Bool
    /// layout order of the elements in storage
    @noDerivative public let layout: Layout
    /// a collection that maps logical coordinates to storage elements
    /// via the current storage layout
    public var logicalElements: LogicalElements<Shape, TensorElement>
    /// the strides to traverse `shape` in logical coordinates
    public let logicalStrides: Shape
    /// the dimensions of the element space
    @noDerivative public let shape: Shape
    /// the element storage buffer.
    public var storage: StorageBufferType
    /// the logical storage buffer base index where this tensor's elements begin
    public let storageBase: Int
    /// The distance to the next element along each dimension
    public let strides: Shape
    /// the number of storage elements spanned by this tensor
    public let stridedSpanCount: Int

    //--------------------------------------------------------------------------
    // functional properties
    /// the unique storage id
    @inlinable public var id: Int { storage.id }
    /// the name of the collection
    @inlinable public var name: String {
        get { storage.name }
        set { storage.name = newValue }
    }
    /// `true` if the tensor elements are densely packed
    @inlinable public var isContiguous: Bool { stridedSpanCount == count }
    
    /// `true` if the tensor contains a single stored element. This is
    /// common for scalar tensors that are repeated.
    @inlinable public var isSingleElement: Bool { stridedSpanCount == 1 }
    
    //--------------------------------------------------------------------------
    /// init(
    /// Used to initialize an element collection subview
    @inlinable public init(
        shape: Shape,
        strides: Shape,
        count: Int,
        storage: StorageBufferType,
        storageBase: Int,
        stridedSpanCount: Int,
        layout: Layout,
        shared: Bool
    ) {
        self.shape = shape
        self.strides = strides
        self.count = count
        self.storage = storage
        self.storageBase = storageBase
        self.stridedSpanCount = stridedSpanCount
        self.isShared = shared
        self.layout = layout
        logicalStrides = shape.strides(for: layout)
        logicalElements = LogicalElements(count,
                                          shape,
                                          strides,
                                          storage,
                                          storageBase,
                                          layout,
                                          stridedSpanCount)
    }

    //--------------------------------------------------------------------------
    /// init(value:shape:layout:name
    /// Used to initialize a tensor with a single Element
    @inlinable public init(
        single value: TensorElement.Value,
        shape: Shape,
        layout: Layout,
        name: String
    ) {
        self.shape = shape
        self.strides = Shape.zero
        self.storageBase = 0
        self.isShared = false
        self.count = shape.elementCount()
        self.stridedSpanCount = 1
        self.layout = layout
        let stored = TensorElement.stored(value: value)
        self.storage = StorageBufferType(storedElement: stored, name: name)
        logicalStrides = shape.strides(for: layout)
        logicalElements = LogicalElements(count,
                                          shape,
                                          strides,
                                          storage,
                                          storageBase,
                                          layout,
                                          stridedSpanCount)
    }
}

//==============================================================================
/// Layout
/// Specifies how to store multi-dimensional data in row-major (C-style)
/// or column-major (Fortran-style) order in memory.
/// These names are following the numpy naming convention
public enum Layout: Int, Codable {
    /// Data is ordered in row-major dense sequential format.
    /// The leading dimension is the stride (in elements) to the beginning
    /// of next row in memory.
    case row
    
    /// Data is ordered in column-major dense sequential format.
    /// The leading dimension is the stride (in elements) to the beginning
    /// of next column in memory.
    case col
    
    public static let C = row, F = col, A = -1
    
    public static var defaultValue: Layout = Layout.row
}

public let _messageLayoutsMustMatch = "input layouts must match"

@inlinable public func layoutsMatch(_ layouts: Layout...) -> Bool {
    layouts.first(where: { $0 != layouts[0] }) == nil
}

//==============================================================================
/// DifferentiableTensor
///
/// While these protocols are not strictly necessary, they are used
/// to reduce the number of generic requirements when writing
/// `@differentiable` attributes
///
public protocol TensorProtocol: Logging {
    associatedtype Shape: TensorShape
    associatedtype TensorElement: StorageElement
}

public protocol DifferentiableTensor: TensorProtocol & Differentiable
where Self == TangentVector, TensorElement.Value: DifferentiableElement {}

/// DifferentiableElement
public protocol DifferentiableElement:
    Differentiable & Numeric where Self == TangentVector {}

extension Float: DifferentiableElement {}
extension Double: DifferentiableElement {}

extension Complex: DifferentiableElement
where RealType: Differentiable, RealType.TangentVector == RealType {}

// Differentiable conformance
extension Tensor: Differentiable & DifferentiableTensor
    where Element: DifferentiableElement
{
    public typealias TangentVector = Self
}

extension Tensor: AdditiveArithmetic where Element: Numeric {
    @inlinable public static var zero: Self { Tensor(0, name: "Zero") }
    @inlinable public static var one: Self { Tensor(1, name: "One") }
}

//==============================================================================
// Tensor Codable
public enum TensorCodingKeys: String, CodingKey {
    case data, shape, name, layout
}

extension Tensor: Codable where Element: Codable {
    /// encodes the contents of the array
    @inlinable public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: TensorCodingKeys.self)
        try container.encode(storage.name, forKey: .name)
        try container.encode(shape, forKey: .shape)
        try container.encode(layout, forKey: .layout)
        var dataContainer = container.nestedUnkeyedContainer(forKey: .data)
        if isBufferIterable {
            try self.buffer.forEach {
                try dataContainer.encode($0)
            }
        } else {
            try self.forEach {
                try dataContainer.encode($0)
            }
        }
    }
    
    @inlinable public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TensorCodingKeys.self)
        let name = try container.decode(String.self, forKey: .name)
        let shape = try container.decode(Shape.self, forKey: .shape)
        let layout = try container.decode(Layout.self, forKey: .layout)
        var dataContainer = try container.nestedUnkeyedContainer(forKey: .data)
        self = Self(shape: shape, layout: layout)
        self.name = name

        assert(self.count == dataContainer.count)
        var buffer = self.mutableBuffer
        for i in buffer.indices {
            buffer[i] = try dataContainer.decode(Element.self)
        }
    }
}

//==============================================================================
/// ElementIndex
/// Common index type used to iterate through collection elements
/// `position` is the index position in n-dimensional space
/// `sequencePosition` is the linear sequence position when iterating
/// and used for comparison
public struct ElementIndex<Shape>: Comparable, Codable
    where Shape: TensorShape
{
    /// the logical position along each axis
    public let position: Shape
    /// linear sequence position
    public let sequencePosition: Int

    // init(position:sequencePosition:
    @inlinable public init(_ position: Shape, _ sequencePosition: Int) {
        self.position = position
        self.sequencePosition = sequencePosition
    }

    /// init(sequencePosition:
    /// initializer for collections that ignore logical position
    @inlinable public init(at sequencePosition: Int) {
        self.position = Shape.zero
        self.sequencePosition = sequencePosition
    }

    /// incremented(lower:upper:
    /// increments `position` with the range `lower..<upper`
    @inlinable
    public func incremented(between lower: Self, and upper: Self) -> Self {
        let pos = position.incremented(between: lower.position,
                                       and: upper.position)
        return ElementIndex(pos, sequencePosition + 1)
    }
    
    @inlinable public func linearIndex(_ strides: Shape) -> Int {
        position.index(stridedBy: strides)
    }

    // Equatable
    @inlinable public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.sequencePosition == rhs.sequencePosition
    }
    
    // Comparable
    @inlinable public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.sequencePosition < rhs.sequencePosition
    }
}

//==============================================================================
// Tensor collection and sub view extensions
public extension Tensor {
    @inlinable var isBufferIterable: Bool {
        isSingleElement || isContiguous
    }
    
    //--------------------------------------------------------------------------
    // sequential buffer element iterators
    @inlinable var buffer: BufferElements<Shape,TensorElement> {
        BufferElements(tensor: self)
    }
    
    @inlinable var mutableBuffer: BufferElements<Shape,TensorElement> {
        mutating get { BufferElements(tensor: &self) }
    }
    
    //--------------------------------------------------------------------------
    // logical coordinate element iterators
    @inlinable var elements: LogicalElements<Shape,TensorElement> {
        logicalElements.synchronizeForRead()
        return logicalElements
    }
    
    @inlinable var mutableElements: LogicalElements<Shape,TensorElement> {
        mutating get {
            logicalElements.synchronizeForReadWrite()
            return logicalElements
        }
    }

    //--------------------------------------------------------------------------
    /// the starting index zero relative to the storage buffer
    @inlinable var startIndex: Index {
        logicalElements.synchronizeForReadWrite()
        return logicalElements.startIndex
    }
    
    //--------------------------------------------------------------------------
    /// the ending index zero relative to the storage buffer
    @inlinable var endIndex: Index {
        logicalElements.endIndex
    }

    //--------------------------------------------------------------------------
    /// makeIndex(position:
    /// makes an index from a logical position within `shape`
    /// - Parameters:
    ///  - position: the n-dimensional coordinate position within `shape`
    /// - Returns: the index
    @inlinable func makeIndex(at position: Shape) -> Index {
        Index(position, position.index(stridedBy: logicalStrides))
    }

    //--------------------------------------------------------------------------
    /// index(i:
    @inlinable func index(after i: Index) -> Index {
        logicalElements.index(after: i)
    }

    //--------------------------------------------------------------------------
    // elemment subscript
    @inlinable subscript(i: Index) -> Element {
        get {
            logicalElements.synchronizeForRead()
            return logicalElements[i]
        }
        set {
            prepareForWrite(using: Context.currentQueue)
            logicalElements.synchronizeForReadWrite()
            logicalElements[i] = newValue
        }
    }

    //--------------------------------------------------------------------------
    // sub view subscript
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable subscript(lower: Shape, upper: Shape) -> Self {
        get { createView(lower, upper, isShared) }
        set {
            prepareForWrite(using: Context.currentQueue)
            var view = createView(lower, upper, true)
            copy(from: newValue, to: &view)
        }
    }
    
    //--------------------------------------------------------------------------
    // creates a tensor subview
    @inlinable func createView(
        _ lower: Shape,
        _ upper: Shape,
        _ share: Bool
    ) -> Self {
        let shape = upper &- lower
        let count = shape.elementCount()
        let spanCount = strides.areSequential(for: shape) ? count :
                shape.spanCount(stridedBy: strides)

        return Tensor(
            shape: shape,
            strides: strides,
            count: count,
            storage: storage,
            storageBase: storageBase + lower.index(stridedBy: strides),
            stridedSpanCount: spanCount,
            layout: layout,
            shared: share)
    }

    //--------------------------------------------------------------------------
    /// `prepareForWrite`
    /// called before a write operation to ensure that the storage buffer
    /// is unique for this tensor unless it `isShared`
    /// It also expands repeated tensors to a full dense storage
    /// representation for write, which most often happens via element
    /// subscripting.
    @inlinable mutating func prepareForWrite(using queue: DeviceQueue) {
        // if repeated then expand to full dense tensor
        if stridedSpanCount < count {
            var expanded = Tensor(like: self)

            diagnostic(
                "\(expandingString) \(name)(\(id)) " +
                    "\(Element.self)[\(stridedSpanCount)] to: \(expanded.name)"
                    + "(\(expanded.id)) \(Element.self)[\(expanded.count)]",
                categories: [.dataCopy, .dataExpanding])

            // do an indexed copy
            copy(from: self, to: &expanded)
            self = expanded

        } else if !(isKnownUniquelyReferenced(&storage) || isShared) {
            // if not uniquely held then copy before creating the shared view
            diagnostic("\(mutationString) \(storage.name)(\(storage.id)) " +
                        "\(Element.self)[\(count)]",
                       categories: [.dataCopy, .dataMutation])
            
            storage = StorageBufferType(copying: storage, using: queue)
            logicalElements = LogicalElements(tensor: self)
        }
    }

    //--------------------------------------------------------------------------
    /// - Returns: the collection elements as a 1D Swift array
    @inlinable var flatArray: [Element] {
        usingSyncQueue {
            isBufferIterable ? [Element](buffer) : [Element](elements)
        }
    }
}

//==============================================================================
/// Derivative registration
extension Tensor where TensorElement.Value: DifferentiableElement {
    // https://github.com/apple/swift/blob/37b507b31c77ef969151f385cd1902dd44fb3b7f/stdlib/public/core/Array.swift#L2091
    
    @derivative(of: subscript)
    @inlinable func _vjpSubscript(lower: Shape, upper: Shape)
        -> (value: Self, pullback: (Self) -> Self)
    {
        return (self[lower, upper], { v in
            var result = zeros(like: self)
            result[lower, upper] = v
            return result
        })
    }
}

//==============================================================================
// Tensor read write access
public extension Tensor {
    //--------------------------------------------------------------------------
    /// `read`
    /// Synchronizes the collection of stored elements with the caller
    /// for reading. This function blocks until the elements are available.
    /// `Elements` are accessed by the application using `Collection`
    /// enumeration via `indices` or integer subscripting.
    @inlinable func read() -> UnsafeBufferPointer<TensorElement.Stored> {
        read(using: Context.syncQueue)
    }
    
    //--------------------------------------------------------------------------
    /// `read(queue:
    /// Synchronizes the collection of elements for reading
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    @inlinable func read(
        using queue: DeviceQueue
    ) -> UnsafeBufferPointer<TensorElement.Stored> {
        let (i, storedCount) = TensorElement
                .storedRange(start: storageBase, count: stridedSpanCount)

        return storage.read(type: TensorElement.Stored.self,
                            at: i, count: storedCount, using: queue)
    }

    //--------------------------------------------------------------------------
    /// `deviceRead(queue:
    /// Synchronizes the collection of elements for reading
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    @inlinable func deviceRead(using queue: DeviceQueue) -> UnsafeRawPointer {
        UnsafeRawPointer(read(using: queue).baseAddress!)
    }
    
    //--------------------------------------------------------------------------
    /// `readWrite`
    /// Synchronizes the collection of elements with the caller for read write
    /// This function blocks until the elements are available.
    /// `Elements` are accessed by the application using `MutableCollection`
    /// enumeration via `indices` or subscripting.
    @inlinable mutating func readWrite()
        -> UnsafeMutableBufferPointer<TensorElement.Stored> {
        readWrite(using: Context.syncQueue)
    }
    
    //--------------------------------------------------------------------------
    /// `readWrite(queue:`
    /// Synchronizes the collection of elements with the caller for read write
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    @inlinable mutating func readWrite(using queue: DeviceQueue)
    -> UnsafeMutableBufferPointer<TensorElement.Stored>
    {
        prepareForWrite(using: queue)

        let (i, storedCount) = TensorElement
                .storedRange(start: storageBase, count: stridedSpanCount)
        
        return storage.readWrite(type: TensorElement.Stored.self,
                                 at: i, count: storedCount, using: queue)
    }
    
    //--------------------------------------------------------------------------
    /// `deviceReadWrite(queue:`
    /// Synchronizes the collection of elements with the caller for read write
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    @inlinable mutating func deviceReadWrite(
        using queue: DeviceQueue
    ) -> UnsafeMutableRawPointer {
        UnsafeMutableRawPointer(readWrite(using: queue).baseAddress!)
    }
}

//==============================================================================
// Tensor element properties
public extension Tensor {
    /// first
    /// - Returns: the first element in the tensor
    @inlinable var first: Element {
        TensorElement.getValue(from: read(), at: storageBase)
    }

    /// element
    /// can get and set the value of a single element tensor.
    /// - Returns: the only element in the tensor
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable var element: Element {
        get {
            assert(count == 1, "the `element` property expects " +
                "the tensor to have a single Element. Use `first` for sets")
            return TensorElement.getValue(from: read(), at: storageBase)
        }
        set {
            assert(count == 1, "the `element` property expects " +
                "the tensor to have a single Element")
            TensorElement.set(value: newValue, in: readWrite(), at: storageBase)
        }
    }

    @derivative(of: element)
    @inlinable func vjpElement() -> (
      value: Element,
      pullback: (Element) -> Self
    ) where Element: DifferentiableElement {
      (element, { v in
        var result = zeros(like: self)
        result.element = v
        return result
      })
    }
}


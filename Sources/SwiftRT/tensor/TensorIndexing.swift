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

//==========================================================================
public protocol TensorIndexing: Strideable {
    associatedtype Position
    /// sequential logical view element index
    var viewIndex: Int { get }
    /// linear data buffer element index
    var dataIndex: Int { get }

    /// initializer for starting at any position
    init<T>(view: T, at position: Position) where T: TensorView
    /// initializer specifically for the endIndex
    init<T>(endOf view: T) where T: TensorView
    
    /// highest frequency function to move the index
    /// use advanced(by n: for jumps or negative movement
    func increment() -> Self
}

public extension TensorIndexing {
    // Equatable
    @inlinable @inline(__always)
    static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.viewIndex == rhs.viewIndex
    }
    
    // Comparable
    @inlinable @inline(__always)
    static func < (lhs: Self, rhs: Self) -> Bool {
        return lhs.viewIndex < rhs.viewIndex
    }
    
    @inlinable @inline(__always)
    func distance(to other: Self) -> Int {
        return other.viewIndex - viewIndex
    }
}

//==============================================================================
/// TensorValueCollection
public struct TensorValueCollection<View>: RandomAccessCollection
    where View: TensorView
{
    // properties
    public let view: View
    public let buffer: UnsafeBufferPointer<View.Element>
    public let startIndex: View.Index
    public let endIndex: View.Index
    public let count: Int

    public init(view: View, buffer: UnsafeBufferPointer<View.Element>) throws {
        self.view = view
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.shape.elementCount
    }

    //--------------------------------------------------------------------------
    // Collection
    @inlinable @inline(__always)
    public func index(before i: View.Index) -> View.Index {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    public func index(after i: View.Index) -> View.Index {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    public subscript(index: View.Index) -> View.Element {
        return buffer[index.dataIndex]
    }
}

//==============================================================================
/// TensorMutableValueCollection
public struct TensorMutableValueCollection<View>: RandomAccessCollection,
    MutableCollection where View: TensorView
{
    // properties
    public let buffer: UnsafeMutableBufferPointer<View.Element>
    public let startIndex: View.Index
    public let endIndex: View.Index
    public let count: Int
    
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<View.Element>) throws {
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.shape.elementCount
    }
    
    //--------------------------------------------------------------------------
    // Collection
    @inlinable @inline(__always)
    public func index(before i: View.Index) -> View.Index {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    public func index(after i: View.Index) -> View.Index {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    public subscript(index: View.Index) -> View.Element {
        get {
            return buffer[index.dataIndex]
        }
        set {
            buffer[index.dataIndex] = newValue
        }
    }
}

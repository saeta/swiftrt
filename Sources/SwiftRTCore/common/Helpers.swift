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

public typealias CStringPointer = UnsafePointer<CChar>

//==============================================================================
// clamping
extension Comparable {
    @inlinable public func clamped(to range: ClosedRange<Self>) -> Self {
        if (self > range.upperBound) {
            return range.upperBound
        } else if self < range.lowerBound {
            return range.lowerBound
        }
        return self
    }
}

//==============================================================================
// composing
public extension UInt64 {
    @inlinable init(msb: UInt32, lsb: UInt32) {
        self = (UInt64(msb) << 32) | UInt64(lsb)
    }

    @inlinable var split: (msb: UInt32, lsb: UInt32) {
        let mask: UInt64 = 0x00000000FFFFFFFF
        return (UInt32((self >> 32) & mask), UInt32(self & mask))
    }
}

//==============================================================================
// Memory sizes
extension Int {
    @inlinable public var KB: Int { self * 1024 }
    @inlinable public var MB: Int { self * 1024 * 1024 }
    @inlinable public var GB: Int { self * 1024 * 1024 * 1024 }
    @inlinable public var TB: Int { self * 1024 * 1024 * 1024 * 1024 }
}

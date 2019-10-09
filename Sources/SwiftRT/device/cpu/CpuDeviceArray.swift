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
public class CpuDeviceArray : DeviceArray {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public let buffer: UnsafeMutableRawBufferPointer
    public let device: ComputeDevice
    public var version = 0
    public let isReadOnly: Bool
    
    //--------------------------------------------------------------------------
	/// with count
	public init(device: ComputeDevice, count: Int) {
        self.device = device
        buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: count, alignment: MemoryLayout<Double>.alignment)
        self.isReadOnly = false
        self.trackingId = ObjectTracker.global.register(self)
	}

    //--------------------------------------------------------------------------
    /// readOnly uma buffer
    public init(device: ComputeDevice, buffer: UnsafeRawBufferPointer) {
        assert(buffer.baseAddress != nil)
        self.isReadOnly = true
        self.device = device
        let pointer = UnsafeMutableRawPointer(mutating: buffer.baseAddress!)
        self.buffer = UnsafeMutableRawBufferPointer(start: pointer,
                                                    count: buffer.count)
        self.trackingId = ObjectTracker.global.register(self)
    }

    //--------------------------------------------------------------------------
    /// readWrite uma buffer
    public init(device: ComputeDevice, buffer: UnsafeMutableRawBufferPointer) {
        self.isReadOnly = false
        self.device = device
        self.buffer = buffer
        self.trackingId = ObjectTracker.global.register(self)
    }

    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
}

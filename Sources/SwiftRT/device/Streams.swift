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

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

//==============================================================================
/// Executes a closure on the specified stream
/// - Parameter stream: the stream to set as the `currentStream`
/// - Parameter body: A closure whose operations are to be executed on the
///             specified stream
public func using<R>(_ stream: DeviceStream,
                     perform body: () throws -> R) rethrows -> R {
    // sets the default stream and logging info for the current scope
    _Streams.local.push(stream: stream)
    defer { _Streams.local.popStream() }
    // execute the body
    return try body()
}

//==============================================================================
/// _Streams
/// Manages the scope for the current stream, log, and error handlers
@usableFromInline
class _Streams {
    /// stack of default device streams, logging, and exception handler
    var streamStack: [DeviceStream]

    //--------------------------------------------------------------------------
    /// thread data key
    private static let key: pthread_key_t = {
        var key = pthread_key_t()
        pthread_key_create(&key) {
            #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
            let _: AnyObject = Unmanaged.fromOpaque($0).takeRetainedValue()
            #else
            let _: AnyObject = Unmanaged.fromOpaque($0!).takeRetainedValue()
            #endif
        }
        return key
    }()

    //--------------------------------------------------------------------------
    /// returns the thread local instance of the streams stack
    @usableFromInline
    static var local: _Streams {
        // try to get an existing state
        if let state = pthread_getspecific(key) {
            return Unmanaged.fromOpaque(state).takeUnretainedValue()
        } else {
            // create and return new state
            let state = _Streams()
            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
            return state
        }
    }

    //--------------------------------------------------------------------------
    /// current
    public static var current: DeviceStream {
        return _Streams.local.streamStack.last!
    }
    
    //--------------------------------------------------------------------------
    /// hostStream
    public static var hostStream: DeviceStream {
        return _Streams.current.device.memoryAddressing == .unified ?
            _Streams.current : _Streams._umaStream
    }
    
    private static var _umaStream: DeviceStream = {
        // create dedicated stream for app data transfer
        return try! Platform.local.createStream(
            deviceId: 0, serviceName: "cpu", name: "host", isStatic: true)
    }()
    
    //--------------------------------------------------------------------------
    /// auxHostStream
    // create dedicated stream for data transfer when accessing
    // within a stream closure or within HostMultiWrite
    public static var auxHostStream: DeviceStream = {
        return try! Platform.local.createStream(
            deviceId: 0, serviceName: "cpu", name: "dataSync", isStatic: true)
    }()
    
    //--------------------------------------------------------------------------
    /// logInfo
    // there will always be the platform default stream and logInfo
    public var logInfo: LogInfo { return streamStack.last!.logInfo }

    //--------------------------------------------------------------------------
    /// updateDefault
    public func updateDefault(stream: DeviceStream) {
        streamStack[0] = stream
    }
    
    //--------------------------------------------------------------------------
    // initializers
    private init() {
        do {
            // create the default stream based on service and device priority.
            let stream = try Platform.local.defaultDevice.createStream(
                    name: "default", isStatic: true)
            streamStack = [stream]
        } catch {
            print("Failed to create default streams")
            exit(1)
        }
    }
    
    //--------------------------------------------------------------------------
    /// push(stream:
    /// pushes the specified stream onto a stream stack which makes
    /// it the current stream used by operator functions
    @usableFromInline
    func push(stream: DeviceStream) {
        streamStack.append(stream)
    }
    
    //--------------------------------------------------------------------------
    /// popStream
    /// restores the previous current stream
    @usableFromInline
    func popStream() {
        assert(streamStack.count > 1)
        _ = streamStack.popLast()
    }
}

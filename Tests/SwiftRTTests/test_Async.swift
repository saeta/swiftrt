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
import XCTest
import Foundation

@testable import SwiftRT

class test_Async: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_hostMultiWrite", test_hostMultiWrite),
//        ("test_defaultStreamOp", test_defaultStreamOp),
//        ("test_secondaryDiscreetMemoryStream", test_secondaryDiscreetMemoryStream),
//        ("test_threeStreamInterleave", test_threeStreamInterleave),
//        ("test_tensorReferenceBufferSync", test_tensorReferenceBufferSync),
//        ("test_temporaryStreamShutdown", test_temporaryStreamShutdown),
//        ("test_StreamEventWait", test_StreamEventWait),
//        ("test_perfCreateStreamEvent", test_perfCreateStreamEvent),
//        ("test_perfRecordStreamEvent", test_perfRecordStreamEvent),
    ]

    //==========================================================================
    // test_hostMultiWrite
    // accesses a tensor on the host by dividing the first dimension
    // into batches and concurrently executing a user closure for each batch
    func test_hostMultiWrite() {
        do {
            Platform.log.level = .diagnostic
            typealias Pixel = RGB<UInt8>
            typealias ImageSet = Volume<Pixel>
            let expected = Pixel(0, 127, 255)
            var trainingSet = ImageSet((100, 256, 256))

            try trainingSet.hostMultiWrite { batch in
                for i in 0..<batch.items {
                    // get a view of the item at `i`
                    var itemView = batch.view(item: i)
                    
                    // get a writable buffer for the view
                    let buffer = itemView.hostMultiWriteBuffer()
                    
                    // at this point load image data from a file or database,
                    // decompress, type convert, whatever is needed
                    // In this example we'll just fill the buffer with
                    // the `expected` value
                    buffer.initialize(repeating: expected)
                }
            }

            // check the last item to see if it contains the expected value
            let item = trainingSet.view(item: trainingSet.items - 1)
            let values = try item.array()
            XCTAssert(values[0] == expected)
        } catch {
            XCTFail(String(describing: error))
        }
        
        // check for object leaks
        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_defaultStreamOp
    // initializes two matrices and adds them together
    func test_defaultStreamOp() {
        do {
            Platform.log.level = .diagnostic

            let m1 = Matrix<Int32>((2, 5), name: "m1", any: 0..<10)
            let m2 = Matrix<Int32>((2, 5), name: "m2", any: 0..<10)
            let result = m1 + m2
            let values = try result.array()
            
            let expected = (0..<10).map { Int32($0 * 2) }
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_secondaryDiscreetMemoryStream
    // initializes two matrices on the app thread, executes them on `stream1`,
    // the retrieves the results
    func test_secondaryDiscreetMemoryStream() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .scheduling, .streamSync]

            // create a named stream on a discreet device
            // cpuUnitTest device 1 is a discreet memory versions for testing
            let stream1 = try Platform.local
                .createStream(deviceId: 1, serviceName: "cpuUnitTest")
            
            let m1 = Matrix<Int32>((2, 5), name: "m1", any: 0..<10)
            let m2 = Matrix<Int32>((2, 5), name: "m2", any: 0..<10)

            // perform on user provided discreet memory stream
            let result = using(stream1) { m1 + m2 }

            // synchronize with host stream and retrieve result values
            let values = try result.array()
            
            let expected = (0..<10).map { Int32($0 * 2) }
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_threeStreamInterleave
    func test_threeStreamInterleave() {
        do {
            Platform.log.level = .diagnostic
            
            // create named streams on two discreet devices
            // cpuUnitTest device 1 and 2 are discreet memory versions
            let stream1 = try Platform.local
                .createStream(deviceId: 1, serviceName: "cpuUnitTest")
            let stream2 = try Platform.local
                .createStream(deviceId: 2, serviceName: "cpuUnitTest")

            let m1 = Matrix<Int32>((2, 3), name: "m1", any: 0..<6)
            let m2 = Matrix<Int32>((2, 3), name: "m2", any: 0..<6)
            let m3 = Matrix<Int32>((2, 3), name: "m3", any: 0..<6)

            // sum the values with a delay on device 1
            let sum_m1m2: Matrix<Int32> = using(stream1) {
                delayStream(atLeast: 0.1)
                return m1 + m2
            }

            // multiply the values on device 2
            let result = using(stream2) {
                sum_m1m2 * m3
            }

            // synchronize with host stream and retrieve result values
            let values = try result.array()
            
            let expected = (0..<6).map { Int32(($0 + $0) * $0) }
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_tensorReferenceBufferSync
    func test_tensorReferenceBufferSync() {
    }

    //==========================================================================
    // test_temporaryStreamShutdown
    func test_temporaryStreamShutdown() {
        do {
            Platform.log.level = .diagnostic
            
            for i in 0..<1000 {
                // create a matrix without any storage allocated
                var matrix = Matrix<Int32>((3, 4))
                
                // create a stream just for this closure
                // it will probably try to deinit before `fillWithIndex` is
                // complete
                using(try Platform.local.createStream()) {
                    fillWithIndex(&matrix)
                }
                
                // synchronize with host stream and retrieve result values
                let values = try matrix.array()
                let expected = [Int32](0..<12)
                XCTAssert(values == expected, "iteration: \(i) failed")
            }
        } catch {
            XCTFail(String(describing: error))
        }
        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }

    //==========================================================================
    // test_StreamEventWait
    func test_StreamEventWait() {
        do {
            Platform.log.level = .diagnostic
            Platform.local.log.categories = [.streamSync]
            
            let stream = try Platform.local
                    .createStream(serviceName: "cpuUnitTest")
            let event = try stream.createEvent()
            stream.delayStream(atLeast: 0.001)
            try stream.record(event: event).wait()
            XCTAssert(event.occurred, "wait failed to block")
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_perfCreateStreamEvent
    // measures the event overhead of creating 10,000 events
    func test_perfCreateStreamEvent() {
        #if !DEBUG
        do {
            let stream = try Platform.local.createStream()
            self.measure {
                do {
                    for _ in 0..<10000 {
                        _ = try stream.createEvent()
                    }
                } catch {
                    XCTFail(String(describing: error))
                }
            }
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }

    //==========================================================================
    // test_perfRecordStreamEvent
    // measures the event overhead of processing 10,000 tensors
    func test_perfRecordStreamEvent() {
        #if !DEBUG
        do {
            let stream = try Platform.local.createStream()
            self.measure {
                do {
                    for _ in 0..<10000 {
                        _ = try stream.record(event: stream.createEvent())
                    }
                    try stream.waitUntilStreamIsComplete()
                } catch {
                    XCTFail(String(describing: error))
                }
            }
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }
}

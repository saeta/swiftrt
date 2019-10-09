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

/// `Zip3Sequence` iterator
public struct Zip3Iterator<I1, I2, I3> where
I1: IteratorProtocol, I2: IteratorProtocol, I3: IteratorProtocol {
    var _I1: I1
    var _I2: I2
    var _I3: I3
    
    init(_ I1: I1, _ I2: I2, _ I3: I3) {
        self._I1 = I1
        self._I2 = I2
        self._I3 = I3
    }
}

extension Zip3Iterator: IteratorProtocol {
    public typealias Element = (I1.Element, I2.Element, I3.Element)
    
    public mutating func next() -> Zip3Iterator.Element? {
        guard let next1 = _I1.next(),
            let next2 = _I2.next(),
            let next3 = _I3.next() else {
                return nil
        }
        return (next1, next2, next3)
    }
}

public struct Zip3Sequence<S1, S2, S3> where
S1: Sequence, S2: Sequence, S3: Sequence {
    let _s1: S1
    let _s2: S2
    let _s3: S3
    
    init(_ s1: S1, _ s2: S2, _ s3: S3) {
        self._s1 = s1
        self._s2 = s2
        self._s3 = s3
    }
}

extension Zip3Sequence : Sequence {
    public typealias Iterator = Zip3Iterator<S1.Iterator, S2.Iterator, S3.Iterator>
    
    public func makeIterator() -> Zip3Sequence.Iterator {
        return Zip3Iterator.init(_s1.makeIterator(), _s2.makeIterator(),
                                 _s3.makeIterator())
    }
    
    func reduce<Result>(
        _ initialResult: Result,
        _ nextPartialResult: (Result, (S1.Element, S2.Element, S3.Element)) throws -> Result) rethrows -> Result
    {
        var result = initialResult
        for value in self {
            result = try nextPartialResult(result, value)
        }
        return result
    }
}

public func zip<S1, S2, S3>(_ s1: S1, _ s2: S2, _ s3: S3) ->
    Zip3Sequence<S1, S2, S3> where S1 : Sequence, S2 : Sequence, S3: Sequence {
    return Zip3Sequence(s1, s2, s3)
}
